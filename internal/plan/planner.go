package plan

import (
	"errors"
	"fmt"
	"sort"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/ngram"
)

// Model is what gets planned: the measured inventory of the target and of
// its drafter, if it has one.
type Model struct {
	ID      string
	Target  gguf.Inventory
	Drafter *gguf.Inventory
}

// ModelFrom builds a Model from the per-file inventories of a catalog entry
// (registry.Manifest.Inventory): the "model" file is the target, the
// "drafter" file, if any, the drafter.
func ModelFrom(id string, files []gguf.Inventory) (Model, error) {
	m := Model{ID: id}
	found := false
	for i := range files {
		switch files[i].Role {
		case "model":
			m.Target, found = files[i], true
		case "drafter":
			d := files[i]
			m.Drafter = &d
		}
	}
	if !found {
		return m, fmt.Errorf("%s has no measured model file; run `nanollm models inspect --tensors`", id)
	}
	return m, nil
}

// Request is one planning question. Zero values mean "the planner decides".
type Request struct {
	Model  Model
	Preset Preset
	// Device is the accelerator; nil plans for SyntheticDevice(Preset).
	Device *Device
	// Mode forces one mode instead of trying PlannerOrder.
	Mode Mode
	// KVType is the cache type for K and V: f16 (default) or q8_0.
	KVType string
	// NCtx (per slot) and Slots fix the context instead of growing it toward
	// the preset's target.
	NCtx  int
	Slots int
	// NoMTP plans without the drafter; NoMTPReason says why, for the plan.
	NoMTP       bool
	NoMTPReason string
	// DraftMax bounds MTP drafts per step; 0 is DefaultDraftMax.
	DraftMax  int
	DraftPMin float64
	// NGram turns on prompt-lookup drafting with this match length (OPT-5);
	// an MTP drafter, when the plan runs one, takes precedence.
	NGram int
	// NoFlashAttention plans for a device without it.
	NoFlashAttention bool
	// NoRepack plans without the CPU's repacked weights where a mode would
	// use them: M7 then range-locks the file mapping of its core (its
	// behaviour before OPT-1), several times slower to decode.
	NoRepack bool
	// NoSWAFull keeps the sliding-window layers at their window even when the
	// full cache would fit: attention over 512 positions instead of the whole
	// context, at the price of prefix reuse only within the window (a
	// measurement setting, the default is to spend a spare budget on the full
	// cache).
	NoSWAFull bool
	// Proven is the table of modes the planner may pick; nil is DefaultProven.
	Proven ProvenModes
	// NBatch and NUBatch default to DefaultNBatch and DefaultNUBatch.
	NBatch  int
	NUBatch int
}

// Defaults for what a request leaves open.
const (
	// DefaultDraftMax is provisional: the O3 sweeps replace it per device
	// class (Manifest.Drafter.SpecDraftNMax).
	DefaultDraftMax = 3
	DefaultNBatch   = 2048
	DefaultNUBatch  = 512
)

// Make plans a request. An error means the request itself is wrong (a
// research mode, a quantized V without flash attention, a model with no
// inventory); a request that simply does not fit returns a plan with Refused
// set and the shortfall of every mode it tried.
func Make(req Request) (*Plan, error) {
	if len(req.Model.Target.Tensors) == 0 {
		return nil, fmt.Errorf("%s has no measured tensor inventory; run `nanollm models inspect --tensors`", req.Model.ID)
	}
	h := req.Model.Target.HParams
	if h.BlockCount == 0 || h.VocabSize == 0 {
		return nil, fmt.Errorf("%s: inventory has no hyperparameters", req.Model.ID)
	}
	kvType := req.KVType
	if kvType == "" {
		kvType = "f16"
	}
	kt, err := gguf.ParseType(kvType)
	if err != nil {
		return nil, err
	}
	if kt != gguf.TypeF16 && kt != gguf.TypeQ8_0 && kt != gguf.TypeBF16 && kt != gguf.TypeF32 {
		return nil, fmt.Errorf("KV type %s is not supported; use f16 or q8_0", kvType)
	}
	fa := !req.NoFlashAttention
	// llama.cpp refuses a quantized V cache without flash attention.
	if !fa && kt.BlockSize() > 1 {
		return nil, fmt.Errorf("a %s V cache needs flash attention", kvType)
	}
	if req.Mode != "" && req.Mode != ModeAuto {
		info, ok := Info(req.Mode)
		if !ok {
			return nil, fmt.Errorf("unknown mode %q", req.Mode)
		}
		if info.Stage == StageResearch {
			return nil, fmt.Errorf("%s (%s) is a research mode; the planner does not place it", info.ID, info.Mode)
		}
	}
	if req.NCtx < 0 || req.Slots < 0 {
		return nil, errors.New("negative n_ctx or slots")
	}

	pl := &planner{req: req, h: h, kvType: kt, fa: fa}
	pl.dev = req.Device
	if pl.dev == nil {
		pl.dev = SyntheticDevice(req.Preset)
	}
	pl.proven = req.Proven
	if pl.proven == nil {
		pl.proven = DefaultProven
	}
	pl.nBatch, pl.nUBatch = req.NBatch, req.NUBatch
	if pl.nBatch <= 0 {
		pl.nBatch = DefaultNBatch
	}
	if pl.nUBatch <= 0 {
		pl.nUBatch = min(DefaultNUBatch, pl.nBatch)
	}
	pl.draftMax = req.DraftMax
	if pl.draftMax <= 0 {
		pl.draftMax = DefaultDraftMax
	}
	pl.mtp = req.Model.Drafter != nil && !req.NoMTP
	if req.NGram > 0 && !pl.mtp {
		pl.ngram = max(req.NGram, ngram.MinN)
	}
	return pl.run(), nil
}

type planner struct {
	req      Request
	h        gguf.HParams
	kvType   gguf.Type
	fa       bool
	dev      *Device
	proven   ProvenModes
	nBatch   int
	nUBatch  int
	draftMax int
	mtp      bool
	ngram    int // prompt-lookup match length; 0 when off or MTP runs
}

// shape is what the planner grows: context per slot, slots, and the
// sliding-window decision.
type shape struct {
	nCtx, slots int
	swaFull     bool
}

func (pl *planner) run() *Plan {
	modes := PlannerOrder
	if pl.req.Mode != "" && pl.req.Mode != ModeAuto {
		modes = []Mode{pl.req.Mode}
	}
	target := TargetFor(pl.req.Preset.Name)
	floor := shape{nCtx: FloorNCtx, slots: FloorSlots}
	// A fixed value is both floor and target: the planner neither grows it
	// nor settles for less.
	if pl.req.NCtx > 0 {
		floor.nCtx, target.NCtx = pl.req.NCtx, pl.req.NCtx
	}
	if pl.req.Slots > 0 {
		floor.slots, target.Slots = pl.req.Slots, pl.req.Slots
	}

	var tried []Attempt
	for _, m := range modes {
		info, _ := Info(m)
		a := Attempt{Mode: m, ID: info.ID}
		if why := pl.applies(m); why != "" {
			a.Skipped = why
			tried = append(tried, a)
			continue
		}
		if why := pl.proven.why(m); why != "" {
			a.Skipped = why
			tried = append(tried, a)
			continue
		}
		t, short := pl.fitFloor(m, floor)
		if t == nil {
			a.Shortfall = &short
			tried = append(tried, a)
			continue
		}
		a.Fits = true
		if m == VRAMPartial {
			a.LayersOnGPU = t.n
		}
		tried = append(tried, a)
		t = pl.grow(t, floor, target)
		p := pl.finish(t, target)
		p.Tried = tried
		return p
	}

	p := &Plan{
		Model: pl.req.Model.ID, Preset: pl.req.Preset.Name, Device: pl.dev,
		Target: target, Tried: tried, FlashAttention: pl.fa,
		NBatch: pl.nBatch, NUBatch: pl.nUBatch,
	}
	p.Refused = refusal(tried, floor)
	return p
}

// refusal summarizes why nothing fit, smallest shortfall first: that is the
// mode closest to fitting and the one a user should read about.
func refusal(tried []Attempt, floor shape) string {
	type miss struct {
		id    string
		bytes int64
	}
	var misses []miss
	for _, a := range tried {
		if a.Shortfall != nil {
			misses = append(misses, miss{a.ID, a.Shortfall.VRAM + a.Shortfall.RAM + a.Shortfall.Unified})
		}
	}
	if len(misses) == 0 {
		return "no mode applies to this preset"
	}
	sort.SliceStable(misses, func(i, j int) bool { return misses[i].bytes < misses[j].bytes })
	return fmt.Sprintf("nothing fits at n_ctx %d × %d slots; closest is %s, short by %s",
		floor.nCtx, floor.slots, misses[0].id, mibStr(misses[0].bytes))
}

// applies says why a mode cannot apply to this preset and device at all.
func (pl *planner) applies(m Mode) string {
	b := pl.req.Preset.Budget
	gpuMode := m == VRAMFull || m == VRAMPLEHost || m == VRAMPLEDisk || m == VRAMPartial
	unified := pl.dev != nil && pl.dev.Unified
	switch {
	case gpuMode && (pl.dev == nil || b.VRAM == 0 && b.Unified == 0):
		return "the preset grants no device memory"
	case gpuMode && unified && m == VRAMPLEHost:
		return "the same as M1 on unified memory"
	case gpuMode && unified && m == VRAMPartial:
		return "no partial offload on unified memory"
	case !gpuMode && b.RAM == 0 && b.Unified == 0:
		return "the preset grants no host RAM"
	}
	if m == VRAMPartial && pl.h.BlockCount < 1 {
		return "the model has no repeating layers"
	}
	return ""
}

// fitFloor finds the configuration of m that fits at the floor: for M4 the
// largest N. It returns the shortfall of the closest miss otherwise.
func (pl *planner) fitFloor(m Mode, floor shape) (*trial, Shortfall) {
	if m != VRAMPartial {
		t := pl.evaluate(m, pl.h.BlockCount, floor)
		if t.short.zero() {
			return t, Shortfall{}
		}
		return nil, t.short
	}
	var last Shortfall
	for n := pl.h.BlockCount - 1; n >= 0; n-- {
		t := pl.evaluate(m, n, floor)
		if t.short.zero() {
			return t, Shortfall{}
		}
		last = t.short
	}
	return nil, last
}

// grow spends what the floor leaves on context, then slots, up to the target:
// the largest per-slot context first, because a chat that does not fit is
// worse than a second user waiting. Then it tries the full sliding-window
// cache at the chosen size.
func (pl *planner) grow(t *trial, floor shape, target Target) *trial {
	best := t
	for ctx := target.NCtx; ctx >= floor.nCtx; ctx /= 2 {
		found := false
		for slots := target.Slots; slots >= floor.slots; slots-- {
			s := shape{nCtx: ctx, slots: slots}
			if s == best.shape {
				found = true
				break
			}
			c := pl.evaluate(t.mode, t.n, s)
			if c.short.zero() {
				best, found = c, true
				break
			}
		}
		if found {
			break
		}
	}
	if pl.req.NoSWAFull {
		return best
	}
	full := best.shape
	full.swaFull = true
	if c := pl.evaluate(best.mode, best.n, full); c.short.zero() {
		return c
	}
	return best
}

// trial is one evaluated configuration.
type trial struct {
	mode  Mode
	n     int // repeating layers on the device
	shape shape
	spec  modeSpec

	groups      map[gguf.Group]Bytes
	cells       KVCells
	kvBytes     int64
	locks       []lockedTensor
	random      []gguf.Tensor // file-backed tensors read at random (G1)
	stream      []gguf.Tensor // file-backed layer tensors every token reads (OPT-8)
	repacked    []string      // target tensors overridden to the repacked buffer (M8)
	nOutputsMax int
	mem         Memory
	short       Shortfall
}

type lockedTensor struct {
	role string
	t    gguf.Tensor
	pad  int64
}

// modeSpec is how a mode places the model.
type modeSpec struct {
	gpu        bool // layers go to the device
	unified    bool // the device maps host memory (Metal)
	inputOnGPU bool // overrides move the input tensors into VRAM (M1)
	hostAnon   bool // host weights are read into anonymous memory (dio)
	pleFile    bool // G1 stays file-backed whatever the rest does
	repack     bool
	load       backend.LoadMode
	lazy       backend.LazyMode
	lockAll    bool // range-lock everything but G1 (M7)
	lockHot    bool // range-lock G0, G2, G4, G6 and what fits of G3 (M8)
	pleOnDisk  bool // M4 variant with host weights file-backed
}

func (pl *planner) spec(m Mode, n int) modeSpec {
	unified := pl.dev != nil && pl.dev.Unified
	switch m {
	case VRAMFull:
		if unified {
			// Metal maps the file without copying: one budget, nothing to
			// override, nothing read twice.
			return modeSpec{gpu: true, unified: true, load: backend.LoadMMap, lazy: backend.LazyOff}
		}
		return modeSpec{gpu: true, inputOnGPU: true, hostAnon: true, load: backend.LoadDirectIO, lazy: backend.LazyOff}
	case VRAMPLEHost:
		return modeSpec{gpu: true, hostAnon: true, load: backend.LoadDirectIO, lazy: backend.LazyOff}
	case VRAMPLEDisk:
		return modeSpec{gpu: true, unified: unified, pleFile: true, load: backend.LoadMMap, lazy: backend.LazyOn}
	case VRAMPartial:
		// PLE in host RAM when the preset budgets RAM and it fits, on disk
		// otherwise (SPEC §15, vram-2g). evaluate picks the variant.
		return modeSpec{gpu: true, hostAnon: true, load: backend.LoadDirectIO, lazy: backend.LazyOff}
	case RAMFull:
		return modeSpec{hostAnon: true, repack: true, load: backend.LoadDirectIO, lazy: backend.LazyOff}
	case RAMPLEDisk:
		if pl.req.NoRepack {
			return modeSpec{pleFile: true, lockAll: true, load: backend.LoadMMap, lazy: backend.LazyOn}
		}
		// OPT-1: the core is repacked (the kernels of the CPU backend need
		// the weights in their own layout, 4x faster to decode than the
		// file's), which copies it into anonymous memory at load. Nothing is
		// range-locked then: the copy is not file-backed, and the pages of
		// the mapping it was read from are free to go. The PLE table stays
		// file-backed.
		return modeSpec{pleFile: true, repack: true, load: backend.LoadMMap, lazy: backend.LazyOn}
	case RAMHybrid:
		return modeSpec{pleFile: true, lockHot: true, load: backend.LoadMMap, lazy: backend.LazyOn}
	default: // MMap
		return modeSpec{load: backend.LoadMMap, lazy: backend.LazyOff}
	}
}

func pleOnDiskSpec(s modeSpec) modeSpec {
	s.hostAnon, s.pleFile, s.pleOnDisk = false, true, true
	s.load, s.lazy = backend.LoadMMap, backend.LazyOn
	return s
}

// evaluate places the model in mode m with n layers on the device and checks
// the budgets. For M4 it tries PLE in host RAM first when the preset budgets
// RAM, then on disk.
func (pl *planner) evaluate(m Mode, n int, s shape) *trial {
	spec := pl.spec(m, n)
	if m == VRAMPartial {
		if pl.req.Preset.Budget.RAM > 0 {
			if t := pl.place(m, n, s, spec); t.short.zero() {
				return t
			}
		}
		return pl.place(m, n, s, pleOnDiskSpec(spec))
	}
	return pl.place(m, n, s, spec)
}
