// Package placement turns a model, a device and the configuration into a plan
// and a load: what the server runs, what `nanollm plan` shows and what the
// autotuner (internal/tune) builds its candidates from. Keeping it in one place
// means the autotuner plans exactly what the server will run.
//
// The settings come in layers (docs/OPTIMIZE.md, §3.7): a key in the
// configuration beats the autotuner's profile, which beats the limits of the
// mode, which beat the planner's static rules. Where a mode or a profile had a
// say, Result.Sources records which layer.
package placement

import (
	"cmp"
	"fmt"
	"reflect"
	"runtime"
	"strings"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/ngram"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

// DefaultNGramDraftMax is how many tokens a prompt-lookup draft carries when
// the configuration does not say: provisional until the sweep of docs/STAGE2.md
// 2.7 sets it.
const DefaultNGramDraftMax = 4

// Layers a setting can come from.
const (
	SourceConfig  = "config"
	SourceProfile = "profile"
	SourceMode    = "mode"
	SourceRule    = "rule"
)

// Tune is what the autotuner contributes to a placement: the limits of the
// mode and the settings of the profile. The zero value contributes nothing.
type Tune struct {
	// Source names the layer Preset, Threads, Override, MTP, DraftMax and
	// DraftPMin come from: profile or mode. MemCap and MinNCtx are always the
	// mode's.
	Source string
	// Preset is the memory budget to use when the configuration names none and
	// the device has at least that much free (the silent mode's ram-4g); the
	// derived preset of the device still wins when it is smaller.
	Preset string
	// MemCap and MinNCtx are plan.Request.MemCap and MinNCtx.
	MemCap  int64
	MinNCtx int
	// Threads and ThreadsBatch are the CPU teams; the configuration's
	// devices.cpu_threads beats them.
	Threads, ThreadsBatch int
	// Override fills the fields of the model's override that the configuration
	// leaves unset: mode, kv_type, repack, flash_attention, sampling, cuda_nano.
	Override config.ModelOverride
	// MTP, DraftMax and DraftPMin say whether the drafter runs and with what
	// limits; nil leaves the rule. A pointer, because a p_min of 0 is a value.
	MTP       *bool
	DraftMax  *int
	DraftPMin *float64
}

// Zero reports whether the tune contributes nothing.
func (t Tune) Zero() bool {
	return t.Preset == "" && t.MemCap == 0 && t.MinNCtx == 0 && t.Threads == 0 && t.ThreadsBatch == 0 &&
		t.MTP == nil && t.DraftMax == nil && t.DraftPMin == nil && reflect.DeepEqual(t.Override, config.ModelOverride{})
}

// Input is one placement question.
type Input struct {
	Manifest registry.Manifest
	Device   backend.Device
	// Override is the model's block of the configuration.
	Override config.ModelOverride
	// ModelPath and DrafterPath are the files on disk.
	ModelPath   string
	DrafterPath string
	// Threads and ThreadsBatch are devices.cpu_threads and cpu_threads_batch.
	Threads      int
	ThreadsBatch int
	Tune         Tune
	// GOOS selects the modes proven on the OS; empty is the running one.
	GOOS string
}

// Result is a resolved placement.
type Result struct {
	Plan   *plan.Plan
	Spec   backend.LoadSpec
	Preset plan.Preset
}

// Resolve plans the model on the device and builds the load. An error means
// the model does not fit the preset or the request is wrong.
func Resolve(in Input) (*Result, error) {
	model, err := plan.ModelFrom(in.Manifest.ID, in.Manifest.Inventory)
	if err != nil {
		return nil, err
	}
	src := map[string]string{}
	ov := in.Override
	tune := in.Tune
	from := cmp.Or(tune.Source, SourceMode)
	fill := func(name string, set bool, apply func()) {
		if set {
			apply()
			src[name] = from
		}
	}
	unset := func(s string) bool { return s == "" || s == "auto" }
	fill("mode", unset(ov.Mode) && !unset(tune.Override.Mode), func() { ov.Mode = tune.Override.Mode })
	fill("kv_type", ov.KVType == "" && tune.Override.KVType != "", func() { ov.KVType = tune.Override.KVType })
	fill("repack", unset(ov.Repack) && !unset(tune.Override.Repack), func() { ov.Repack = tune.Override.Repack })
	fill("flash_attention", unset(ov.FlashAttention) && !unset(tune.Override.FlashAttention), func() { ov.FlashAttention = tune.Override.FlashAttention })
	fill("sampling", unset(ov.Sampling) && !unset(tune.Override.Sampling), func() { ov.Sampling = tune.Override.Sampling })
	fill("cuda_nano", len(ov.CudaNano) == 0 && len(tune.Override.CudaNano) > 0, func() { ov.CudaNano = tune.Override.CudaNano })

	var preset plan.Preset
	explicit := false
	switch {
	case ov.Preset != "":
		if preset, err = plan.LookupPreset(ov.Preset); err != nil {
			return nil, err
		}
		explicit = true
		src["preset"] = SourceConfig
	default:
		if preset, err = plan.DerivedPreset(in.Device); err != nil {
			return nil, err
		}
		if tune.Preset != "" {
			// The mode's budget, unless the device offers less right now.
			if want, err := plan.ParsePreset(tune.Preset); err == nil && want.Budget.Total() < preset.Budget.Total() && sameKind(want, preset) {
				preset = want
				src["preset"] = from
			}
		}
	}
	mode, err := plan.ParseMode(ov.Mode)
	if err != nil {
		return nil, err
	}
	r := plan.Request{
		Model: model, Preset: preset, Mode: mode, KVType: ov.KVType, Proven: plan.ProvenFor(cmp.Or(in.GOOS, runtime.GOOS)),
		NCtx: ov.NCtx.N, Slots: ov.Slots.N,
		NoMTP: ov.Speculative.Type == config.SpecNone, DraftMax: ov.Speculative.NMax, DraftPMin: ov.Speculative.PMin,
		VerifyRows: ov.Speculative.VerifyRows, NoRepack: ov.Repack == "off",
	}
	// A preset in the configuration is a hard budget the user chose: the
	// mode's limit does not apply to that model.
	if !explicit && tune.MemCap > 0 {
		r.MemCap = tune.MemCap
		src["mem_cap"] = SourceMode
	}
	if tune.MinNCtx > 0 && ov.NCtx.IsAuto() {
		r.MinNCtx = tune.MinNCtx
		src["min_n_ctx"] = SourceMode
	}
	// MTP on the CPU: on by default where the device lists the multi-row Q4_0
	// kernels, off elsewhere with the reason in the plan (plan.DefaultCPUMTP).
	// speculative.type: mtp turns it on anywhere; none turns it off.
	if ov.Speculative.Type == "" && in.Device.Type == backend.CPU {
		r.DefaultCPUMTP(in.Device.Has(backend.FeatureMultiRowQ4_0))
	}
	// The bound on the rows spent on drafts follows the kernels, not how MTP was
	// asked for: speculative.type: mtp on a CPU with them drafts while one
	// sequence decodes, too (speculative.verify_rows overrides).
	if ov.Speculative.Type == config.SpecMTP && r.VerifyRows == 0 && in.Device.Type == backend.CPU && in.Device.Has(backend.FeatureMultiRowQ4_0) {
		r.VerifyRows = 1 + cmp.Or(r.DraftMax, plan.DefaultDraftMax)
	}
	if ov.Speculative.Type == config.SpecNGram {
		r.NoMTP, r.NoMTPReason = true, "prompt lookup (speculative.type: ngram) drafts instead"
		r.NGram = cmp.Or(ov.Speculative.Match, ngram.DefaultN)
		r.DraftMax = cmp.Or(ov.Speculative.NMax, DefaultNGramDraftMax)
	}
	// The profile's drafting settings beat the rules, but not the configuration.
	if ov.Speculative.Type == "" {
		if tune.MTP != nil {
			r.NoMTP = !*tune.MTP
			if r.NoMTP {
				r.NoMTPReason = "off by the " + from + " of the autotuner"
			} else {
				r.NoMTPReason = ""
			}
			src["mtp"] = from
		}
		if tune.DraftMax != nil && ov.Speculative.NMax == 0 {
			r.DraftMax = *tune.DraftMax
			src["draft_max"] = from
		}
		if tune.DraftPMin != nil && ov.Speculative.PMin == 0 {
			r.DraftPMin = *tune.DraftPMin
			src["draft_p_min"] = from
		}
	}
	if in.Device.Type != backend.CPU {
		r.FlashAttnSmallBatch = in.Device.Has(backend.FeatureFlashAttnSmallBatch)
		r.BackendSampling = in.Device.Has(backend.FeatureBackendSampling)
		r.Device = &plan.Device{
			Selector: in.Device.Selector(), Backend: plan.Kind(in.Device.ID),
			Memory: plan.DeviceMemory(in.Device), Unified: in.Device.Unified,
		}
	}
	switch ov.FlashAttention {
	case "on":
		r.FlashAttnSmallBatch = false
	case "off":
		r.NoFlashAttention = true
	case "small":
		r.FlashAttnSmallBatch = true
	}
	if len(ov.CudaNano) > 0 && in.Device.Has(backend.FeatureCudaNano) {
		r.CudaNano = make(map[string]int, len(ov.CudaNano))
		for name, v := range ov.CudaNano {
			r.CudaNano[strings.ToUpper(name)] = v
		}
	}
	switch ov.Sampling {
	case "host":
		r.BackendSampling = false
	case "device":
		r.BackendSampling = true
	}
	// OPT-4: the output shortlist, where the device's library has it. An
	// explicit setting the device cannot keep is not an error (the same
	// configuration serves machines with different libraries): the plan says so.
	var headNote string
	if ov.Head != "" && ov.Head != registry.HeadFull {
		switch {
		case in.Device.Type != backend.CPU:
			headNote = fmt.Sprintf("head: %s is a CPU option; the whole head runs on %s", ov.Head, in.Device.ID)
		case !in.Device.Has(backend.FeatureOutputShortlist):
			headNote = fmt.Sprintf("head: %s needs libraries with patch 0009 (device feature %s); the whole head runs", ov.Head, backend.FeatureOutputShortlist)
		default:
			ids, err := registry.ShortlistIDs(in.Manifest, ov.Head)
			if err != nil {
				return nil, err
			}
			r.Head, r.HeadIDs = ov.Head, ids
		}
	}
	p, err := plan.Make(r)
	if err != nil {
		return nil, err
	}
	if !p.OK() {
		return nil, fmt.Errorf("preset %s: %s", preset.Name, p.Refused)
	}
	if headNote != "" {
		p.Warnings = append(p.Warnings, headNote)
	}
	if in.DrafterPath == "" {
		p.MTP.Enabled = false
	}
	spec, err := p.LoadSpec(in.ModelPath, in.DrafterPath)
	if err != nil {
		return nil, err
	}
	if p.Device == nil {
		spec.Device = in.Device.Selector()
	}
	if spec.Stream != nil {
		spec.Stream.Ahead, spec.Stream.Keep, spec.Stream.Evict = ov.Stream.Ahead, ov.Stream.Keep, ov.Stream.Evict
		spec.Stream.Method, spec.Stream.Depth = ov.Stream.Method, ov.Stream.Depth
	}
	switch ov.Repack {
	case "off":
		spec.Repack = false
	case "on":
		spec.Repack = true // an expert setting: the budget of the modes that keep weights in the file is not kept
	}
	spec.Threads, spec.ThreadsBatch = in.Threads, in.ThreadsBatch
	if spec.Threads != 0 {
		src["threads"] = SourceConfig
	} else if tune.Threads > 0 && !spec.FullyOnGPU() {
		// A model fully on a GPU keeps no CPU team (LoadSpec.WithDefaultThreads):
		// the mode's limit has nothing to bound.
		spec.Threads, spec.ThreadsBatch = tune.Threads, cmp.Or(in.ThreadsBatch, tune.ThreadsBatch, tune.Threads)
		src["threads"] = from
	}
	spec = spec.WithDefaultThreads()
	if len(src) > 0 {
		p.Sources = src
	}
	return &Result{Plan: p, Spec: spec, Preset: preset}, nil
}

// sameKind reports whether two presets grant the same kinds of memory, so a
// ram- budget of a mode never replaces a vram- one.
func sameKind(a, b plan.Preset) bool {
	return (a.Budget.VRAM > 0) == (b.Budget.VRAM > 0) && (a.Budget.RAM > 0) == (b.Budget.RAM > 0) && (a.Budget.Unified > 0) == (b.Budget.Unified > 0)
}
