package tune

import (
	"cmp"
	"fmt"
	"sort"
	"strings"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/placement"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

// SilentBudgets are the CPU memory budgets the silent mode tries, smallest
// first (docs/OPTIMIZE.md, §3.2).
var SilentBudgets = []string{"ram-1g", "ram-1.5g", "ram-2g", "ram-3g", "ram-4g", "ram-6g", "ram-8g"}

// Candidate is a device with a memory budget, planned and ready to be measured.
type Candidate struct {
	ID     string `json:"id"`
	Device Device `json:"device"`
	// Preset is the budget the plan was made under (derived from the device for
	// the optimal and powerful modes).
	Preset string `json:"preset"`
	// Mode is the memory mode the planner chose (M1…M9 and its name).
	Mode  string           `json:"mode"`
	Plan  *plan.Plan       `json:"plan,omitempty"`
	Spec  backend.LoadSpec `json:"spec"`
	Class Class            `json:"class"`
	NCtx  int              `json:"n_ctx"`
	Slots int              `json:"slots"`
	MTP   bool             `json:"mtp"`
	// PlannedPrivate and PlannedVRAM are the plan's host memory and video
	// memory in bytes; PlannedBytes their sum, the number the mode's limit is
	// checked against.
	PlannedPrivate int64 `json:"planned_private"`
	PlannedVRAM    int64 `json:"planned_vram"`
	PlannedBytes   int64 `json:"planned_bytes"`
	// DriverHost is the part of PlannedPrivate that is not in the plan: the host
	// memory of the GPU driver's context (plan.GPUHostMemory).
	DriverHost int64    `json:"driver_host,omitempty"`
	Estimate   Estimate `json:"estimate"`
	// NUMA is the memory policy the candidate is measured (and applied) under
	// on a machine with several NUMA nodes: interleave or distribute; empty
	// elsewhere.
	NUMA string `json:"numa,omitempty"`
	// Reject says why the candidate is not measured; nil means it is a
	// contender.
	Reject *Rejection `json:"reject,omitempty"`
	// Measure says the candidate is to be measured (a contender within reach of
	// the best estimate, or one with no estimate).
	Measure bool `json:"measure,omitempty"`
}

// Rejection is the reason a candidate is out.
type Rejection struct {
	Code   string `json:"code"`
	Reason string `json:"reason"`
}

// Rejection codes.
const (
	RejectNoFit    = "no_fit"    // the planner could not place the model in the budget
	RejectVRAM     = "vram"      // more video memory than the device has free
	RejectRAM      = "ram"       // more host memory than is free
	RejectDisk     = "disk"      // the disk allows less than the minimum speed
	RejectSlower   = "slower"    // far behind the best estimate
	RejectNoDevice = "no_device" // the mode excludes the device
)

// PlanInput is what candidates are planned from.
type PlanInput struct {
	Manifest    registry.Manifest
	Override    config.ModelOverride
	ModelPath   string
	DrafterPath string
	// Threads and ThreadsBatch are devices.cpu_threads of the configuration.
	Threads      int
	ThreadsBatch int
	Matrix       *Matrix
	GOOS         string
}

// ClassOf is the hardware class of a device for the matrix: the CPU
// attributes for the CPU device, the GPU's for a GPU.
func ClassOf(s *Survey, d Device) Class {
	c := Class{AttrOS: s.OS.Name}
	if d.Type == string(backend.CPU) {
		c[AttrCPUArch] = map[string]string{"amd64": "x86_64", "arm64": "arm64"}[s.CPU.Arch]
		if s.CPU.Variant != "" {
			c[AttrCPUVariant] = s.CPU.Variant
		}
		c[AttrHybrid] = fmt.Sprint(s.CPU.Hybrid())
		c[AttrNUMA] = fmt.Sprint(max(s.CPU.NUMANodes, 1))
		return c
	}
	c[AttrGPUBackend] = d.Kind
	if d.Vendor != "" {
		c[AttrGPUVendor] = strings.ToLower(d.Vendor)
	}
	if d.Unified {
		c[AttrGPUMemory] = "shared"
	} else {
		c[AttrGPUMemory] = "own"
	}
	if d.CC > 0 {
		c[AttrGPUCC] = fmt.Sprint(d.CC)
	}
	return c
}

// backendDevice converts a surveyed device back for the planner.
func backendDevice(d Device) backend.Device {
	return backend.Device{
		Backend: d.Backend, ID: d.ID, Type: backend.DeviceType(d.Type), Name: d.Name,
		MemoryTotal: d.MemoryTotal, MemoryFree: d.MemoryFree, Unified: d.Unified, Features: d.Features,
	}
}

// BuildCandidates plans the devices the mode allows (docs/OPTIMIZE.md, §3.2)
// and prunes what cannot work: a plan the machine's free memory does not hold,
// a setup the disk cannot feed at the minimum speed, a candidate far behind
// the best estimate. The result is ordered by the mode's preference. Problems
// found on the way (low_free_ram) are returned too.
func BuildCandidates(s *Survey, k Constraints, in PlanInput) ([]*Candidate, []Problem) {
	if in.Matrix == nil {
		in.Matrix = Default
	}
	var cands []*Candidate
	var problems []Problem
	for _, d := range s.Devices {
		if k.CPUOnly && d.Type != string(backend.CPU) {
			continue
		}
		if d.Type == string(backend.NPU) {
			continue
		}
		if k.Mode == Silent {
			for _, preset := range SilentBudgets {
				cands = append(cands, planCandidate(s, k, in, d, preset))
			}
			continue
		}
		base := planCandidate(s, k, in, d, "")
		if d.Type == string(backend.CPU) && s.CPU.NUMANodes > 1 && base.Reject == nil {
			// Several nodes: where the pages go is a choice worth a load each
			// (EPYC 7351P: distribute is 23 % faster than interleave).
			for _, policy := range []string{"interleave", "distribute"} {
				c := *base
				c.NUMA, c.ID = policy, base.ID+"#"+policy
				cands = append(cands, &c)
			}
			continue
		}
		cands = append(cands, base)
	}
	dropSameCard(s, cands)
	problems = append(problems, prune(s, k, in, cands)...)
	order(cands, k)
	return cands, problems
}

// planCandidate plans one device; preset is empty for the derived budget.
func planCandidate(s *Survey, k Constraints, in PlanInput, d Device, preset string) *Candidate {
	c := &Candidate{Device: d, Class: ClassOf(s, d), Preset: preset}
	c.ID = d.Selector
	if preset != "" {
		c.ID += "/" + preset
	}
	ov := in.Override
	tune := ModeTune(k)
	if preset != "" {
		// A budget of the silent mode's ladder is the candidate itself, not a
		// default: it stands as if the configuration had set it.
		ov.Preset = preset
		tune.MemCap = 0
	}
	res, err := placement.Resolve(placement.Input{
		Manifest: in.Manifest, Device: backendDevice(d), Override: ov, ModelPath: in.ModelPath, DrafterPath: in.DrafterPath,
		Threads: in.Threads, ThreadsBatch: in.ThreadsBatch, Tune: tune, GOOS: in.GOOS,
	})
	if err != nil {
		c.Reject = &Rejection{RejectNoFit, err.Error()}
		return c
	}
	c.Plan, c.Spec = res.Plan, res.Spec
	if preset == "" {
		c.Preset = res.Preset.Name
	}
	c.Mode, c.NCtx, c.Slots, c.MTP = res.Plan.ModeID+" "+string(res.Plan.Mode), res.Plan.NCtx, res.Plan.Slots, res.Plan.MTP.Enabled
	if m := res.Plan.Memory; m.Unified != nil {
		c.PlannedPrivate = m.Unified.Used
	} else {
		if m.RAM != nil {
			c.PlannedPrivate = m.RAM.Used
		}
		if m.VRAM != nil {
			c.PlannedVRAM = m.VRAM.Used
		}
	}
	if d.Type == string(backend.GPU) && (d.Kind == "cuda" || d.Kind == "vulkan") {
		c.DriverHost = plan.GPUHostMemory // what a process with a GPU context holds on the host beyond the plan
		c.PlannedPrivate += c.DriverHost
	}
	c.PlannedBytes = c.PlannedPrivate + c.PlannedVRAM
	return c
}

// ModeTune is the autotuner's contribution to a placement without a profile:
// the limits of the mode.
func ModeTune(k Constraints) placement.Tune {
	t := placement.Tune{Source: placement.SourceMode, MinNCtx: k.MinNCtx, Threads: k.Threads, ThreadsBatch: k.ThreadsBatch}
	if k.MemCap > 0 {
		// The plan may take up to half the tolerance over the nominal limit: the
		// measurement is checked against the whole tolerance, and this leaves it
		// the other half for the error of the plan.
		t.MemCap = k.MemCap + k.MemTolerance/2
	}
	if k.Mode == Silent {
		t.Preset = "ram-4g"
	}
	return t
}

// prune sets the rejection and measure flags and returns the problems it found.
func prune(s *Survey, k Constraints, in PlanInput, cands []*Candidate) []Problem {
	var problems []Problem
	diskMiBs := 0.0
	if s.Disk.Speed != nil {
		diskMiBs = s.Disk.Speed.SeqMiBs
	}
	lowRAM := false
	for _, c := range cands {
		if c.Reject != nil {
			continue
		}
		d := c.Device
		if d.Discrete() {
			if free := freeVRAM(s, d); free > 0 && c.PlannedVRAM > free {
				c.Reject = &Rejection{RejectVRAM, fmt.Sprintf("needs %s of video memory, %s is free", gib(c.PlannedVRAM), gib(free))}
				continue
			}
		}
		if s.RAM.Available > 0 && c.PlannedPrivate > s.RAM.Available {
			c.Reject = &Rejection{RejectRAM, fmt.Sprintf("needs %s of host memory, %s is free", gib(c.PlannedPrivate), gib(s.RAM.Available))}
			lowRAM = true
			continue
		}
		threads := c.Spec.Threads
		c.Estimate = in.Matrix.EstimateFor(c.Class, deviceKindOf(d), threads, c.MTP, c.Plan, diskMiBs)
		if c.Estimate.DiskBound && c.Estimate.DiskTokS < k.MinTokS {
			c.Reject = &Rejection{RejectDisk, fmt.Sprintf("the disk (%.0f MB/s) allows %.2f tokens per second, below %.0f", diskMiBs*1.048576, c.Estimate.DiskTokS, k.MinTokS)}
			continue
		}
	}
	if lowRAM {
		problems = append(problems, Problem{
			Code: LowFreeRAM, Severity: SevWarn,
			Message: fmt.Sprintf("Only %s of RAM is free: setups that need more are left out.", gib(s.RAM.Available)),
			Advice:  "Close other programs and repeat the check.",
			Params:  map[string]any{"available_bytes": s.RAM.Available},
		})
	}
	markMeasured(cands, k)
	return problems
}

func deviceKindOf(d Device) string {
	if d.Type == string(backend.CPU) {
		return "cpu"
	}
	return d.Kind
}

// freeVRAM is the video memory a discrete GPU has free: the smaller of what
// the backend reports and what nvidia-smi leaves, when both are known (WDDM
// sometimes reports zero free memory under pressure).
func freeVRAM(s *Survey, d Device) int64 {
	free := d.MemoryFree
	for _, g := range s.GPUs {
		if (d.Kind == "cuda" && g.Index == deviceIndex(d.ID)) || (d.Kind != "cuda" && g.Name == d.Name) {
			if left := (g.MemTotalMiB - g.MemUsedMiB) * MiB; left > 0 && (free == 0 || left < free) {
				free = left
			}
		}
	}
	return free
}

// ReachFactor is how much slower than the best estimate a candidate may be
// estimated to be and still get measured.
const ReachFactor = 1.5

// markMeasured decides which contenders are measured: all with no estimate, and
// those whose estimated answer time is within ReachFactor of the best known.
// Where a candidate clearly wins by estimate (a discrete GPU with our CUDA
// library against a CPU) the rest are left out.
func markMeasured(cands []*Candidate, k Constraints) {
	best := 0.0
	for _, c := range cands {
		if c.Reject == nil && c.Estimate.Known && (best == 0 || c.Estimate.AnswerS < best) {
			best = c.Estimate.AnswerS
		}
	}
	for _, c := range cands {
		if c.Reject != nil {
			continue
		}
		switch {
		case k.Goal == GoalMinMemory, !c.Estimate.Known, best == 0:
			c.Measure = true
		case c.Estimate.AnswerS <= ReachFactor*best:
			c.Measure = true
		default:
			c.Reject = &Rejection{RejectSlower, fmt.Sprintf("estimated %.0f s for a typical answer against %.0f s for the best", c.Estimate.AnswerS, best)}
		}
	}
}

// order sorts candidates in the order they are measured: by the mode's goal.
// Silent goes from the smallest memory up; the others from the best estimate
// down, GPUs before the CPU on a tie, candidates without an estimate last.
func order(cands []*Candidate, k Constraints) {
	rank := func(c *Candidate) int {
		if c.Reject != nil {
			return 2
		}
		return 0
	}
	sort.SliceStable(cands, func(i, j int) bool {
		a, b := cands[i], cands[j]
		if ra, rb := rank(a), rank(b); ra != rb {
			return ra < rb
		}
		if k.Goal == GoalMinMemory {
			return a.PlannedBytes < b.PlannedBytes
		}
		ea, eb := a.Estimate.AnswerS, b.Estimate.AnswerS
		if a.Estimate.Known != b.Estimate.Known {
			return a.Estimate.Known
		}
		if ea != eb {
			return ea < eb
		}
		// With nothing to estimate by (a machine no baseline resembles) the
		// posture of the type of device decides: a card with memory of its own
		// first, then a GPU on shared memory, then the processor. It costs at
		// worst a load, not the result: the measurement has the last word.
		if ra, rb := deviceRank(a.Device), deviceRank(b.Device); ra != rb {
			return ra < rb
		}
		return cmp.Compare(a.PlannedBytes, b.PlannedBytes) < 0
	})
}

// dropSameCard rules out a Vulkan device that is a card CUDA already offers:
// NanoLLM's own CUDA library is the faster way to run it, and measuring both
// would cost a load for nothing.
func dropSameCard(s *Survey, cands []*Candidate) {
	cuda := map[string]string{}
	for _, d := range s.Devices {
		if d.Kind == "cuda" {
			cuda[d.Name] = d.Selector
		}
	}
	for _, c := range cands {
		if c.Device.Kind != "vulkan" || c.Reject != nil {
			continue
		}
		if sel, ok := cuda[c.Device.Name]; ok {
			c.Reject = &Rejection{RejectNoDevice, "the same card runs through " + sel}
		}
	}
}

// deviceRank orders the types of device by how likely they are to be the
// fastest where nothing is known: 0 a GPU with memory of its own, 1 a GPU on
// shared memory, 2 the CPU.
func deviceRank(d Device) int {
	switch {
	case d.Discrete():
		return 0
	case d.Type == string(backend.GPU):
		return 1
	}
	return 2
}
