package placement

import (
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

const gib = int64(1) << 30

func manifest(t *testing.T) registry.Manifest {
	t.Helper()
	cat, err := registry.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := registry.Find(cat, "gemma-4-e4b-qat")
	if !ok {
		t.Fatal("no gemma-4-e4b-qat in the catalog")
	}
	return m
}

var (
	rtx = backend.Device{Backend: "llamacpp", ID: "cuda0", Type: backend.GPU, Name: "RTX", MemoryTotal: 8 * gib, MemoryFree: 7 * gib,
		Features: []string{backend.FeatureCudaNano, backend.FeatureBackendSampling}}
	cpu = backend.Device{Backend: "llamacpp", ID: "cpu", Type: backend.CPU, Name: "CPU", MemoryTotal: 32 * gib, MemoryFree: 20 * gib,
		Features: []string{backend.FeatureMultiRowQ4_0}}
)

func resolve(t *testing.T, in Input) *Result {
	t.Helper()
	in.Manifest = manifest(t)
	in.ModelPath, in.DrafterPath = "m.gguf", "d.gguf"
	r, err := Resolve(in)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// Under the optimal mode's cap a discrete GPU does not get the modes that keep
// the PLE table and the embeddings in the card or in pinned host memory: M3
// holds the same model in less.
func TestMemCapMovesTheGPUToM3(t *testing.T) {
	open := resolve(t, Input{Device: rtx})
	if open.Plan.Mode != plan.VRAMFull {
		t.Fatalf("without a cap: %s; with 7 GiB free everything goes to the card (M1)", open.Plan.Mode)
	}
	capped := resolve(t, Input{Device: rtx, Tune: Tune{Source: SourceMode, MemCap: 4*gib + 100<<20, MinNCtx: 4608, Threads: 7}})
	p := capped.Plan
	if p.Mode != plan.VRAMPLEDisk {
		t.Fatalf("under the cap: %s, want M3", p.Mode)
	}
	if p.NCtx < 4608 {
		t.Fatalf("context %d under the optimal mode's minimum", p.NCtx)
	}
	if c := p.Memory.Combined; c == nil || c.Used > c.Budget || c.Budget != 4*gib+100<<20 {
		t.Fatalf("combined %+v", c)
	}
	if p.Sources["mem_cap"] != SourceMode || p.Sources["min_n_ctx"] != SourceMode {
		t.Fatalf("sources %v", p.Sources)
	}
	// A GPU model keeps no CPU team.
	if capped.Spec.Threads != 0 {
		t.Fatalf("threads %d on a model fully on the GPU", capped.Spec.Threads)
	}
}

func TestAnExplicitPresetBeatsTheCap(t *testing.T) {
	r := resolve(t, Input{Device: rtx, Override: config.ModelOverride{Preset: "vram-6g"}, Tune: Tune{MemCap: 4 * gib, MinNCtx: 4608}})
	if r.Plan.Memory.Combined != nil || r.Plan.Mode != plan.VRAMFull {
		t.Fatalf("mode %s, combined %+v: a preset the user wrote is a hard budget of its own", r.Plan.Mode, r.Plan.Memory.Combined)
	}
	if r.Plan.Sources["preset"] != SourceConfig {
		t.Fatalf("sources %v", r.Plan.Sources)
	}
}

func TestSilentCPUTakesTheModeBudgetAndThreads(t *testing.T) {
	r := resolve(t, Input{Device: cpu, Tune: Tune{Source: SourceMode, Preset: "ram-4g", MinNCtx: 4608, Threads: 2, ThreadsBatch: 2}})
	if r.Preset.Name != "ram-4g" || r.Plan.Sources["preset"] != SourceMode {
		t.Fatalf("preset %s sources %v: the mode's budget is smaller than the 20 GiB the device offers", r.Preset.Name, r.Plan.Sources)
	}
	if r.Spec.Threads != 2 || r.Spec.ThreadsBatch != 2 || r.Plan.Sources["threads"] != SourceMode {
		t.Fatalf("threads %d/%d %v", r.Spec.Threads, r.Spec.ThreadsBatch, r.Plan.Sources)
	}
	if r.Plan.NCtx < 4608 {
		t.Fatalf("context %d", r.Plan.NCtx)
	}
	// The configuration's own thread count beats the mode's.
	r = resolve(t, Input{Device: cpu, Threads: 3, ThreadsBatch: 5, Tune: Tune{Threads: 2, ThreadsBatch: 2}})
	if r.Spec.Threads != 3 || r.Spec.ThreadsBatch != 5 || r.Plan.Sources["threads"] != SourceConfig {
		t.Fatalf("threads %d/%d %v", r.Spec.Threads, r.Spec.ThreadsBatch, r.Plan.Sources)
	}
}

func TestAProfileSetsDraftingBelowTheConfiguration(t *testing.T) {
	zero, two := 0.0, 2
	on := true
	prof := Tune{Source: SourceProfile, MTP: &on, DraftMax: &two, DraftPMin: &zero}
	r := resolve(t, Input{Device: cpu, Tune: prof})
	if !r.Plan.MTP.Enabled || r.Spec.DraftMax != 2 || r.Spec.DraftPMin != 0 {
		t.Fatalf("MTP %+v spec %d/%v: a p_min of 0 is a value, not 'unset'", r.Plan.MTP, r.Spec.DraftMax, r.Spec.DraftPMin)
	}
	if r.Plan.Sources["draft_p_min"] != SourceProfile {
		t.Fatalf("sources %v", r.Plan.Sources)
	}
	// The rule without a profile: 2 / 0.5 on a CPU with the multi-row kernels.
	r = resolve(t, Input{Device: cpu})
	if r.Spec.DraftMax != plan.CPUMTPDraftMax || r.Spec.DraftPMin != plan.CPUMTPDraftPMin {
		t.Fatalf("rule: %d/%v", r.Spec.DraftMax, r.Spec.DraftPMin)
	}
	// The configuration beats the profile.
	r = resolve(t, Input{Device: cpu, Tune: prof, Override: config.ModelOverride{Speculative: config.Speculative{Type: config.SpecNone}}})
	if r.Plan.MTP.Enabled {
		t.Fatal("speculative.type: none was overridden by a profile")
	}
	off := false
	r = resolve(t, Input{Device: cpu, Tune: Tune{Source: SourceProfile, MTP: &off}})
	if r.Plan.MTP.Enabled {
		t.Fatal("a profile that turns drafting off did not")
	}
}

func TestTheZeroTuneChangesNothing(t *testing.T) {
	if !(Tune{}).Zero() || (Tune{MemCap: 1}).Zero() {
		t.Fatal("Zero")
	}
	a := resolve(t, Input{Device: rtx})
	b := resolve(t, Input{Device: rtx, Tune: Tune{}})
	if a.Plan.Mode != b.Plan.Mode || a.Plan.NCtx != b.Plan.NCtx || len(b.Plan.Sources) != 0 {
		t.Fatalf("%v %v %v", a.Plan.Mode, b.Plan.Mode, b.Plan.Sources)
	}
}
