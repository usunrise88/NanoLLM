package tune

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/registry"
)

// recordingRunner counts the probes and keeps their specs.
type recordingRunner struct {
	inner Runner
	mu    sync.Mutex
	specs []ProbeSpec
}

func (r *recordingRunner) Probe(ctx context.Context, spec ProbeSpec) (ProbeResult, error) {
	r.mu.Lock()
	r.specs = append(r.specs, spec)
	r.mu.Unlock()
	return r.inner.Probe(ctx, spec)
}

func fakeRun(t *testing.T, mode Mode, opts fake.Options) (*Result, *recordingRunner, []Event) {
	t.Helper()
	cat, err := registry.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := registry.Find(cat, "gemma-4-e4b-qat")
	if !ok {
		t.Fatal("no gemma-4-e4b-qat in the catalog")
	}
	b := fake.New(opts)
	rr := &recordingRunner{inner: InProcessRunner{Backend: b}}
	var events []Event
	res, err := Run(context.Background(), RunOptions{
		Options: Options{
			Mode: mode,
			Survey: SurveyOptions{
				Backends: []backend.Backend{b}, Manifest: m, ModelPath: "fake.gguf", DrafterPath: "fake-drafter.gguf",
				SkipDisk: true, SkipMachine: true, LiveWindow: 10 * time.Millisecond, Version: "test", LibsRelease: "b1-n1",
			},
			Plan: PlanInput{Manifest: m, ModelPath: "fake.gguf", DrafterPath: "fake-drafter.gguf"},
		},
		Runner:   rr,
		Cooldown: func(context.Context, Device, int) {},
		Progress: func(e Event) { events = append(events, e) },
		Budget:   2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	return res, rr, events
}

var quickFake = fake.Options{
	PrefillRate: 40000, DecodeRate: 500, MaxDraft: 3, Acceptance: 0.6, Memory: 16 << 30,
	BatchCost: func(tokens, outputs int) time.Duration {
		return time.Duration(10_000+400*(tokens-1)) * time.Microsecond
	},
}

func TestRunOnAScriptedBackend(t *testing.T) {
	res, rr, events := fakeRun(t, Optimal, quickFake)
	p := res.Profile
	if p == nil || p.Mode != Optimal || p.Key.Model != "gemma-4-e4b-qat" || p.Key.LibsRelease != "b1-n1" {
		t.Fatalf("profile %+v", p)
	}
	if len(rr.specs) == 0 || len(rr.specs) > MaxLoads {
		t.Fatalf("%d loads, at most %d", len(rr.specs), MaxLoads)
	}
	// The events come in the order the TUI shows them.
	var types []string
	for _, e := range events {
		if len(types) == 0 || types[len(types)-1] != e.Type {
			types = append(types, e.Type)
		}
	}
	if types[0] != "survey" || types[len(types)-1] != "done" {
		t.Fatalf("events %v", types)
	}
	seen := map[string]bool{}
	for _, e := range events {
		seen[e.Type] = true
	}
	for _, want := range []string{"survey", "candidate", "probe_start", "probe", "decision", "done"} {
		if !seen[want] {
			t.Errorf("no %q event in %v", want, types)
		}
	}
	// The scripted drafter accepts 60 % and a batch of 4 costs 1.12 of one: drafting pays.
	d, ok := decodeDraft(p.Chosen.MTP.Value)
	if !ok || !d.Enabled || d.DraftMax < 2 {
		t.Fatalf("MTP %+v", p.Chosen.MTP)
	}
	if p.Chosen.MTP.Source != SourceMeasured && p.Chosen.MTP.Source != SourceMechanism {
		t.Fatalf("MTP source %q", p.Chosen.MTP.Source)
	}
	if p.Answer.AnswerS <= 0 || p.Answer.PP <= 0 || p.Answer.TG <= 0 {
		t.Fatalf("answer %+v", p.Answer)
	}
	if p.Memory.MeasuredBytes <= 0 || p.Memory.LimitBytes != 4*GiB {
		t.Fatalf("memory %+v", p.Memory)
	}
	if !p.Verified.DraftChecked || !p.Verified.DraftMatch {
		t.Fatalf("verified %+v (skipped %v)", p.Verified, p.Skipped)
	}
	// The chosen setup is not slower than the one that runs without a profile.
	if p.Verified.StaticTokS > 0 && p.Verified.ChosenTokS < p.Verified.StaticTokS {
		t.Fatalf("verified %+v", p.Verified)
	}
	// What the profile says survives a round trip through the store.
	st := Store{Dir: t.TempDir()}
	if err := st.Save(p); err != nil {
		t.Fatal(err)
	}
	back, err := st.Load("gemma-4-e4b-qat", Optimal)
	if err != nil || back.ID() != p.ID() {
		t.Fatalf("round trip: %v %v", back, err)
	}
}

func TestSilentStopsAtTheFirstSetupThatWorks(t *testing.T) {
	res, rr, _ := fakeRun(t, Silent, quickFake)
	if len(rr.specs) != 1 {
		t.Fatalf("the quiet mode loaded %d times; the smallest budget works, so one", len(rr.specs))
	}
	c := res.Decision.Chosen.Candidate
	if c.Device.Type != "cpu" {
		t.Fatalf("the quiet mode chose %s", c.ID)
	}
	if spec := rr.specs[0]; spec.Load.Threads != 2 || spec.Load.ThreadsBatch != 2 {
		t.Fatalf("threads %d/%d, the quiet mode runs 2/2", spec.Load.Threads, spec.Load.ThreadsBatch)
	}
	// The smallest budget of the ladder that the planner can place.
	for _, o := range res.Report.Candidates {
		if o.Device.Type == "cpu" && o.Reject == nil && o.PlannedBytes < c.PlannedBytes {
			t.Fatalf("%s holds less than the chosen %s", o.ID, c.ID)
		}
	}
}

func TestPowerfulUsesNoMemoryCap(t *testing.T) {
	res, _, _ := fakeRun(t, Powerful, quickFake)
	if res.Report.Constraints.MemCap != 0 || res.Profile.Memory.LimitBytes != 0 {
		t.Fatalf("%+v", res.Report.Constraints)
	}
}

func TestVariantsOfADevice(t *testing.T) {
	cands := &Candidate{Device: Device{Type: "gpu", Kind: "cuda"}, Class: rtx3070ti}
	vs, static, mech := variantsFor(&Survey{}, cands, PlanInput{})
	names := map[string]bool{}
	for _, v := range vs {
		names[v.Name] = true
	}
	if !names["off"] || !names["mtp3/0"] || !names["mtp2/0"] || static != "off" || mech {
		t.Fatalf("cuda variants %v static %q mech %v", vs, static, mech)
	}
	cpu := &Candidate{Device: Device{Type: "cpu", Kind: "cpu"}, Class: zen4CPU}
	vs, _, mech = variantsFor(&Survey{}, cpu, PlanInput{})
	if len(vs) != 1 || !mech {
		t.Fatalf("a CPU without the fast kernels: %v mech %v", vs, mech)
	}
	cpu.Device.Features = []string{backend.FeatureMultiRowQ4_0}
	cpu.Class = meteorLakeCPU
	vs, _, mech = variantsFor(&Survey{}, cpu, PlanInput{})
	names = map[string]bool{}
	for _, v := range vs {
		names[v.Name] = true
	}
	if mech || !names["mtp2/0.5"] || !names["mtp3/0"] || !names["mtp2/0"] {
		t.Fatalf("a CPU with the kernels: %v", vs)
	}
}

// Several NUMA nodes: the processor is measured under each memory policy, in a
// worker placed by it, and the winner is written to the configuration.
func TestNUMAPoliciesAreCandidatesOnAMultiNodeMachine(t *testing.T) {
	cat, _ := registry.Builtin()
	m, _ := registry.Find(cat, "gemma-4-e4b-qat")
	b := fake.New(quickFake)
	so := SurveyOptions{Backends: []backend.Backend{b}, Manifest: m, ModelPath: "fake.gguf", DrafterPath: "fake-drafter.gguf", SkipDisk: true, SkipMachine: true, LiveWindow: time.Millisecond}
	s := TakeSurvey(context.Background(), so)
	s.CPU.NUMANodes = 4
	opts := Options{Mode: Powerful, Survey: so, Plan: PlanInput{Manifest: m, ModelPath: "fake.gguf", DrafterPath: "fake-drafter.gguf"}}
	rep := Analyze(s, opts)
	got := map[string]*Candidate{}
	for _, c := range rep.Candidates {
		if c.NUMA != "" {
			got[c.NUMA] = c
		}
	}
	if len(got) != 2 || got["interleave"] == nil || got["distribute"] == nil {
		t.Fatalf("NUMA candidates %v", got)
	}
	if got["interleave"].ID == got["distribute"].ID {
		t.Fatal("the two policies share an id")
	}
	spec := BuildProbeSpec(s, rep.Constraints, opts.Plan, got["distribute"], opts)
	if env := spec.Env["NANOLLM_PLACE"]; !strings.HasPrefix(env, "distribute;threads=") {
		t.Fatalf("worker environment %v: the repack threads must be the decode threads", spec.Env)
	}
	if spec := BuildProbeSpec(s, rep.Constraints, opts.Plan, got["interleave"], opts); spec.Env["NANOLLM_PLACE"] != "interleave" {
		t.Fatalf("worker environment %v", spec.Env)
	}
	// One node: no policy to choose.
	s.CPU.NUMANodes = 1
	for _, c := range Analyze(s, opts).Candidates {
		if c.NUMA != "" {
			t.Fatalf("a single-node machine got the %s policy", c.NUMA)
		}
	}
}

func TestApplyWritesTheModeAndThePolicy(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "nanollm.yaml")
	body := "# my configuration\nserver:\n  data_dir: " + strconvQuote(dir) + "\nbackends:\n  lib_dir: ./lib\n  enabled: [fake]\nauth:\n  mode: open\n# keep this comment\ndevices:\n  default: auto\n"
	if err := os.WriteFile(cfg, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	p := &Profile{Schema: ProfileSchema, Mode: Silent, Key: Key{Model: "m"}, Chosen: Chosen{NUMA: &Value{Value: "distribute"}}}
	res, err := Apply(p, ApplyOptions{Store: Store{Dir: filepath.Join(dir, "tune")}, ConfigPath: cfg})
	if err != nil || !res.ProfileSaved || !res.ConfigWritten {
		t.Fatalf("%+v %v", res, err)
	}
	if len(res.RestartRequired) != 1 || !strings.Contains(res.RestartRequired[0], "devices.numa: distribute") {
		t.Fatalf("restart: %v", res.RestartRequired)
	}
	out, _ := os.ReadFile(cfg)
	for _, want := range []string{"# my configuration", "# keep this comment", "mode: silent", "numa: distribute"} {
		if !strings.Contains(string(out), want) {
			t.Errorf("the configuration lost %q:\n%s", want, out)
		}
	}
	// A configuration that cannot be written is a note, with the profile saved.
	res, err = Apply(p, ApplyOptions{Store: Store{Dir: filepath.Join(dir, "tune")}, ConfigPath: filepath.Join(dir, "missing", "nanollm.yaml")})
	if err != nil || !res.ProfileSaved || res.ConfigWritten || res.Note == "" {
		t.Fatalf("%+v %v", res, err)
	}
	if _, err := Apply(nil, ApplyOptions{}); err == nil {
		t.Fatal("applied nothing without complaint")
	}
}

func strconvQuote(s string) string { return fmt.Sprintf("%q", s) }
