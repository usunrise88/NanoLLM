package bench

import (
	"context"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/sched"
)

func TestScenariosOnFake(t *testing.T) {
	// Rates give every scenario a measurable duration: an instant fake reads
	// as zero on a coarse clock (Windows).
	m, err := fake.New(fake.Options{MaxDraft: 2, Acceptance: 0.5, PrefillRate: 1e6, DecodeRate: 2e4}).Load(context.Background(), backend.LoadSpec{NCtx: 8192, NBatch: 8192, DrafterPath: "d"})
	if err != nil {
		t.Fatal(err)
	}
	loop := sched.New(m)
	defer loop.Close()
	res, err := run(context.Background(), m, loop, WorkerSpec{Reps: 2}, WorkerResult{})
	if err != nil {
		t.Fatal(err)
	}
	mt := res.Metrics
	if mt.PPTokS <= 0 || mt.TGTokS.D0 <= 0 || mt.TGTokS.DN <= 0 || mt.TGTokS.DepthN != DepthN || mt.ChatTokS <= 0 {
		t.Fatalf("%+v", mt)
	}
	if mt.MTP == nil || res.Output.SHA256 == "" || res.Output.Tokens == 0 {
		t.Fatalf("mtp %+v output %+v", mt.MTP, res.Output)
	}
}

func TestChoosePresets(t *testing.T) {
	cpu := &backend.Device{Backend: "llamacpp", ID: "cpu", Type: backend.CPU, MemoryFree: 16 << 30}
	gpu := &backend.Device{Backend: "llamacpp", ID: "cuda0", Type: backend.GPU, MemoryFree: 7 << 30, MemoryTotal: 8 << 30}
	got := choosePresets(Options{}, cpu, nil)
	if len(got) != 3 || got[0] != "ram-12g" || got[1] != "ram-8g" {
		t.Fatalf("cpu quick: %v", got)
	}
	got = choosePresets(Options{}, cpu, gpu)
	if got[0] != "vram-7g" || got[1] != "vram-8g" {
		t.Fatalf("gpu quick: %v", got)
	}
	full := choosePresets(Options{Full: true}, cpu, nil)
	for _, p := range full {
		if p[:3] != "ram" {
			t.Fatalf("a CPU-only machine is not asked for %s", p)
		}
	}
	if PromptsSHA256() == "" || planHash(map[string]int{"a": 1}) == planHash(map[string]int{"a": 2}) {
		t.Fatal("hashes")
	}
}

// TestPromptsPinned fails when the suite's prompts or parameters change, so
// the change is made on purpose: the bench service refuses results whose
// hash it does not know.
func TestPromptsPinned(t *testing.T) {
	const want = "ad01c12080c9d8d68bf3bd9414940694635d0fdf85c2539ffee4b127c7429754"
	if got := PromptsSHA256(); got != want {
		t.Fatalf("standard/%s prompts hash %s, pinned %s: bump SuiteVersion once results are public; "+
			"until then update this constant, services/bench ingest.StandardV1Prompts and services/bench/testdata/sample-result.json", SuiteVersion, got, want)
	}
}

func TestMedianStd(t *testing.T) {
	m, s := medianStd([]float64{3, 1, 2})
	if m != 2 || s != 1 {
		t.Fatalf("%v %v", m, s)
	}
}

func TestFakeResultsStayOffThePublicService(t *testing.T) {
	res := &benchfmt.Result{Suite: benchfmt.Suite{ID: SuiteStandard}, Client: benchfmt.Client{GGMLCPUVariant: benchfmt.FakeCPUVariant}}
	_, err := (&Client{BaseURL: DefaultSubmitURL}).Submit(context.Background(), res, Anonymous())
	if err == nil || !strings.Contains(err.Error(), "fake backend") {
		t.Fatalf("a fake result went to %s: %v", DefaultSubmitURL, err)
	}
}
