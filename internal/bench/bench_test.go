package bench

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/plan"
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

func TestSpawnTimesOut(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("needs a shell script as the worker")
	}
	exe := filepath.Join(t.TempDir(), "slow-worker")
	if err := os.WriteFile(exe, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	o := Options{Exe: exe, Backend: fake.New(fake.Options{}), Advisory: true, Timeout: 200 * time.Millisecond}
	start := time.Now()
	_, _, err := o.spawn(context.Background(), WorkerSpec{}, plan.Budget{})
	if err == nil || !strings.Contains(err.Error(), "too slow") {
		t.Fatalf("got %v", err)
	}
	if d := time.Since(start); d > 10*time.Second {
		t.Fatalf("took %s to give up", d)
	}
}

// The throughput curve: one point per number of sequences, each with and
// without drafting, and the slots the points need follow from the key.
func TestBatchCurveOnFake(t *testing.T) {
	m, err := fake.New(fake.Options{MaxDraft: 2, Acceptance: 0.5, PrefillRate: 1e6, DecodeRate: 2e4}).Load(context.Background(), backend.LoadSpec{NCtx: 8192, NBatch: 8192, Slots: 4, DrafterPath: "d"})
	if err != nil {
		t.Fatal(err)
	}
	loop := sched.New(m)
	defer loop.Close()
	res, err := run(context.Background(), m, loop, WorkerSpec{Reps: 2, Batch: 4}, WorkerResult{})
	if err != nil {
		t.Fatal(err)
	}
	var seqs []int
	for _, p := range res.Metrics.Batch {
		seqs = append(seqs, p.Seqs)
		if p.AggTokS <= 0 || p.PerSeqTokS <= 0 || p.MTPAggTokS <= 0 || p.MTPPerSeqTokS <= 0 || p.Acceptance <= 0 {
			t.Errorf("point %+v", p)
		}
	}
	if !slices.Equal(seqs, []int{1, 2, 4}) {
		t.Fatalf("points %v", seqs)
	}
}

func TestBatchSizesAndKey(t *testing.T) {
	for top, want := range map[int][]int{2: {1, 2}, 3: {1, 2, 3}, 4: {1, 2, 4}, 6: {1, 2, 4, 6}, 8: {1, 2, 4, 8}} {
		if got := batchSizes(top); !slices.Equal(got, want) {
			t.Errorf("batchSizes(%d) = %v", top, got)
		}
	}
	if _, err := ParseSets([]string{"batch=1"}); err == nil {
		t.Error("batch=1 is a curve of one point: refuse it")
	}
	vs, err := ParseSets([]string{"batch=4"})
	if err != nil || len(vs) != 1 {
		t.Fatalf("%v %v", vs, err)
	}
	req := plan.Request{Slots: 2}
	vs[0].Plan(&req)
	if req.Slots != 4 {
		t.Errorf("slots %d: the curve needs a slot for each sequence", req.Slots)
	}
	req = plan.Request{Slots: 8}
	vs[0].Plan(&req)
	if req.Slots != 8 {
		t.Errorf("slots %d: more slots than the curve needs stay", req.Slots)
	}
}
