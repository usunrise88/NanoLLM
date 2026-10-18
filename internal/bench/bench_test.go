package bench

import (
	"context"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/sched"
)

func TestScenariosOnFake(t *testing.T) {
	m, err := fake.New(fake.Options{MaxDraft: 2, Acceptance: 0.5}).Load(context.Background(), backend.LoadSpec{NCtx: 8192, NBatch: 8192, DrafterPath: "d"})
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

func TestMedianStd(t *testing.T) {
	m, s := medianStd([]float64{3, 1, 2})
	if m != 2 || s != 1 {
		t.Fatalf("%v %v", m, s)
	}
}
