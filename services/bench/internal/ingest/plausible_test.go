package ingest

import (
	"math"
	"testing"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

func TestCeiling(t *testing.T) {
	r := sample(60)
	cost := Models["gemma-4-e4b-qat"]
	total := float64(cost.LayerBytes + cost.OutputBytes)
	if total != 2614424744 {
		t.Fatalf("E4B bytes per token %v, want the measured core without PLE", total)
	}
	// All on the GPU: 448 GB/s.
	got := Ceiling(r, r.Runs[0], "Intel Core i7-12700H", 1)
	if want := 448e9 / total; math.Abs(got-want) > 1e-6 {
		t.Fatalf("GPU ceiling %v, want %v", got, want)
	}
	// On the CPU: DDR5-4800 over a 256-bit laptop bus, 153.6 GB/s.
	got = Ceiling(r, r.Runs[1], "Intel Core i7-12700H", 1)
	if want := 153.6e9 / total; math.Abs(got-want) > 1e-6 {
		t.Fatalf("CPU ceiling %v, want %v", got, want)
	}
	// Half the layers on the GPU: two bandwidths.
	run := benchfmt.Run{Preset: "vram-4g", Status: "ok", Reps: 3, Plan: benchfmt.Plan{Mode: "vram-partial", Device: "CUDA0", GPULayers: 21}}
	per := float64(cost.LayerBytes) / 42
	want := 1 / (21*per/448e9 + (21*per+float64(cost.OutputBytes))/153.6e9)
	if got := Ceiling(r, run, "Intel Core i7-12700H", 1); math.Abs(got-want) > 1e-6 {
		t.Fatalf("mixed ceiling %v, want %v", got, want)
	}
	// Apple: unified memory, the SoC's bandwidth.
	mac := sample(60)
	mac.Machine.GPUs = []benchfmt.GPU{{Name: "Apple M2 Pro", Vendor: "Apple", Cores: 19, Unified: true}}
	mac.Machine.Memory.SpeedMTS = 0
	run = benchfmt.Run{Preset: "uma-8g", Status: "ok", Reps: 3, Plan: benchfmt.Plan{Mode: "vram-full", Device: "Metal", GPULayers: -1}}
	if got, want := Ceiling(mac, run, "Apple M2 Pro", 1), 200e9/total; math.Abs(got-want) > 1e-6 {
		t.Fatalf("M2 Pro ceiling %v, want %v", got, want)
	}
}

func TestCeilingUnknown(t *testing.T) {
	r := sample(60)
	r.Model.ID = "some-other-model"
	if Ceiling(r, r.Runs[0], "Intel Core i7-12700H", 1) != 0 {
		t.Error("unknown model has a ceiling")
	}
	r = sample(60)
	r.Machine.System.FormFactor = benchfmt.FormVM
	if Ceiling(r, r.Runs[1], "Intel Core i7-12700H", 1) != 0 {
		t.Error("a VM without a table entry has a ceiling")
	}
	r = sample(60)
	r.Machine.Memory.SpeedMTS = 0
	if p, _ := Plausibility(r, r.Runs[1], "Intel Core i7-12700H", 1.3); p != store.PlausibleUnknown {
		t.Error(p)
	}
	// A server CPU in the table needs no memory speed.
	r.Machine.System.FormFactor = benchfmt.FormServer
	if Ceiling(r, r.Runs[1], "AMD EPYC 9454P", 1) == 0 {
		t.Error("EPYC Genoa has no ceiling")
	}
	// A refused run is not judged.
	if p, _ := Plausibility(r, r.Runs[2], "AMD EPYC 9454P", 1.3); p != store.PlausibleUnknown {
		t.Error(p)
	}
}
