package ingest

import (
	"math"
	"testing"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/services/bench/internal/normalize"
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

// EPYC names as the hardware report gives them, then as the service
// normalizes them: the table is matched on the normalized one.
func TestHostBandwidthEPYC(t *testing.T) {
	for _, c := range []struct {
		model string
		want  float64 // GB/s
	}{
		// Naples: 8 × DDR4-2666 per socket, the P parts single-socket.
		{"AMD EPYC 7351P 16-Core Processor", 170.7},
		{"AMD EPYC 7401P 24-Core Processor", 170.7},
		{"AMD EPYC 7551P 32-Core Processor", 170.7},
		{"AMD EPYC 7251 8-Core Processor", 170.7 * 2},
		{"AMD EPYC 7371 16-Core Processor", 170.7 * 2},
		{"AMD EPYC 7551 32-Core Processor", 170.7 * 2},
		{"AMD EPYC 7601 32-Core Processor", 170.7 * 2},
		// The neighbours keep their rows.
		{"AMD EPYC 7452 32-Core Processor", 204.8 * 2},
		{"AMD EPYC 7742 64-Core Processor", 204.8 * 2},
		{"AMD EPYC 7313P 16-Core Processor", 204.8 * 2},
		{"AMD EPYC 7763 64-Core Processor", 204.8 * 2},
		{"AMD EPYC 9454P 48-Core Processor", 460.8 * 2},
	} {
		cpu := normalize.CPU("AuthenticAMD", c.model)
		for _, name := range []string{cpu, c.model} {
			if got := HostBandwidth(benchfmt.Machine{}, name); math.Abs(got-c.want) > 1e-9 {
				t.Errorf("%q: %v GB/s, want %v", name, got, c.want)
			}
		}
	}
	// A name that only looks like Naples is not one.
	if got := HostBandwidth(benchfmt.Machine{}, "AMD EPYC 7V13"); got != 0 {
		t.Errorf("EPYC 7V13: %v GB/s from a table row, want none", got)
	}
}

// The reference machine, a 7351P, decodes E4B at about 20 tok/s; a number
// well beyond what its memory can deliver is refused, one near it is not.
func TestPlausibilityEPYCNaples(t *testing.T) {
	r := sample(60)
	r.Machine.CPU = benchfmt.CPU{Vendor: "AMD", Model: "AMD EPYC 7351P", Cores: 16, Threads: 32}
	r.Machine.GPUs = nil
	r.Machine.Memory = benchfmt.Memory{TotalBytes: 128 << 30}
	r.Machine.System = benchfmt.System{FormFactor: benchfmt.FormServer}
	cost := Models["gemma-4-e4b-qat"]
	total := float64(cost.LayerBytes + cost.OutputBytes)
	run := benchfmt.Run{Preset: "ram-8g", Status: benchfmt.StatusOK, Reps: 3,
		Plan: benchfmt.Plan{Mode: "ram-full", Device: "llamacpp:cpu", NCtx: 8192}}
	cpu := normalize.CPU("AMD", r.Machine.CPU.Model)
	if got, want := Ceiling(r, run, cpu, 1), 170.7e9/total; math.Abs(got-want) > 1e-6 {
		t.Fatalf("7351P ceiling %v tok/s, want %v", got, want)
	}
	run.Metrics.TGTokS = benchfmt.TG{D0: 20.8, DN: 19}
	run.Metrics.ChatTokS = 20.2
	if p, c := Plausibility(r, run, cpu, 1.3); p != store.PlausibleOK || c < 80 || c > 90 {
		t.Errorf("a measured 20.8 tok/s: %q under %v tok/s", p, c)
	}
	run.Metrics.TGTokS.D0 = 90
	if p, _ := Plausibility(r, run, cpu, 1.3); p != store.PlausibleNo {
		t.Errorf("90 tok/s on a 7351P: %q", p)
	}
	// A 7551 may be a two-socket machine reading both sockets' memory.
	cpu = normalize.CPU("AMD", "AMD EPYC 7551 32-Core Processor")
	if p, _ := Plausibility(r, run, cpu, 1.3); p != store.PlausibleOK {
		t.Errorf("90 tok/s on a 7551: %q", p)
	}
}
