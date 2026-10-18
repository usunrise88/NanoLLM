package plan

import (
	"math"
	"strings"
	"testing"
)

// TestComputeMatchesLlamaLog checks the CPU fit against `CPU compute buffer
// size` lines of b11600 on the stage-1 VM. The first group ran with the
// engine default n_outputs_max (= n_batch, 2048); the second is llama-server
// with the MTP drafter, which sets n_outputs_max = slots·(1+n_draft) on the
// target and = slots on the draft context.
func TestComputeMatchesLlamaLog(t *testing.T) {
	model := e4b(t)
	target, drafter := model.Target.HParams, model.Drafter.HParams
	for _, c := range []struct {
		name     string
		drafter  bool
		shape    ComputeShape
		measured float64 // MiB
		tol      float64 // relative
	}{
		{"ub512", false, ComputeShape{512, 2048, 4096, true}, 563.03, 0.005},
		{"ub256", false, ComputeShape{256, 2048, 4096, true}, 281.39, 0.005},
		{"ub128 b128", false, ComputeShape{128, 128, 4096, true}, 140.70, 0.005},
		{"c32768", false, ComputeShape{512, 2048, 32768, true}, 591.03, 0.005},
		{"np4 c8192", false, ComputeShape{512, 2048, 2048, true}, 561.03, 0.005},
		{"no FA", false, ComputeShape{512, 2048, 4096, false}, 576.02, 0.01},
		{"no FA c32768", false, ComputeShape{512, 2048, 32768, false}, 658.02, 0.01},
		{"mtp target", false, ComputeShape{512, 4, 4096, true}, 112.03, 0.02},
		{"mtp target ub256", false, ComputeShape{256, 4, 4096, true}, 55.89, 0.02},
		{"mtp target np4", false, ComputeShape{512, 16, 4096, true}, 112.03, 0.02},
		{"mtp target ub128", false, ComputeShape{128, 4, 4096, true}, 27.94, 0.02},
		{"mtp target c32768", false, ComputeShape{512, 4, 32768, true}, 140.03, 0.02},
		{"mtp drafter", true, ComputeShape{512, 1, 4096, true}, 25.00, 0.06},
		{"mtp drafter ub256", true, ComputeShape{256, 1, 4096, true}, 12.38, 0.06},
		{"mtp drafter ub128", true, ComputeShape{128, 1, 4096, true}, 6.19, 0.15}, // 0.8 MiB high
		{"mtp drafter c32768", true, ComputeShape{512, 1, 32768, true}, 53.00, 0.02},
		// Without flash attention the fit errs high, on purpose: it is the
		// rare path, and an overestimate only costs context.
		{"mtp target no FA", false, ComputeShape{512, 4, 4096, false}, 154.02, 0.07},
		{"mtp drafter no FA", true, ComputeShape{512, 1, 4096, false}, 51.00, 0.15},
	} {
		h := target
		if c.drafter {
			h = drafter
		}
		got := float64(cpuCoeffs.ComputeBytes(h, c.shape)) / mib
		if math.Abs(got-c.measured)/c.measured > c.tol {
			t.Errorf("%s: estimate %.2f MiB, logged %.2f", c.name, got, c.measured)
		}
	}
}

func TestCalibrate(t *testing.T) {
	h := e4b(t).Target.HParams
	// A GPU whose activations run 30% above the CPU fit, at both peaks.
	for _, s := range []ComputeShape{{512, 2048, 4096, true}, {512, 4, 4096, true}} {
		start := uncalibrated(cpuCoeffs)
		p := start.peaks(h, s)
		measured := int64(float64(max(p.tail, p.mid))*1.3) + p.mask
		got := start.Calibrate(h, s, measured)
		if !got.Calibrated {
			t.Error("not marked calibrated")
		}
		if est := got.ComputeBytes(h, s); math.Abs(float64(est-measured)) > 1<<20 {
			t.Errorf("%+v: calibrated estimate %d, measured %d", s, est, measured)
		}
	}
}

func TestParseComputeBuffers(t *testing.T) {
	log := `0.08.110.151 I sched_reserve:        CPU compute buffer size =   112.03 MiB
llama_context:        CPU  output buffer size =     1.00 MiB
0.08.110.152 I sched_reserve:      CUDA0 compute buffer size =    96.50 MiB
0.09.000.000 I sched_reserve:        CPU compute buffer size =    25.00 MiB
`
	got, err := ParseComputeBuffers(strings.NewReader(log))
	if err != nil {
		t.Fatal(err)
	}
	cpu, cuda := 112.03, 96.5
	if len(got["CPU"]) != 2 || got["CPU"][0] != int64(cpu*mib) || got["CUDA0"][0] != int64(cuda*mib) {
		t.Errorf("%v", got)
	}
}
