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
		{"ub512", false, ComputeShape{512, 2048, 4096, true, false}, 563.03, 0.005},
		{"ub256", false, ComputeShape{256, 2048, 4096, true, false}, 281.39, 0.005},
		{"ub128 b128", false, ComputeShape{128, 128, 4096, true, false}, 140.70, 0.005},
		{"c32768", false, ComputeShape{512, 2048, 32768, true, false}, 591.03, 0.005},
		{"np4 c8192", false, ComputeShape{512, 2048, 2048, true, false}, 561.03, 0.005},
		{"no FA", false, ComputeShape{512, 2048, 4096, false, false}, 576.02, 0.01},
		{"no FA c32768", false, ComputeShape{512, 2048, 32768, false, false}, 658.02, 0.01},
		{"mtp target", false, ComputeShape{512, 4, 4096, true, false}, 112.03, 0.02},
		{"mtp target ub256", false, ComputeShape{256, 4, 4096, true, false}, 55.89, 0.02},
		{"mtp target np4", false, ComputeShape{512, 16, 4096, true, false}, 112.03, 0.02},
		{"mtp target ub128", false, ComputeShape{128, 4, 4096, true, false}, 27.94, 0.02},
		{"mtp target c32768", false, ComputeShape{512, 4, 32768, true, false}, 140.03, 0.02},
		{"mtp drafter", true, ComputeShape{512, 1, 4096, true, false}, 25.00, 0.06},
		{"mtp drafter ub256", true, ComputeShape{256, 1, 4096, true, false}, 12.38, 0.06},
		{"mtp drafter ub128", true, ComputeShape{128, 1, 4096, true, false}, 6.19, 0.15}, // 0.8 MiB high
		{"mtp drafter c32768", true, ComputeShape{512, 1, 32768, true, false}, 53.00, 0.02},
		{"mtp target no FA", false, ComputeShape{512, 4, 4096, false, false}, 154.02, 0.02},
		{"mtp target no FA c32768", false, ComputeShape{512, 4, 32768, false, false}, 658.02, 0.02},
		// The drafter's KQ peak carries less besides the matrix than the
		// target's; the shared fit errs high by 13 MiB.
		{"mtp drafter no FA", true, ComputeShape{512, 1, 4096, false, false}, 51.00, 0.30},
		{"mtp drafter no FA c32768", true, ComputeShape{512, 1, 32768, false, false}, 331.00, 0.05},
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

// TestComputeSmallBatchFA: flash attention on small batches only keeps the
// f16 mask of flash attention and adds the KQ peak of the matmul attention.
func TestComputeSmallBatchFA(t *testing.T) {
	h := e4b(t).Target.HParams
	for _, nkv := range []int{4096, 32768} {
		fa := ComputeShape{NUBatch: 512, NOutputsMax: 4, NKV: nkv, FlashAttention: true}
		small := fa
		small.FlashAttnSmallBatch = true
		noFA := fa
		noFA.FlashAttention = false
		pf, ps, pn := cpuCoeffs.peaks(h, fa), cpuCoeffs.peaks(h, small), cpuCoeffs.peaks(h, noFA)
		if pf.kq != 0 || ps.kq != pn.kq || ps.mask != pf.mask {
			t.Errorf("n_kv %d: kq %d/%d/%d, mask %d/%d (flash attention / small batch / none)", nkv, pf.kq, ps.kq, pn.kq, pf.mask, ps.mask)
		}
		if cpuCoeffs.ComputeBytes(h, small) <= cpuCoeffs.ComputeBytes(h, fa) {
			t.Errorf("n_kv %d: the small-batch buffer is not larger than flash attention's", nkv)
		}
	}
}

func TestCalibrate(t *testing.T) {
	h := e4b(t).Target.HParams
	// A GPU whose activations run 30% above the CPU fit, at each peak.
	for _, s := range []ComputeShape{{512, 2048, 4096, true, false}, {512, 4, 4096, true, false}, {512, 4, 32768, false, false}} {
		start := uncalibrated(cpuCoeffs)
		p := start.peaks(h, s)
		measured := int64(float64(max(p.tail, p.mid, p.kq))*1.3) + p.mask
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
