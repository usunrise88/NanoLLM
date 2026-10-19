package bench

import (
	"bytes"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/plan"
)

func runOf(preset, variant string, pp, ppSD, tg, tgSD float64) benchfmt.Run {
	return benchfmt.Run{
		Preset: preset, Variant: variant, Status: benchfmt.StatusOK, Reps: 5,
		Metrics: benchfmt.Metrics{PPTokS: pp, TGTokS: benchfmt.TG{D0: tg}, StdDev: map[string]float64{"pp_tok_s": ppSD, "tg_tok_s.d0": tgSD}},
	}
}

func TestCompareVerdicts(t *testing.T) {
	a := &benchfmt.Result{Runs: []benchfmt.Run{runOf("ram-8g", "", 100, 2, 10, 0.2)}}
	b := &benchfmt.Result{Runs: []benchfmt.Run{runOf("ram-8g", "", 130, 2, 10.1, 0.2)}}
	rows := Compare(a, b)
	got := map[string]string{}
	for _, r := range rows {
		got[r.Metric] = r.Verdict
	}
	if got["pp tok/s"] != VerdictGain {
		t.Errorf("a 30%% prefill gain with a 2%% spread is %q", got["pp tok/s"])
	}
	if got["tg tok/s"] != VerdictNoise {
		t.Errorf("a 1%% decode change is %q", got["tg tok/s"])
	}
	// The same numbers the other way round are a loss; for a lower-is-better
	// metric a rise is a loss too.
	rows = Compare(b, a)
	for _, r := range rows {
		if r.Metric == "pp tok/s" && r.Verdict != VerdictLoss {
			t.Errorf("reversed: %q", r.Verdict)
		}
	}
	lo := &benchfmt.Result{Runs: []benchfmt.Run{{Preset: "p", Status: benchfmt.StatusOK, Reps: 5, Metrics: benchfmt.Metrics{TTFTMS: 100, StdDev: map[string]float64{"ttft_ms": 1}}}}}
	hi := &benchfmt.Result{Runs: []benchfmt.Run{{Preset: "p", Status: benchfmt.StatusOK, Reps: 5, Metrics: benchfmt.Metrics{TTFTMS: 150, StdDev: map[string]float64{"ttft_ms": 1}}}}}
	if r := Compare(lo, hi); len(r) != 1 || r[0].Verdict != VerdictLoss {
		t.Errorf("a slower first token is a loss: %+v", r)
	}
}

func TestCompareOneSweep(t *testing.T) {
	res := &benchfmt.Result{Runs: []benchfmt.Run{
		runOf("ram-8g", "threads=4", 50, 1, 5, 0.1),
		runOf("ram-8g", "threads=8", 90, 1, 9, 0.1),
		runOf("ram-2g", "threads=4", 40, 1, 4, 0.1),
	}}
	rows := Compare(res, nil)
	if len(rows) != 2 {
		t.Fatalf("rows: %+v", rows)
	}
	for _, r := range rows {
		if r.Preset != "ram-8g" || r.Variant != "threads=8" || r.Verdict != VerdictGain {
			t.Errorf("row %+v", r)
		}
	}
	var buf bytes.Buffer
	WriteComparison(&buf, rows)
	if !strings.Contains(buf.String(), "threads=8") || !strings.Contains(buf.String(), "gain") {
		t.Errorf("table:\n%s", buf.String())
	}
}

func TestParseSets(t *testing.T) {
	vs, err := ParseSets([]string{"threads=4,8", "kv=f16,q8_0"})
	if err != nil || len(vs) != 4 {
		t.Fatalf("%v %v", vs, err)
	}
	if vs[0].String() != "kv=f16,threads=4" || vs[3].String() != "kv=q8_0,threads=8" {
		t.Errorf("order/labels: %v", vs)
	}
	for _, bad := range []string{"threads", "threads=x", "nope=1", "kv=f32", "numa=bogus", "draft_p_min=2"} {
		if _, err := ParseSets([]string{bad}); err == nil {
			t.Errorf("%q should be refused", bad)
		}
	}
	if _, err := ParseSets([]string{"mode=M6", "mode=M7"}); err == nil {
		t.Error("a key given twice should be refused")
	}
	if vs, _ := ParseSets(nil); len(vs) != 1 || len(vs[0]) != 0 {
		t.Errorf("no axes is one empty variant: %v", vs)
	}
}

func TestParseNUMA(t *testing.T) {
	for in, want := range map[string]string{"": "off", "off": "off", "interleave": "interleave", "node:2": "node:2"} {
		n, err := ParseNUMA(in)
		if err != nil || n.String() != want {
			t.Errorf("%q: %v %v", in, n, err)
		}
	}
	for _, in := range []string{"node:", "node:x", "all", "distribute"} {
		if _, err := ParseNUMA(in); err == nil {
			t.Errorf("%q should be refused", in)
		}
	}
}

func TestVariantApplies(t *testing.T) {
	vs, err := ParseSets([]string{"n_batch=256", "kv=q8_0", "mode=M7", "mtp=off", "threads=6", "threads_batch=12", "repack=off", "load=mmap", "lazy=on", "env.OMP_PROC_BIND=close"})
	if err != nil || len(vs) != 1 {
		t.Fatal(vs, err)
	}
	v := vs[0]
	var req plan.Request
	v.Plan(&req)
	if req.NBatch != 256 || req.KVType != "q8_0" || req.Mode != plan.RAMPLEDisk || !req.NoMTP {
		t.Errorf("request %+v", req)
	}
	spec := backend.LoadSpec{Repack: true}
	v.Load(&spec)
	if spec.Threads != 6 || spec.ThreadsBatch != 12 || spec.Repack || spec.LoadMode != backend.LoadMMap || spec.LazyMode != backend.LazyOn {
		t.Errorf("spec %+v", spec)
	}
	if env := v.Env(); len(env) != 1 || env["OMP_PROC_BIND"] != "close" {
		t.Errorf("env %v", env)
	}
	if !strings.Contains(v.String(), "env.OMP_PROC_BIND=close") {
		t.Errorf("label %q", v.String())
	}
}

func TestPlacementSpec(t *testing.T) {
	for _, tc := range []struct {
		p    Placement
		want string
	}{
		{Placement{}, "off"},
		{Placement{NUMA: "interleave"}, "interleave"},
		{Placement{NUMA: "node:1", CPUs: "4-7"}, "node:1;cpus=4-7"},
		{Placement{CPUs: "0-3"}, "off;cpus=0-3"},
	} {
		if got := tc.p.Spec(); got != tc.want {
			t.Errorf("%+v: %q, want %q", tc.p, got, tc.want)
		}
	}
	if !(Placement{}).IsZero() || !(Placement{NUMA: "off"}).IsZero() || (Placement{CPUs: "0"}).IsZero() {
		t.Error("IsZero")
	}
}
