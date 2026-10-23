package tune

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/plan"
)

var update = flag.Bool("update", false, "rewrite the golden decisions in testdata/tune/decide")

const decideGolden = "../../testdata/tune/decide"

// A synthetic machine and its candidates, from the facts of docs/AUTOTUNE.md
// and the measurements of bench/results/*/optimize-stage0-*.
type fixture struct {
	name  string
	mode  Mode
	cores Cores
	disk  *hwinfo.DiskSpeed
	cands []*Candidate
	probe map[string]ProbeResult
	err   map[string]string
}

func cpuDev() Device {
	return Device{Selector: "llamacpp:cpu", Kind: "cpu", Type: "cpu", Name: "CPU", Backend: "llamacpp", ID: "cpu"}
}

func cudaDev() Device {
	return Device{Selector: "llamacpp:cuda0", Kind: "cuda", Type: "gpu", Name: "RTX 3070 Ti Laptop", Vendor: "NVIDIA", Backend: "llamacpp", ID: "cuda0", MemoryTotal: 8 * GiB, MemoryFree: 7 * GiB, Features: []string{"cuda-nano"}}
}

func irisDev() Device {
	return Device{Selector: "llamacpp:vulkan1", Kind: "vulkan", Type: "gpu", Name: "Iris Xe", Vendor: "Intel", Backend: "llamacpp", ID: "vulkan1", Unified: true, MemoryTotal: 16 * GiB, MemoryFree: 15 * GiB}
}

func cand(dev Device, preset, mode string, planPrivate, planVRAM int64, cold int64) *Candidate {
	c := &Candidate{ID: dev.Selector, Device: dev, Preset: preset, Mode: mode, NCtx: 8192, Slots: 1, Measure: true,
		PlannedPrivate: planPrivate, PlannedVRAM: planVRAM, PlannedBytes: planPrivate + planVRAM, Plan: &plan.Plan{}}
	if preset != "" && dev.Type == "cpu" {
		c.ID += "/" + preset
	}
	if cold > 0 {
		c.Plan.StreamRanges = []plan.StreamRange{{Length: cold}}
	}
	return c
}

func vr(name string, max int, pMin, tokS float64) VariantResult {
	return VariantResult{Name: name, DraftMax: max, PMin: pMin, TokS: tokS}
}

func probe(mem MemoryResult, pp float64, vs ...VariantResult) ProbeResult {
	return ProbeResult{Memory: mem, Prefill: PrefillResult{Tokens: 4096, PP: pp, PP512: pp}, Variants: vs, LoadMS: 3000}
}

func summarize(f fixture) string {
	s := &Survey{Devices: nil, Disk: DiskInfo{Speed: f.disk}}
	k, _ := ConstraintsFor(f.mode, f.cores, 20*GiB)
	var outs []*Outcome
	for _, c := range f.cands {
		pr := f.probe[c.ID]
		var err error
		if m, bad := f.err[c.ID]; bad {
			err = fmt.Errorf("%s", m)
		}
		outs = append(outs, Evaluate(c, &pr, err, s, k))
	}
	d := Decide(outs, s, k)
	var b strings.Builder
	fmt.Fprintf(&b, "mode %s\n", f.mode)
	for _, o := range outs {
		fmt.Fprintf(&b, "  %-26s %-24s", o.Candidate.ID, o.Verdict)
		if o.Err == "" {
			fmt.Fprintf(&b, " answer %6.1f s  decode %6.1f  held %5d MiB  %s", o.AnswerS, o.Best.TokS, o.Memory.Total>>20, o.Best.Name)
		}
		if o.Reason != "" {
			fmt.Fprintf(&b, "  (%s)", o.Reason)
		}
		fmt.Fprintln(&b)
	}
	if d.Chosen != nil {
		fmt.Fprintf(&b, "chosen %s\n", d.Chosen.Candidate.ID)
	} else {
		fmt.Fprintln(&b, "chosen nothing")
	}
	var codes []string
	for _, p := range d.Problems {
		codes = append(codes, string(p.Code)+"/"+string(p.Severity))
	}
	fmt.Fprintf(&b, "problems %s\n", strings.Join(codes, " "))
	return b.String()
}

func fixtures() []fixture {
	rtxMem := MemoryResult{PeakPrivate: 695 * int64(MiB), VRAMDedicated: 3420 * int64(MiB), VRAMShared: 176 * int64(MiB), VRAMSource: "pdh"}
	cpuMem := MemoryResult{PeakPrivate: 3374 * int64(MiB)}
	nvme := &hwinfo.DiskSpeed{SeqMiBs: 3295}
	hdd := &hwinfo.DiskSpeed{SeqMiBs: 124}
	rtx := func() *Candidate {
		c := cand(cudaDev(), "", "M3 vram-ple-disk", 223*MiB+plan.GPUHostMemory, 3449*MiB, 0)
		c.DriverHost = plan.GPUHostMemory
		return c
	}
	cpu := func(preset string, mib int64) *Candidate {
		return cand(cpuDev(), preset, "M7 ram-ple-disk", mib*MiB, 0, 0)
	}
	cores := Cores{Physical: 14, Logical: 20}

	over := rtxMem
	over.PeakPrivate = 1000 * int64(MiB)
	spill := rtxMem
	spill.VRAMShared = 2084 * int64(MiB)

	return []fixture{
		{
			// The RTX within the optimal mode's limit beats the CPU.
			name: "rusunrisebook-optimal", mode: Optimal, cores: cores, disk: nvme,
			cands: []*Candidate{rtx(), cpu("", 3500)},
			probe: map[string]ProbeResult{
				"llamacpp:cuda0": probe(rtxMem, 6012, vr("off", 0, 0, 94), vr("mtp3/0", 3, 0, 179.8), vr("mtp2/0", 2, 0, 164.8)),
				"llamacpp:cpu":   probe(cpuMem, 148, vr("off", 0, 0, 10.3), vr("mtp2/0.5", 2, 0.5, 19.1)),
			},
		},
		{
			// The card measured over the limit: the CPU is next.
			name: "rusunrisebook-optimal-over", mode: Optimal, cores: cores, disk: nvme,
			cands: []*Candidate{rtx(), cpu("", 3500)},
			probe: map[string]ProbeResult{
				"llamacpp:cuda0": probe(over, 6012, vr("off", 0, 0, 94), vr("mtp3/0", 3, 0, 179.8)),
				"llamacpp:cpu":   probe(cpuMem, 148, vr("off", 0, 0, 10.3), vr("mtp2/0.5", 2, 0.5, 19.1)),
			},
		},
		{
			// Nothing within the limit: the leanest is used and the mode is degraded.
			name: "optimal-nothing-fits", mode: Optimal, cores: cores, disk: nvme,
			cands: []*Candidate{rtx()},
			probe: map[string]ProbeResult{"llamacpp:cuda0": probe(over, 6012, vr("off", 0, 0, 94))},
		},
		{
			name: "rusunrisebook-powerful", mode: Powerful, cores: cores, disk: nvme,
			cands: []*Candidate{rtx(), cpu("", 3500)},
			probe: map[string]ProbeResult{
				"llamacpp:cuda0": probe(rtxMem, 6012, vr("off", 0, 0, 94), vr("mtp3/0", 3, 0, 179.8)),
				"llamacpp:cpu":   probe(cpuMem, 148, vr("off", 0, 0, 10.3), vr("mtp2/0.5", 2, 0.5, 19.1)),
			},
		},
		{
			// A card that spilled into shared memory loses to the CPU.
			name: "powerful-spill", mode: Powerful, cores: cores, disk: nvme,
			cands: []*Candidate{rtx(), cpu("", 3500)},
			probe: map[string]ProbeResult{
				"llamacpp:cuda0": probe(spill, 267, vr("off", 0, 0, 65)),
				"llamacpp:cpu":   probe(cpuMem, 148, vr("off", 0, 0, 10.3), vr("mtp2/0.5", 2, 0.5, 19.1)),
			},
		},
		{
			// Quiet mode on an NVMe: the smallest budget that works, even when it streams.
			name: "silent-nvme", mode: Silent, cores: cores, disk: nvme,
			cands: []*Candidate{
				cand(cpuDev(), "ram-1g", "M9 mmap", 561*MiB, 0, 2131*MiB),
				cpu("ram-4g", 3362),
			},
			probe: map[string]ProbeResult{
				"llamacpp:cpu/ram-1g": probe(MemoryResult{PeakPrivate: 611 * int64(MiB)}, 21, vr("off", 0, 0, 2.7)),
				"llamacpp:cpu/ram-4g": probe(MemoryResult{PeakPrivate: 3374 * int64(MiB)}, 90, vr("off", 0, 0, 11.0)),
			},
		},
		{
			// Quiet mode on a hard disk: the page cache flatters the probe, the
			// disk formula does not, and the setup that needs no disk wins.
			name: "silent-hdd", mode: Silent, cores: Cores{Physical: 16, Logical: 32}, disk: hdd,
			cands: []*Candidate{
				cand(cpuDev(), "ram-1g", "M9 mmap", 561*MiB, 0, 2131*MiB),
				cand(cpuDev(), "ram-3g", "M7 ram-ple-disk", 3010*MiB, 0, 0),
			},
			probe: map[string]ProbeResult{
				"llamacpp:cpu/ram-1g": probe(MemoryResult{PeakPrivate: 611 * int64(MiB)}, 21, vr("off", 0, 0, 2.7)),
				"llamacpp:cpu/ram-3g": probe(MemoryResult{PeakPrivate: 3030 * int64(MiB)}, 30, vr("off", 0, 0, 8.4)),
			},
		},
		{
			// Nothing reaches one token per second.
			name: "silent-nothing-works", mode: Silent, cores: Cores{Physical: 2, Logical: 2}, disk: hdd,
			cands: []*Candidate{cand(cpuDev(), "ram-4g", "M7 ram-ple-disk", 3362*MiB, 0, 0)},
			probe: map[string]ProbeResult{"llamacpp:cpu/ram-4g": probe(cpuMem, 4, vr("off", 0, 0, 0.6))},
		},
		{
			// A drafter that changes the answer is switched off.
			name: "draft-mismatch", mode: Powerful, cores: cores, disk: nvme,
			cands: []*Candidate{cand(cpuDev(), "", "M6 ram-full", 6000*MiB, 0, 0)},
			probe: func() map[string]ProbeResult {
				p := probe(MemoryResult{PeakPrivate: 6000 * int64(MiB)}, 148, vr("off", 0, 0, 10.3), vr("mtp2/0.5", 2, 0.5, 19.1))
				p.Verify = &VerifyResult{Checked: true, Match: false, At: 7, Tokens: 32}
				return map[string]ProbeResult{"llamacpp:cpu": p}
			}(),
		},
		{
			// A probe that failed leaves the others to decide.
			name: "failed-probe", mode: Powerful, cores: cores, disk: nvme,
			cands: []*Candidate{rtx(), cpu("", 3500)},
			probe: map[string]ProbeResult{"llamacpp:cpu": probe(cpuMem, 148, vr("off", 0, 0, 10.3))},
			err:   map[string]string{"llamacpp:cuda0": "worker: exit status 2"},
		},
	}
}

func TestDecideGolden(t *testing.T) {
	for _, f := range fixtures() {
		t.Run(f.name, func(t *testing.T) {
			got := []byte(summarize(f))
			path := filepath.Join(decideGolden, f.name+".txt")
			if *update {
				if err := os.MkdirAll(decideGolden, 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, got, 0o644); err != nil {
					t.Fatal(err)
				}
				return
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("%v (run with -update to create it)", err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("the decision for %s differs from %s:\n%s\nwant:\n%s\nrerun with -update and review the diff", f.name, path, got, want)
			}
		})
	}
}

// The decisions, said in words: the golden files are for diffs, these are for
// the rules of docs/OPTIMIZE.md §3.4 stated once.
func TestDecideRules(t *testing.T) {
	chosen := map[string]string{}
	problems := map[string][]string{}
	for _, f := range fixtures() {
		out := summarize(f)
		for _, line := range strings.Split(out, "\n") {
			if rest, ok := strings.CutPrefix(line, "chosen "); ok {
				chosen[f.name] = rest
			}
			if rest, ok := strings.CutPrefix(line, "problems "); ok {
				problems[f.name] = strings.Fields(rest)
			}
		}
	}
	for name, want := range map[string]string{
		"rusunrisebook-optimal":      "llamacpp:cuda0",
		"rusunrisebook-optimal-over": "llamacpp:cpu",
		"optimal-nothing-fits":       "llamacpp:cuda0",
		"rusunrisebook-powerful":     "llamacpp:cuda0",
		"powerful-spill":             "llamacpp:cpu",
		"silent-nvme":                "llamacpp:cpu/ram-1g",
		"silent-hdd":                 "llamacpp:cpu/ram-3g",
		"silent-nothing-works":       "nothing",
		"draft-mismatch":             "llamacpp:cpu",
		"failed-probe":               "llamacpp:cpu",
	} {
		if chosen[name] != want {
			t.Errorf("%s: chosen %q, want %q", name, chosen[name], want)
		}
	}
	for name, want := range map[string]string{
		"optimal-nothing-fits": "mode_degraded/error",
		"powerful-spill":       "vram_spill/warn",
		"silent-nothing-works": "silent_unavailable/error",
		"draft-mismatch":       "draft_mismatch/warn",
	} {
		found := false
		for _, p := range problems[name] {
			if p == want {
				found = true
			}
		}
		if !found {
			t.Errorf("%s: problems %v lack %s", name, problems[name], want)
		}
	}
}

// The spill rule on the three runs of bench/results/rusunrisebook/optimize-stage0-2026-10-23:
// shared video memory above the plan's host memory (not counting the driver's)
// by more than the slack is a spill.
func TestSpillRule(t *testing.T) {
	mib := int64(MiB)
	rtx := func(host int64) *Candidate {
		c := cand(cudaDev(), "", "M", host*mib+plan.GPUHostMemory, 3449*mib, 0)
		c.DriverHost = plan.GPUHostMemory
		return c
	}
	pdh := func(shared int64) MemoryResult { return MemoryResult{VRAMShared: shared * mib, VRAMSource: "pdh"} }
	for _, tc := range []struct {
		name   string
		c      *Candidate
		m      MemoryResult
		spill  bool
		reason string
	}{
		{"M2 at vram-4g", rtx(2131), pdh(2084), false, "its pinned memory is the plan's"},
		{"M3 at vram-4g", rtx(223), pdh(176), false, ""},
		{"M1 at vram-8g next to the desktop", rtx(274), pdh(698), true, "pp 267 instead of ~5000"},
		{"no counter", rtx(274), MemoryResult{VRAMShared: 698 * mib}, false, "without the counter there is nothing to compare"},
	} {
		if got := spilled(tc.c, tc.m); got != tc.spill {
			t.Errorf("%s: spilled %v, want %v (%s)", tc.name, got, tc.spill, tc.reason)
		}
	}
	// Unified memory has no separate video memory to run out of.
	iris := cand(irisDev(), "", "M3", 3000*mib, 0, 0)
	if spilled(iris, pdh(3500)) {
		t.Error("an iGPU cannot spill")
	}
}
