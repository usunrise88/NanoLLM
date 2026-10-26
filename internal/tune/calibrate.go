package tune

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sort"
	"time"
)

// CalibrationReport is a device measured the long way, in the terms of the
// matrix's baselines.
type CalibrationReport struct {
	Candidate *Candidate      `json:"candidate"`
	Probe     ProbeResult     `json:"probe"`
	Baseline  Baseline        `json:"baseline"`
	Best      string          `json:"best_variant"`
	Mechanism map[int]float64 `json:"mechanism,omitempty"`
}

// Calibrate measures one device without the shortcuts of a run: the whole
// reference prompt, decode at both depths, every drafting variant and the cost
// of a batch. Its output is what a baseline of matrix.yaml holds, and the data
// for checking that the cost of a batch predicts the best draft length
// (docs/OPTIMIZE.md, §3.8). sel names the device ("" is the first that
// can run); threads, when above zero, overrides a CPU's decode team; low
// measures under the quiet mode's priority (below normal, EcoQoS).
func Calibrate(ctx context.Context, o Options, sel string, threads int, low bool, r Runner) (*CalibrationReport, error) {
	s := TakeSurvey(ctx, o.Survey)
	k, _ := ConstraintsFor(Powerful, Cores{Physical: s.CPU.Physical, Logical: s.CPU.Logical}, 0)
	cands, _ := BuildCandidates(s, k, o.Plan)
	var c *Candidate
	for _, x := range cands {
		if x.Plan == nil || x.Reject != nil && x.Reject.Code != RejectSlower && x.Reject.Code != RejectNoDevice {
			continue
		}
		if sel == "" || x.Device.Selector == sel || x.ID == sel {
			c = x
			break
		}
	}
	if c == nil {
		return nil, fmt.Errorf("calibrate: no device %q can be planned", sel)
	}
	c.Reject = nil
	if threads > 0 && c.Device.Type == "cpu" {
		c.Spec.Threads, c.Spec.ThreadsBatch = threads, threads
	}
	spec := BuildProbeSpec(s, k, o.Plan, c, o)
	spec.Calibrate, spec.Verify, spec.Slow = true, false, false
	spec.LowPriority = low
	spec.Threads = nil
	spec.Budget = 20 * time.Minute
	pr, err := r.Probe(ctx, spec)
	if err != nil {
		return nil, err
	}
	cal := pr.Calibration
	if cal == nil {
		return nil, errors.New("calibrate: the worker returned no calibration")
	}
	rep := &CalibrationReport{Candidate: c, Probe: pr, Mechanism: pr.BatchMS}
	b := Baseline{
		Device: deviceKindOf(c.Device), Class: c.Class, Threads: c.Spec.Threads,
		PP512: round1(cal.PP512), TG0: round1(cal.TG0), DepthPP: round2(cal.DepthPP), DepthTG: round2(cal.DepthTG),
		Source: "bench/results/<machine>/optimize-stage0-<date>/calibrate-<device>.json",
	}
	b.MTPGain = 1
	bestGain := 1.0
	var names []string
	for n := range cal.Variants {
		names = append(names, n)
	}
	sort.Strings(names)
	// The gain is judged at the depth of the reference request, where the
	// drafts of a long chat are made.
	for _, n := range names {
		if v := cal.Variants[n]; cal.TG4096 > 0 && v[1]/cal.TG4096 > bestGain {
			bestGain, rep.Best = v[1]/cal.TG4096, n
		}
	}
	b.MTPGain = round2(bestGain)
	rep.Baseline = b
	return rep, nil
}

func round2(f float64) float64 { return float64(int64(f*100+0.5)) / 100 }

// WriteText prints the calibration as a baseline entry and the data behind it.
func (r *CalibrationReport) WriteText(w io.Writer) {
	c, cal := r.Candidate, r.Probe.Calibration
	fmt.Fprintf(w, "calibration of %s (%s, %s, context %d)\n\n", c.Device.Selector, c.Device.Name, c.Mode, c.NCtx)
	fmt.Fprintf(w, "  prefill   %.1f tok/s over the first %d tokens, %.1f tok/s over %d (x%.2f)\n", cal.PP512, ShortPrefix, cal.PP4096, r.Probe.Prefill.Tokens, cal.DepthPP)
	fmt.Fprintf(w, "  decode    %.1f tok/s at an empty context, %.1f after the prompt (x%.2f); %.1f after the first %d tokens\n", cal.TG0, cal.TG4096, cal.DepthTG, cal.ShortTG, ShortPrefix)
	var names []string
	for n := range cal.Variants {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		v := cal.Variants[n]
		fmt.Fprintf(w, "  %-9s %.1f tok/s at 0, %.1f at depth (x%.2f of plain)\n", n, v[0], v[1], v[1]/max(cal.TG4096, 1e-9))
	}
	if len(r.Mechanism) > 0 {
		cost := BatchCost(r.Mechanism)
		fmt.Fprintf(w, "  batch     ")
		for _, k := range []int{1, 2, 3, 4} {
			fmt.Fprintf(w, "c(%d)=%.2f  ", k, cost(k))
		}
		fmt.Fprintln(w)
		d := DraftCostFraction(c.Device.Kind)
		fmt.Fprintf(w, "  expected  ")
		bestK, bestE := 0, 1.0
		for k := 1; k <= 3; k++ {
			e := ExpectedSpeedup(k, cost, DefaultAcceptance, d)
			fmt.Fprintf(w, "E(%d)=%.2f  ", k, e)
			if e > bestE {
				bestK, bestE = k, e
			}
		}
		fmt.Fprintf(w, "-> draft %d; measured best: %s\n", bestK, r.Best)
	}
	b := r.Baseline
	fmt.Fprintf(w, "\nbaseline for matrix.yaml:\n  - device: %s\n    class: %v\n    threads: %d\n    pp512: %v\n    tg0: %v\n    depth_pp: %v\n    depth_tg: %v\n    mtp_gain: %v\n",
		b.Device, b.Class, b.Threads, b.PP512, b.TG0, b.DepthPP, b.DepthTG, b.MTPGain)
}
