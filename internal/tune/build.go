package tune

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// threadGain is how much faster a CPU team must be to replace the rule's.
const threadGain = 1.05

// BuildProfile assembles the profile of a finished run: the key, what was
// found, what was chosen and why. mx is the matrix the run was planned with.
func BuildProfile(s *Survey, rep *Report, res *Result, mx *Matrix, skipped []string, took time.Duration, now time.Time) *Profile {
	o := res.Decision.Chosen
	c := o.Candidate
	p := &Profile{
		Schema: ProfileSchema, Mode: rep.Mode, Constraints: rep.Constraints, Key: KeyNow(s, rep.Mode, c.Device), Created: now.UTC(), DurationS: took.Seconds(),
		Conditions: Conditions{Power: s.Power.Source, CPUPercent: s.Load.CPUPercent, GPUTempC: s.Load.GPUTempC, FreeRAM: s.RAM.Available},
		Survey:     s, Problems: rep.Problems, Skipped: skipped,
	}
	p.Candidates = records(rep, res)
	p.Chosen = Chosen{
		Device:   deviceValue(c, res),
		Preset:   Value{Value: c.Preset, Source: presetSource(rep.Constraints), Why: presetWhy(c, rep.Constraints), Confirmed: true},
		PlanMode: Value{Value: c.Mode, Source: SourceRule, Why: "the planner's first mode that fits the budget" + capNote(rep.Constraints), Confirmed: true},
		NCtx:     c.NCtx, Slots: c.Slots,
	}
	if c.NUMA != "" {
		nv := numaValue(c, res)
		p.Chosen.NUMA = &nv
	}
	p.Chosen.MTP = draftValue(o, c, mx)
	p.Chosen.Threads = threadsValue(o, c, rep.Constraints)
	p.Memory = MemoryCheck{
		PlanBytes: c.PlannedBytes, PrivateBytes: o.Memory.Private, VRAMBytes: o.Memory.VRAM, MeasuredBytes: o.Memory.Total,
		Source: o.Memory.Source,
	}
	if rep.Constraints.MemCap > 0 {
		p.Memory.LimitBytes, p.Memory.ToleranceBytes = rep.Constraints.MemCap, rep.Constraints.MemTolerance
	}
	p.Answer = AnswerInfo{PP: o.PP, TG: o.Best.TokS, AnswerS: o.AnswerS, Extrapolated: o.Probe != nil && (o.Probe.Prefill.Extrapolated || o.Best.Extrapolated), LoadS: o.Probe.LoadMS / 1000}
	p.Verified = verified(o, c)
	return p
}

func records(rep *Report, res *Result) []Record {
	byID := map[string]*Outcome{}
	for _, o := range res.Outcomes {
		byID[o.Candidate.ID] = o
	}
	var out []Record
	for _, c := range rep.Candidates {
		r := Record{ID: c.ID, Mode: c.Mode, PlanBytes: c.PlannedBytes, NCtx: c.NCtx, EstimateS: c.Estimate.AnswerS, Verdict: VerdictNoRun}
		if c.Reject != nil {
			r.Verdict, r.Reason = VerdictOut, c.Reject.Reason
		}
		if o := byID[c.ID]; o != nil {
			r.Measured, r.Verdict, r.Reason = o.Err == "", o.Verdict, o.Reason
			r.MeasuredBytes, r.PP, r.TG, r.AnswerS = o.Memory.Total, o.PP, o.Best.TokS, o.AnswerS
			if o.Probe != nil {
				r.LoadS = o.Probe.LoadMS / 1000
				r.Extrapolated = o.Probe.Prefill.Extrapolated || o.Best.Extrapolated
			}
		}
		out = append(out, r)
	}
	return out
}

func presetSource(k Constraints) string {
	if k.Goal == GoalMinMemory {
		return SourceMeasured
	}
	return SourceMode
}

func presetWhy(c *Candidate, k Constraints) string {
	if k.Goal == GoalMinMemory {
		return fmt.Sprintf("the smallest budget that answers at %.0f tokens per second or more", k.MinTokS)
	}
	return "derived from what the device has free"
}

func capNote(k Constraints) string {
	if k.MemCap > 0 {
		return fmt.Sprintf(", within %s of memory", gib(k.MemCap))
	}
	return ""
}

// deviceValue says why the device was chosen and what the others did.
func deviceValue(c *Candidate, res *Result) Value {
	v := Value{Value: c.Device.Selector, Source: SourceMeasured, Confirmed: true, Alternatives: map[string]float64{}}
	var rest []string
	for _, o := range res.Outcomes {
		if o.Err == "" {
			v.Alternatives[o.Candidate.ID] = o.AnswerS
		}
	}
	for _, x := range res.Report.Candidates {
		if x == c || x.Reject == nil || x.Reject.Code != RejectSlower {
			continue
		}
		v.Alternatives[x.ID] = x.Estimate.AnswerS
		rest = append(rest, fmt.Sprintf("%s ~%.0f s (estimate)", x.ID, x.Estimate.AnswerS))
	}
	why := fmt.Sprintf("a typical answer takes %.1f s here", res.Decision.Chosen.AnswerS)
	for _, o := range res.Outcomes {
		if o != res.Decision.Chosen && o.Err == "" {
			why += fmt.Sprintf(", %.1f s on %s", o.AnswerS, o.Candidate.ID)
		}
	}
	if len(rest) > 0 {
		why += "; " + strings.Join(rest, ", ")
	}
	v.Why = why
	return v
}

// draftValue is the MTP lever: the variant that decoded fastest, with the
// numbers of all that were measured.
func draftValue(o *Outcome, c *Candidate, mx *Matrix) Value {
	alts := map[string]float64{}
	for _, v := range o.Probe.Variants {
		alts[v.Name] = round1(v.TokS)
	}
	best := o.Best
	d := Draft{Enabled: best.DraftMax > 0, DraftMax: best.DraftMax, PMin: best.PMin}
	v := Value{Value: d, Source: SourceMeasured, Alternatives: alts, Confirmed: true}
	switch {
	case o.DraftMismatch:
		v.Source, v.Why = SourceRule, "drafting changed the answer on this machine, so it stays off"
	case !d.Enabled:
		v.Why = "decoding without drafting is the fastest here"
		if off, ok := o.Probe.Variant(VariantOff); ok && len(alts) > 1 {
			v.Why = fmt.Sprintf("drafting did not pay here (%.1f tok/s without it)", off.TokS)
		}
	case strings.HasPrefix(best.Name, "mech"):
		v.Source = SourceMechanism
		v.Why = "the cost of checking a batch on this device favoured this length, and the measurement confirmed it"
	default:
		if off, ok := o.Probe.Variant(VariantOff); ok && off.TokS > 0 {
			v.Why = fmt.Sprintf("%.2f times faster than without drafting", best.TokS/off.TokS)
		}
		if m, ok := mx.Lookup(mtpLever(c), c.Class); ok && m.Cell.Variant != nil && variantOf(m.Cell).Name == best.Name {
			v.Why += "; the setting of " + m.Source + " " + m.Cell.Source
		}
	}
	return v
}

func mtpLever(c *Candidate) string {
	if c.Device.Type == "cpu" {
		return "mtp_cpu"
	}
	return "mtp_gpu"
}

// threadsValue is the CPU team: the rule's, unless another measured faster.
func threadsValue(o *Outcome, c *Candidate, k Constraints) Value {
	t := Team{Threads: c.Spec.Threads, ThreadsBatch: c.Spec.ThreadsBatch}
	v := Value{Value: t, Source: SourceRule, Why: "the limit of the " + string(k.Mode) + " mode", Confirmed: true, Alternatives: map[string]float64{}}
	if c.Device.Type != "cpu" {
		v.Value, v.Why, v.Source = Team{}, "the model runs on the GPU", SourcePosture
		return v
	}
	if k.Threads == 0 {
		v.Why = "every physical core for decoding, every CPU for prompts"
	}
	v.Alternatives["rule"] = round1(o.Best.TokS)
	var win *VariantResult
	for i := range o.Probe.ThreadVariants {
		tv := &o.Probe.ThreadVariants[i]
		v.Alternatives[tv.Name] = round1(tv.TokS)
		if tv.TokS > o.Best.TokS*threadGain && (win == nil || tv.TokS > win.TokS) {
			win = tv
		}
	}
	if win != nil {
		v.Value = Team{Threads: win.Threads, ThreadsBatch: win.ThreadsBatch}
		v.Source = SourceMeasured
		v.Why = fmt.Sprintf("%s decoded %.0f%% faster than the rule's team", win.Name, 100*(win.TokS/o.Best.TokS-1))
	}
	return v
}

func verified(o *Outcome, c *Candidate) Verified {
	var vr Verified
	vr.ChosenTokS = round1(o.Best.TokS)
	if st, ok := o.Probe.Variant(staticName(c)); ok && st.TokS > 0 {
		vr.StaticTokS = round1(st.TokS)
		vr.Gain = round1(o.Best.TokS/st.TokS*100) / 100
	}
	if o.Probe.Verify != nil {
		vr.DraftChecked, vr.DraftMatch, vr.DraftNearTie = o.Probe.Verify.Checked, o.Probe.Verify.Match, o.Probe.Verify.NearTie
	}
	return vr
}

// staticName is the name of the drafting variant the server runs without a
// profile.
func staticName(c *Candidate) string {
	if c.Plan != nil && c.Plan.MTP.Enabled {
		return DraftName(c.Plan.MTP.DraftMax, c.Plan.MTP.DraftPMin)
	}
	return VariantOff
}

func round1(f float64) float64 {
	return float64(int64(f*10+0.5)) / 10
}

// SortRecords orders records for a report: chosen first, then by answer time.
func SortRecords(rs []Record) {
	sort.SliceStable(rs, func(i, j int) bool {
		if (rs[i].Verdict == VerdictChosen) != (rs[j].Verdict == VerdictChosen) {
			return rs[i].Verdict == VerdictChosen
		}
		return rs[i].AnswerS < rs[j].AnswerS
	})
}

// numaValue says which memory policy won on a machine with several nodes.
func numaValue(c *Candidate, res *Result) Value {
	v := Value{Value: c.NUMA, Source: SourceMeasured, Confirmed: true, Alternatives: map[string]float64{}}
	for _, o := range res.Outcomes {
		if o.Err == "" && o.Candidate.NUMA != "" {
			v.Alternatives[o.Candidate.NUMA] = round1(o.Best.TokS)
		}
	}
	v.Why = "the faster of the memory policies measured on this machine's nodes"
	return v
}
