package tune

import (
	"fmt"
	"math"
	"sort"

	"github.com/usunrise88/nanollm/internal/backend"
)

// Outcome is a candidate after its probe: what was measured and what follows
// from it.
type Outcome struct {
	Candidate *Candidate   `json:"-"`
	Probe     *ProbeResult `json:"probe,omitempty"`
	// Err is why there is no measurement: the worker failed or ran out of time.
	Err string `json:"err,omitempty"`

	// Memory is what the process held, counted the way the mode's limit counts
	// it.
	Memory MeasuredMemory `json:"memory"`
	// Best is the fastest drafting variant, PP the prefill rate at the
	// reference depth and AnswerS the time of the reference request with them.
	Best    VariantResult `json:"best"`
	PP      float64       `json:"pp"`
	AnswerS float64       `json:"answer_s"`
	// Spill says the driver put part of the model in system memory.
	Spill bool `json:"spill,omitempty"`
	// DraftMismatch says drafting changed the answer on this setup.
	DraftMismatch bool `json:"draft_mismatch,omitempty"`
	// DiskLimited says the decode rate is the disk formula's, not the probe's.
	DiskLimited bool `json:"disk_limited,omitempty"`
	// Verdict and Reason are the outcome for the report.
	Verdict string `json:"verdict"`
	Reason  string `json:"reason,omitempty"`
}

// MeasuredMemory is the memory of the process as the limit counts it.
type MeasuredMemory struct {
	Private int64 `json:"private"`
	VRAM    int64 `json:"vram"`
	Total   int64 `json:"total"`
	// Source says where the video memory comes from: pdh, nvidia-smi, or
	// buffers (the engine's own sum plus the driver context).
	Source string `json:"source,omitempty"`
}

// Verdicts of an outcome.
const (
	VerdictChosen  = "chosen"
	VerdictOut     = "out"
	VerdictSlower  = "slower"
	VerdictNoRun   = "not measured"
	VerdictFailed  = "failed"
	VerdictTooBig  = "over the memory limit"
	VerdictTooSlow = "below the minimum speed"
)

// SpillSlack is how much Shared Usage beyond the host memory the plan pinned is
// not yet a spill: the driver keeps small staging buffers there. A spill from
// an oversubscribed card measured at +500 MiB or more, a plan that fits at -47
// (bench/results/rusunrisebook/optimize-stage0-2026-10-23).
const SpillSlack = 256 * MiB

// Evaluate derives an outcome from a probe: the memory by the mode's counting,
// the speed at the reference depth, the checks.
func Evaluate(c *Candidate, pr *ProbeResult, err error, s *Survey, k Constraints) *Outcome {
	o := &Outcome{Candidate: c, Probe: pr}
	if err != nil {
		o.Err = err.Error()
		o.Verdict, o.Reason = VerdictFailed, o.Err
		return o
	}
	best, ok := bestOf(pr, staticName(c))
	if !ok {
		o.Err = "the probe measured no decode"
		o.Verdict, o.Reason = VerdictFailed, o.Err
		return o
	}
	o.Best, o.PP = best, pr.Prefill.PP
	o.Memory = measuredMemory(c, pr)
	o.Spill = spilled(c, pr.Memory)
	o.DraftMismatch = pr.Verify != nil && pr.Verify.Checked && !pr.Verify.Match
	// A setup that reads weights from the disk is limited by it whatever the
	// probe's few tokens say (the page cache flatters a short probe).
	if cold := ColdBytesPerToken(c.Plan); cold > 0 && s.Disk.Speed != nil {
		if limit := DiskTokS(s.Disk.Speed.SeqMiBs, cold); limit > 0 && limit < o.Best.TokS {
			o.Best.TokS, o.DiskLimited = limit, true
		}
	}
	o.AnswerS = AnswerSeconds(o.PP, o.Best.TokS)
	return o
}

// StaticMargin is how much faster another drafting setting must be than the one
// the server runs without a profile to replace it: a short probe has several
// percent of noise, and the rule's setting has the whole of the field behind it.
const StaticMargin = 1.05

// bestOf is the fastest drafting variant of a probe, except that the static one
// stays unless another is clearly faster.
func bestOf(pr *ProbeResult, static string) (VariantResult, bool) {
	best, ok := pr.Best()
	if !ok {
		return best, false
	}
	if st, has := pr.Variant(static); has && st.Name != best.Name && best.TokS < st.TokS*StaticMargin {
		return st, true
	}
	return best, true
}

// measuredMemory counts the process's memory the way the limit does
// (docs/OPTIMIZE.md, §2.1): the private resident memory plus the video memory
// of a discrete GPU. On shared memory the GPU buffers are in the private
// figure already. Range-locked pages of the model file are not private memory
// but the plan budgets them, so they are added.
func measuredMemory(c *Candidate, pr *ProbeResult) MeasuredMemory {
	m := pr.Memory
	out := MeasuredMemory{Private: m.PeakPrivate}
	if c.Plan != nil {
		out.Private += lockedBytes(c)
	}
	switch {
	case c.Device.Type == string(backend.CPU) || c.Device.Unified:
		// no separate video memory
	case m.VRAMSource != "":
		out.VRAM, out.Source = m.VRAMDedicated, m.VRAMSource
	default:
		// No counter reads the process: the engine's own buffers and the
		// driver's context stand in.
		out.VRAM, out.Source = m.Buffers+gpuContext(c.Device), "buffers"
	}
	out.Total = out.Private + out.VRAM
	return out
}

// gpuContext is the video memory a driver takes for a process before its first
// buffer: measured 262 MiB on CUDA (3424 MiB held against 3162 of buffers).
func gpuContext(d Device) int64 {
	if d.Kind == "cuda" {
		return 288 * MiB
	}
	return 160 * MiB
}

// lockedBytes is the model file's range-locked pages, which the OS counts with
// the file and not with private memory.
func lockedBytes(c *Candidate) int64 {
	var n int64
	for _, r := range c.Plan.LockRanges {
		n += r.Length
	}
	return n
}

// spilled: a discrete GPU whose process holds more system memory in "shared"
// video memory than the plan pinned on the host means the driver ran out of
// video memory and placed part of the model in system memory, which costs most
// of the speed (vram-8g on an RTX 3070 Ti next to a desktop: pp 267 against
// ~5000, tg 65 against ~100).
func spilled(c *Candidate, m MemoryResult) bool {
	if !c.Device.Discrete() || m.VRAMSource != "pdh" {
		return false
	}
	return m.VRAMShared > c.PlannedPrivate-c.DriverHost+SpillSlack
}

// Decision is the choice among the outcomes.
type Decision struct {
	Chosen   *Outcome
	Outcomes []*Outcome
	Problems []Problem
}

// tieBand is how close two answer times may be to count as equal: the noise of
// a short probe (docs/OPTIMIZE.md, §3.3).
const tieBand = 0.05

// Decide picks the setup by the rule of the mode (docs/OPTIMIZE.md, §3.4) and
// marks the outcomes with their verdicts.
func Decide(outs []*Outcome, s *Survey, k Constraints) *Decision {
	d := &Decision{Outcomes: outs}
	var ok []*Outcome
	for _, o := range outs {
		switch {
		case o.Verdict == VerdictFailed:
		case o.Spill:
			o.Verdict, o.Reason = VerdictOut, "the driver moved part of the model to system memory"
			d.Problems = append(d.Problems, Problem{
				Code: VRAMSpill, Severity: SevWarn,
				Message: fmt.Sprintf("%s did not fit its video memory: the driver put %s of the model in shared memory, so a smaller setup was chosen.", o.Candidate.Device.Name, gib(o.Probe.Memory.VRAMShared)),
				Params:  map[string]any{"device": o.Candidate.Device.Selector, "shared_bytes": o.Probe.Memory.VRAMShared},
			})
		case o.DraftMismatch:
			// Drafting changed the answer: the setup stays, without drafting.
			d.mismatch(o)
			ok = append(ok, o)
		case o.Best.TokS < k.MinTokS:
			o.Verdict, o.Reason = VerdictTooSlow, fmt.Sprintf("%.2f tokens per second after a long prompt, the minimum is %.0f", o.Best.TokS, k.MinTokS)
		default:
			ok = append(ok, o)
		}
	}

	limit := int64(0)
	if k.MemCap > 0 {
		limit = k.MemCap + k.MemTolerance
	}
	var fit []*Outcome
	for _, o := range ok {
		if limit > 0 && o.Memory.Total > limit {
			o.Verdict = VerdictTooBig
			o.Reason = fmt.Sprintf("held %s, the limit is %s plus %s", gib(o.Memory.Total), gib(k.MemCap), gib(k.MemTolerance))
			continue
		}
		fit = append(fit, o)
	}

	switch {
	case len(fit) > 0:
		d.Chosen = pick(fit, k)
	case len(ok) > 0 && limit > 0:
		// Every setup that works holds too much: the leanest is used and the
		// limit is reported as not met.
		sort.SliceStable(ok, func(i, j int) bool { return ok[i].Memory.Total < ok[j].Memory.Total })
		d.Chosen = ok[0]
		d.Problems = append(d.Problems, Problem{
			Code: ModeDegraded, Severity: SevError,
			Message: fmt.Sprintf("No setup stays within %s plus %s: the leanest held %s.", gib(k.MemCap), gib(k.MemTolerance), gib(ok[0].Memory.Total)),
			Params:  map[string]any{"limit_bytes": k.MemCap, "held_bytes": ok[0].Memory.Total},
		})
	case k.Mode == Silent:
		d.Problems = append(d.Problems, Problem{
			Code: SilentUnavailable, Severity: SevError,
			Message: "The quiet mode is unusable on this machine: no setup answers at the minimum speed.",
			Advice:  "Try the optimal mode.",
		})
	}
	for _, o := range outs {
		if o == d.Chosen {
			o.Verdict, o.Reason = VerdictChosen, ""
			continue
		}
		if o.Verdict == "" {
			o.Verdict = VerdictSlower
			if d.Chosen != nil {
				o.Reason = fmt.Sprintf("%.1f s for a typical answer against %.1f s", o.AnswerS, d.Chosen.AnswerS)
				if k.Goal == GoalMinMemory {
					o.Reason = fmt.Sprintf("holds %s against %s", gib(o.Memory.Total), gib(d.Chosen.Memory.Total))
				}
			}
		}
	}
	return d
}

// mismatch records that drafting changed the answer and turns it off for the
// outcome: its best setting is the one without drafting.
func (d *Decision) mismatch(o *Outcome) {
	d.Problems = append(d.Problems, Problem{
		Code: DraftMismatch, Severity: SevWarn,
		Message: fmt.Sprintf("On %s drafting gave a different answer than decoding without it (the first difference at token %d): MTP is off there.", o.Candidate.Device.Name, o.Probe.Verify.At),
		Advice:  "Send the profile (nanollm optimize --json) so the cause can be found.",
		Params:  map[string]any{"device": o.Candidate.Device.Selector, "at": o.Probe.Verify.At},
	})
	if v, ok := o.Probe.Variant("off"); ok {
		o.Best = v
		o.AnswerS = AnswerSeconds(o.PP, v.TokS)
	}
}

// pick chooses among the outcomes that meet the limits.
func pick(fit []*Outcome, k Constraints) *Outcome {
	best := fit[0]
	for _, o := range fit[1:] {
		if better(o, best, k) {
			best = o
		}
	}
	return best
}

// better says o beats cur under the mode's goal. Within tieBand of each other
// the answers tie, and the tie goes to less memory, then a GPU (the processor
// stays free), then no drafting (a smaller drafter in memory).
func better(o, cur *Outcome, k Constraints) bool {
	if k.Goal == GoalMinMemory {
		if o.Memory.Total != cur.Memory.Total {
			return o.Memory.Total < cur.Memory.Total
		}
		return o.AnswerS < cur.AnswerS
	}
	if math.Abs(o.AnswerS-cur.AnswerS) > tieBand*math.Min(o.AnswerS, cur.AnswerS) {
		return o.AnswerS < cur.AnswerS
	}
	if o.Memory.Total != cur.Memory.Total {
		return o.Memory.Total < cur.Memory.Total
	}
	if og, cg := o.Candidate.Device.Type == string(backend.GPU), cur.Candidate.Device.Type == string(backend.GPU); og != cg {
		return og
	}
	return o.Best.DraftMax < cur.Best.DraftMax
}
