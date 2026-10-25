package tune

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/memlock"
	"github.com/usunrise88/nanollm/internal/numa"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/quietexec"
	"github.com/usunrise88/nanollm/internal/registry"
)

// Event is one step of a run, for the terminal and for the SSE stream of the
// API. Type is survey, problem, candidate, probe_start, probe, decision, done
// or error.
type Event struct {
	Type    string    `json:"type"`
	Time    time.Time `json:"time"`
	Message string    `json:"message,omitempty"`
	// Candidate names the candidate a probe event is about.
	Candidate string `json:"candidate,omitempty"`
	// Data carries the payload of the event: the survey, a problem, an outcome.
	Data any `json:"data,omitempty"`
	// ElapsedS and RemainS are the time spent and the time left of the budget.
	ElapsedS float64 `json:"elapsed_s,omitempty"`
	RemainS  float64 `json:"remain_s,omitempty"`
}

// Budget is how long a run may take in all (docs/OPTIMIZE.md, §3.3): about two
// minutes on a typical laptop.
const Budget = 120 * time.Second

// MaxLoads is the most models a run loads outside the silent mode, which goes
// up the ladder of budgets until one works.
const MaxLoads = 2

// RunOptions are what a run needs besides the options of the look.
type RunOptions struct {
	Options
	Runner   Runner
	Progress func(Event)
	// Budget is the time for the whole run; zero is Budget.
	Budget time.Duration
	// Survey, when set, is used instead of taking one (the API keeps one for a
	// couple of minutes).
	Survey *Survey
	// Cooldown waits for a GPU to cool before it is measured; nil is
	// WaitForCool.
	Cooldown func(ctx context.Context, d Device, idleC int)
	// Now is the clock, for tests.
	Now func() time.Time
}

// Result is a finished run.
type Result struct {
	Report   *Report
	Decision *Decision
	Profile  *Profile
	Outcomes []*Outcome
}

// Run looks the machine over, measures what the mode allows, chooses and
// builds the profile. The profile is not saved or applied; see Apply.
func Run(ctx context.Context, o RunOptions) (*Result, error) {
	if !o.Mode.Valid() {
		return nil, fmt.Errorf("unknown mode %q", o.Mode)
	}
	if o.Runner == nil {
		return nil, errors.New("tune: no runner")
	}
	if o.Budget <= 0 {
		o.Budget = Budget
	}
	if o.Now == nil {
		o.Now = time.Now
	}
	started := o.Now()
	emit := func(e Event) {
		if o.Progress == nil {
			return
		}
		e.Time = o.Now()
		e.ElapsedS = e.Time.Sub(started).Seconds()
		e.RemainS = max(0, (o.Budget - e.Time.Sub(started)).Seconds())
		o.Progress(e)
	}
	s := o.Survey
	if s == nil {
		s = TakeSurvey(ctx, o.Options.Survey)
	}
	emit(Event{Type: "survey", Message: fmt.Sprintf("surveyed in %.1f s", s.Elapsed), Data: s})
	rep := Analyze(s, o.Options)
	for _, p := range rep.Problems {
		emit(Event{Type: "problem", Message: p.String(), Data: p})
	}
	for _, c := range rep.Candidates {
		emit(Event{Type: "candidate", Candidate: c.ID, Message: candidateLine(c), Data: c})
	}

	res := &Result{Report: rep}
	loads := 0
	var skipped []string
	toMeasure := measurable(rep.Candidates)
	for i, c := range toMeasure {
		left := o.Budget - o.Now().Sub(started)
		if left < 15*time.Second || ctx.Err() != nil {
			for _, r := range toMeasure[i:] {
				skipped = append(skipped, "candidate "+r.ID)
			}
			break
		}
		if rep.Constraints.Mode != Silent && loads >= MaxLoads {
			for _, r := range toMeasure[i:] {
				skipped = append(skipped, "candidate "+r.ID+" (the limit of loads)")
			}
			break
		}
		if c.Device.Type == string(backend.GPU) {
			emit(Event{Type: "probe_start", Candidate: c.ID, Message: "waiting for the GPU to cool"})
			cool := o.Cooldown
			if cool == nil {
				cool = WaitForCool
			}
			cool(ctx, c.Device, s.Load.GPUTempC)
		}
		spec := BuildProbeSpec(s, rep.Constraints, o.Plan, c, o.Options)
		share := time.Duration(int64(left) / int64(max(len(toMeasure)-i, 1)))
		spec.Budget = max(min(left-10*time.Second, 2*share), 20*time.Second)
		if rep.Constraints.Mode == Silent {
			spec.Budget = max(left-10*time.Second, 20*time.Second)
		}
		emit(Event{Type: "probe_start", Candidate: c.ID, Message: "loading " + c.ID})
		loads++
		pr, err := o.Runner.Probe(ctx, spec)
		out := Evaluate(c, &pr, err, s, rep.Constraints)
		res.Outcomes = append(res.Outcomes, out)
		emit(Event{Type: "probe", Candidate: c.ID, Message: outcomeLine(out), Data: out})
		if len(pr.Skipped) > 0 {
			skipped = append(skipped, pr.Skipped...)
		}
		if err != nil && ctx.Err() != nil {
			return res, ctx.Err()
		}
		// The quiet mode stops at the first setup that works: the ladder goes
		// from the least memory up.
		if rep.Constraints.Mode == Silent && out.Err == "" && !out.Spill && out.Best.TokS >= rep.Constraints.MinTokS {
			for _, r := range toMeasure[i+1:] {
				skipped = append(skipped, "candidate "+r.ID+" (a smaller one works)")
			}
			break
		}
	}

	res.Decision = Decide(res.Outcomes, s, rep.Constraints)
	rep.Problems = append(rep.Problems, res.Decision.Problems...)
	sortProblems(rep.Problems)
	emit(Event{Type: "decision", Message: decisionLine(res.Decision)})
	if res.Decision.Chosen == nil {
		return res, errors.New("no setup could be measured on this machine")
	}
	res.Profile = BuildProfile(s, rep, res, o.Plan.matrix(), skipped, o.Now().Sub(started), o.Now())
	emit(Event{Type: "done", Message: "done", Data: res.Profile})
	return res, nil
}

// measurable are the candidates a run measures, in order.
func measurable(cands []*Candidate) []*Candidate {
	var out []*Candidate
	for _, c := range cands {
		if c.Reject == nil && c.Measure {
			out = append(out, c)
		}
	}
	return out
}

func candidateLine(c *Candidate) string {
	if c.Reject != nil {
		return fmt.Sprintf("%s: out (%s)", c.ID, c.Reject.Reason)
	}
	est := ""
	if c.Estimate.Known {
		est = fmt.Sprintf(", about %.1f s for a typical answer", c.Estimate.AnswerS)
	}
	return fmt.Sprintf("%s: %s, context %d, %s%s", c.ID, c.Mode, c.NCtx, gib(c.PlannedBytes), est)
}

func outcomeLine(o *Outcome) string {
	if o.Err != "" {
		return fmt.Sprintf("%s: failed (%s)", o.Candidate.ID, o.Err)
	}
	return fmt.Sprintf("%s: prefill %.0f tok/s, decode %.1f tok/s, %.1f s for a typical answer, %s held", o.Candidate.ID, o.PP, o.Best.TokS, o.AnswerS, gib(o.Memory.Total))
}

func decisionLine(d *Decision) string {
	if d.Chosen == nil {
		return "nothing works"
	}
	return "chosen: " + d.Chosen.Candidate.ID
}

// BuildProbeSpec turns a candidate into what the worker measures.
func BuildProbeSpec(s *Survey, k Constraints, in PlanInput, c *Candidate, o Options) ProbeSpec {
	load := c.Spec
	cpu := c.Device.Type == string(backend.CPU)
	vs, static, mech := variantsFor(s, c, in)
	maxDraft := 0
	for _, v := range vs {
		maxDraft = max(maxDraft, v.DraftMax)
	}
	if mech {
		maxDraft = max(maxDraft, 3)
	}
	if maxDraft > 0 && in.DrafterPath != "" {
		// The model loads with the longest draft the probe tries; shorter ones
		// are set on the loaded model.
		load.DrafterPath, load.DraftMax = in.DrafterPath, maxDraft
		load.NOutputsMax = max(load.Slots, 1) * (1 + maxDraft)
	} else {
		vs, static, mech = []DraftVariant{{Name: "off"}}, "off", false
	}
	spec := ProbeSpec{
		LowPriority: k.LowPriority,
		Backend:     c.Device.Backend, LibDir: o.Survey.LibDir, Load: load, Variants: vs, StaticVariant: static,
		Mechanism: mech, Kind: c.Device.Kind, BatchSizes: []int{1, 3, 4},
		Reuse: c.Plan != nil && c.Plan.KV.SWAFull, PromptTokens: min(RefPromptTokens, c.NCtx-DecodeTokens-64),
		DepthPP: 0.85, DepthTG: 0.66,
	}
	if b, _, ok := in.matrix().BaselineFor(deviceKindOf(c.Device), c.Class); ok {
		spec.DepthPP, spec.DepthTG = b.DepthPP, b.DepthTG
	}
	spec.Verify = mech
	for _, v := range vs {
		if v.DraftMax > 0 {
			spec.Verify = true
		}
	}
	if c.Estimate.DiskBound || ColdBytesPerToken(c.Plan) > 0 {
		spec.Slow, spec.Verify = true, false
	}
	if c.Plan != nil {
		for _, lr := range c.Plan.LockRanges {
			r := memlock.Range{Off: lr.Offset, Len: lr.Length}
			if lr.Role == registry.RoleDrafter {
				spec.DrafterLocks = append(spec.DrafterLocks, r)
			} else {
				spec.Locks = append(spec.Locks, r)
			}
		}
		for _, rr := range c.Plan.RandomRanges {
			spec.Random = append(spec.Random, memlock.Range{Off: rr.Offset, Len: rr.Length})
		}
	}
	if c.NUMA != "" {
		place := c.NUMA
		if c.NUMA == "distribute" && load.Threads > 0 {
			// The repack threads must be the decode threads (internal/numa).
			place += ";threads=" + strconv.Itoa(load.Threads)
		}
		spec.Env = map[string]string{numa.EnvPlace: place}
	}
	if cpu && s.CPU.Hybrid() && k.Mode == Powerful && c.Spec.Threads > 0 {
		// The performance cores alone against all the cores, with all the CPUs
		// for prompt chunks either way.
		spec.Threads = []ThreadVariant{
			{Name: "all-cores", Threads: s.CPU.Physical, ThreadsBatch: s.CPU.Logical},
		}
	}
	return spec
}

func (in PlanInput) matrix() *Matrix {
	if in.Matrix != nil {
		return in.Matrix
	}
	return Default
}

// Variant names.
const (
	VariantOff = "off"
)

// DraftName is the name of a drafting variant.
func DraftName(max int, pMin float64) string {
	if max <= 0 {
		return VariantOff
	}
	return "mtp" + strconv.Itoa(max) + "/" + strconv.FormatFloat(pMin, 'g', -1, 64)
}

// variantsFor lists the drafting settings to measure for a candidate
// (docs/OPTIMIZE.md, §3.3) and the name of the one the server runs without a
// profile. mech says the probe adds the draft length the batch cost favours.
func variantsFor(s *Survey, c *Candidate, in PlanInput) (vs []DraftVariant, static string, mech bool) {
	off := DraftVariant{Name: VariantOff}
	mk := func(max int, pMin float64) DraftVariant {
		return DraftVariant{Name: DraftName(max, pMin), DraftMax: max, PMin: pMin}
	}
	add := func(v DraftVariant) {
		for _, x := range vs {
			if x.Name == v.Name {
				return
			}
		}
		vs = append(vs, v)
	}
	vs = []DraftVariant{off}
	static = VariantOff
	if c.Plan != nil && c.Plan.MTP.Enabled {
		static = DraftName(c.Plan.MTP.DraftMax, c.Plan.MTP.DraftPMin)
	}
	switch {
	case c.Device.Type == string(backend.CPU) && c.Device.Has(backend.FeatureMultiRowQ4_0):
		add(mk(plan.CPUMTPDraftMax, plan.CPUMTPDraftPMin))
		add(mk(3, 0))
		// A neighbouring class's measured setting is a candidate too.
		if m, ok := in.matrix().Lookup("mtp_cpu", c.Class); ok {
			add(variantOf(m.Cell))
		}
	case c.Device.Type == string(backend.CPU):
		// No fast kernels for the check of a batch: only the batch-cost
		// mechanism says whether drafting pays here.
		mech = true
	case c.Device.Kind == "cuda":
		add(mk(3, 0))
		add(mk(2, 0))
	default:
		add(mk(2, 0))
		add(mk(3, 0))
	}
	if c.Device.Type != string(backend.CPU) {
		if m, ok := in.matrix().Lookup("mtp_gpu", c.Class); ok && m.Source == SourceClass {
			add(variantOf(m.Cell))
		}
	}
	if static != VariantOff {
		if v, ok := parseDraftName(static); ok {
			add(v)
		}
	}
	return vs, static, mech
}

func variantOf(c Cell) DraftVariant {
	max, _ := c.Variant["draft_max"].(int)
	pMin := 0.0
	switch x := c.Variant["p_min"].(type) {
	case float64:
		pMin = x
	case int:
		pMin = float64(x)
	}
	return DraftVariant{Name: DraftName(max, pMin), DraftMax: max, PMin: pMin}
}

func parseDraftName(n string) (DraftVariant, bool) {
	rest, ok := strings.CutPrefix(n, "mtp")
	if !ok {
		return DraftVariant{}, false
	}
	a, b, ok := strings.Cut(rest, "/")
	if !ok {
		return DraftVariant{}, false
	}
	max, err1 := strconv.Atoi(a)
	pMin, err2 := strconv.ParseFloat(b, 64)
	if err1 != nil || err2 != nil {
		return DraftVariant{}, false
	}
	return DraftVariant{Name: n, DraftMax: max, PMin: pMin}, true
}

// CoolTimeout is the longest the autotuner waits for a GPU to cool before a
// measurement, and CoolMargin how far above the idle temperature it accepts.
const (
	CoolTimeout = 30 * time.Second
	CoolMargin  = 3
)

// WaitForCool polls the GPU's temperature once a second until it is within
// CoolMargin of idleC, the temperature the survey found, or CoolTimeout passes.
// A laptop that never gets back to its idle temperature is not waited for
// longer than the timeout, and the run goes on with the GPU as it is.
func WaitForCool(ctx context.Context, d Device, idleC int) {
	if d.Vendor != "NVIDIA" || idleC <= 0 {
		return
	}
	deadline := time.Now().Add(CoolTimeout)
	for time.Now().Before(deadline) && ctx.Err() == nil {
		out, err := quietexec.Command(ctx, "nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits").Output()
		if err != nil {
			return
		}
		f := strings.Split(strings.TrimSpace(strings.Split(string(out), "\n")[0]), ",")
		if len(f) < 2 {
			return
		}
		temp, _ := strconv.Atoi(strings.TrimSpace(f[0]))
		util, _ := strconv.Atoi(strings.TrimSpace(f[1]))
		if temp <= idleC+CoolMargin && util < 15 {
			return
		}
		select {
		case <-time.After(time.Second):
		case <-ctx.Done():
		}
	}
}
