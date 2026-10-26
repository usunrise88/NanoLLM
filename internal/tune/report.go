package tune

import (
	"context"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/plan"
)

// Options say what to look at and which mode to look for.
type Options struct {
	Mode   Mode
	Survey SurveyOptions
	Plan   PlanInput
}

// Report is the result of looking the machine over for a mode, with or without
// the measurement that follows.
type Report struct {
	Mode        Mode         `json:"mode"`
	Constraints Constraints  `json:"constraints"`
	Survey      *Survey      `json:"survey"`
	Candidates  []*Candidate `json:"candidates"`
	Problems    []Problem    `json:"problems,omitempty"`
}

// FreeBudget is the largest memory budget any device offers right now: what
// plan.DerivedPreset grants it. The powerful mode compares it with
// PowerfulMinFree.
func FreeBudget(s *Survey) int64 {
	var best int64
	for _, d := range s.Devices {
		if d.Type == string(backend.NPU) {
			continue
		}
		if p, err := plan.DerivedPreset(backendDevice(d)); err == nil {
			best = max(best, p.Budget.Total())
		}
	}
	return best
}

// Look is the dry run (docs/OPTIMIZE.md, §3): the survey, the candidates the
// mode allows and what is wrong with the machine. It loads no model.
func Look(ctx context.Context, o Options) (*Report, error) {
	if !o.Mode.Valid() {
		return nil, fmt.Errorf("unknown mode %q", o.Mode)
	}
	s := TakeSurvey(ctx, o.Survey)
	return Analyze(s, o), nil
}

// Analyze builds the report from a survey already taken.
func Analyze(s *Survey, o Options) *Report {
	k, deg := ConstraintsFor(o.Mode, Cores{Physical: s.CPU.Physical, Logical: s.CPU.Logical}, FreeBudget(s))
	r := &Report{Mode: o.Mode, Constraints: k, Survey: s}
	r.Problems = s.Problems(k)
	if deg != nil {
		r.Problems = append([]Problem{*deg}, r.Problems...)
	}
	cands, cp := BuildCandidates(s, k, o.Plan)
	r.Candidates = cands
	r.Problems = append(r.Problems, cp...)
	sortProblems(r.Problems)
	return r
}

// Contenders are the candidates that were not ruled out.
func (r *Report) Contenders() []*Candidate {
	var out []*Candidate
	for _, c := range r.Candidates {
		if c.Reject == nil {
			out = append(out, c)
		}
	}
	return out
}

// WriteText prints the report as the dry run shows it.
func (r *Report) WriteText(w io.Writer) {
	s := r.Survey
	fmt.Fprintf(w, "NanoLLM optimize: %s (%s)\n\n", r.Mode, describe(r.Constraints))
	fmt.Fprintf(w, "  machine   %s, %d cores (%d threads), %s RAM (%s free)\n", cpuLine(s.CPU), s.CPU.Physical, s.CPU.Logical, gib(s.RAM.Total), gib(s.RAM.Available))
	for _, d := range s.Devices {
		if d.Type != string(backend.GPU) {
			continue
		}
		extra := ""
		if d.Driver != "" {
			extra = ", driver " + d.Driver
		}
		kind := "own memory"
		if d.Unified {
			kind = "shared memory"
		}
		fmt.Fprintf(w, "            %s %s (%s, %s free%s)\n", d.Selector, d.Name, kind, gib(d.MemoryFree), extra)
	}
	fmt.Fprintf(w, "  model     %s on %s\n", s.Model.ID, diskLine(s.Disk))
	fmt.Fprintf(w, "  survey    %.1f s\n\n", s.Elapsed)

	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  candidate\tmode\tcontext\tmemory\tanswer\tverdict")
	for _, c := range r.Candidates {
		mode, ctx, mem, ans := "-", "-", "-", "-"
		if c.Plan != nil {
			mode, ctx, mem = c.Mode, fmt.Sprint(c.NCtx), gib(c.PlannedBytes)
		}
		if c.Estimate.Known {
			ans = fmt.Sprintf("~%.1f s", c.Estimate.AnswerS)
		}
		verdict := "measure"
		switch {
		case c.Reject != nil:
			verdict = "out: " + c.Reject.Reason
		case !c.Measure:
			verdict = "-"
		}
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\t%s\t%s\n", c.ID, mode, ctx, mem, ans, verdict)
	}
	_ = tw.Flush()
	if len(r.Problems) > 0 {
		fmt.Fprintln(w)
	}
	for _, p := range r.Problems {
		fmt.Fprintf(w, "  %s %-22s %s\n", mark(p.Severity), p.Code, p)
	}
}

func mark(s Severity) string {
	switch s {
	case SevError:
		return "x"
	case SevWarn:
		return "!"
	}
	return "i"
}

func describe(k Constraints) string {
	var parts []string
	switch {
	case k.CPUOnly:
		parts = append(parts, "CPU only")
	case k.MemCap > 0:
		parts = append(parts, "up to "+gib(k.MemCap)+" of memory")
	default:
		parts = append(parts, "all free memory")
	}
	if k.Threads > 0 {
		parts = append(parts, fmt.Sprintf("%d threads", k.Threads))
	} else {
		parts = append(parts, "all cores")
	}
	if k.LowPriority {
		parts = append(parts, "low priority")
	}
	return strings.Join(parts, ", ")
}

func cpuLine(c CPUInfo) string {
	if c.Hybrid() {
		return fmt.Sprintf("%s %dP+%dE", c.Model, c.PCores, c.ECores)
	}
	return c.Model
}

func diskLine(d DiskInfo) string {
	if d.Path == "" {
		return "no file"
	}
	s := d.Kind.Type
	if d.Speed != nil {
		s += fmt.Sprintf(" %.0f MB/s", d.Speed.SeqMiBs*1.048576)
	}
	return s
}

// WriteResult prints a finished run: what was chosen and why, the candidates
// with their numbers, the problems (docs/OPTIMIZE.md, §4.1).
func (res *Result) WriteResult(w io.Writer) { res.Profile.WriteText(w) }

func describeLever(w io.Writer, name string, v Value) {
	if v.Why == "" {
		return
	}
	conf := ""
	if !v.Confirmed {
		conf = ", not confirmed"
	}
	fmt.Fprintf(w, "  %-9s [%s%s] %s\n", name, v.Source, conf, v.Why)
}

func draftText(v Value) string {
	d, ok := decodeDraft(v.Value)
	switch {
	case !ok || !d.Enabled:
		return "no drafting"
	case d.PMin > 0:
		return fmt.Sprintf("MTP %d (p_min %g)", d.DraftMax, d.PMin)
	}
	return fmt.Sprintf("MTP %d", d.DraftMax)
}

// ModeLines describes each mode in a line with this machine's numbers, for the
// menu: what the mode allows and the best estimate of a typical answer within
// it (docs/OPTIMIZE.md, §4.1).
func ModeLines(s *Survey, o Options) map[Mode]string {
	out := map[Mode]string{}
	for _, m := range Modes {
		o.Mode = m
		rep := Analyze(s, o)
		line := describe(rep.Constraints)
		var best *Candidate
		for _, c := range rep.Contenders() {
			if !c.Estimate.Known {
				continue
			}
			if rep.Constraints.Goal == GoalMinMemory {
				if best == nil || c.PlannedBytes < best.PlannedBytes {
					best = c
				}
				continue
			}
			if best == nil || c.Estimate.AnswerS < best.Estimate.AnswerS {
				best = c
			}
		}
		if best != nil {
			line += fmt.Sprintf("; a typical answer in about %.0f s on %s", best.Estimate.AnswerS, best.Device.Selector)
		}
		out[m] = line
	}
	return out
}
