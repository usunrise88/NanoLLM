package tune

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
)

// WriteText prints a profile as the end of a run shows it: what was chosen and
// why, the candidates with their numbers, the problems (docs/OPTIMIZE.md,
// §4.1). It needs nothing but the profile, so a client that read one over HTTP
// prints it the same way.
func (p *Profile) WriteText(w io.Writer) {
	fmt.Fprintf(w, "NanoLLM optimize: %s (%s), %.0f s\n\n", p.Mode, describe(p.Constraints), p.DurationS)
	sel, _ := p.Chosen.Device.Value.(string)
	name := sel
	if p.Survey != nil {
		for _, d := range p.Survey.Devices {
			if d.Selector == sel {
				name = sel + " " + d.Name
			}
		}
	}
	mode, _ := p.Chosen.PlanMode.Value.(string)
	fmt.Fprintf(w, "  chosen   %s, %s, %d tokens of context, %s\n", name, mode, p.Chosen.NCtx, draftText(p.Chosen.MTP))
	ex := ""
	if p.Answer.Extrapolated {
		ex = " (carried over from a shorter prompt)"
	}
	fmt.Fprintf(w, "           typical answer %.1f s%s: a %d-token prompt at %.0f tok/s, %d tokens at %.1f tok/s\n",
		p.Answer.AnswerS, ex, RefPromptTokens, p.Answer.PP, RefAnswerTokens, p.Answer.TG)
	lim := ""
	if p.Memory.LimitBytes > 0 {
		lim = fmt.Sprintf(" (limit %s + %s)", gib(p.Memory.LimitBytes), gib(p.Memory.ToleranceBytes))
	}
	fmt.Fprintf(w, "  memory   %s planned, %s measured%s\n", gib(p.Memory.PlanBytes), gib(p.Memory.MeasuredBytes), lim)
	if v := p.Verified; v.Gain > 0 && (v.Gain >= 1.02 || v.Gain <= 0.98) {
		fmt.Fprintf(w, "  against  %.2f times the speed of the setup that runs without a profile (%.1f against %.1f tok/s)\n", v.Gain, v.ChosenTokS, v.StaticTokS)
	}
	fmt.Fprintln(w)

	tw := tabwriter.NewWriter(w, 2, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  candidate\tmemory\tprefill\tdecode\tanswer\tverdict")
	recs := append([]Record(nil), p.Candidates...)
	SortRecords(recs)
	for _, r := range recs {
		mem, pp, tg, ans := gib(r.PlanBytes), "-", "-", "-"
		switch {
		case r.Measured:
			mem, pp, tg, ans = gib(r.MeasuredBytes), fmt.Sprintf("%.0f", r.PP), fmt.Sprintf("%.1f", r.TG), fmt.Sprintf("%.1f s", r.AnswerS)
		case r.EstimateS > 0:
			ans = fmt.Sprintf("~%.0f s", r.EstimateS)
		}
		verdict := r.Verdict
		if r.Reason != "" {
			verdict += ": " + r.Reason
		}
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\t%s\t%s\n", r.ID, mem, pp, tg, ans, verdict)
	}
	_ = tw.Flush()

	fmt.Fprintln(w)
	describeLever(w, "device", p.Chosen.Device)
	describeLever(w, "drafting", p.Chosen.MTP)
	describeLever(w, "threads", p.Chosen.Threads)
	if p.Chosen.NUMA != nil {
		describeLever(w, "memory policy", *p.Chosen.NUMA)
	}
	if len(p.Skipped) > 0 {
		fmt.Fprintf(w, "  not measured: %s\n", strings.Join(p.Skipped, ", "))
	}
	if len(p.Problems) > 0 {
		fmt.Fprintln(w)
	}
	for _, pr := range p.Problems {
		fmt.Fprintf(w, "  %s %-22s %s\n", mark(pr.Severity), pr.Code, pr)
	}
}
