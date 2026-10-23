package bench

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/usunrise88/nanollm/benchfmt"
)

// Verdicts of a comparison.
const (
	VerdictGain  = "gain"
	VerdictLoss  = "loss"
	VerdictNoise = "noise"
	VerdictNone  = "-" // no spread recorded for this metric
)

// MinEffect is the smallest relative change a comparison calls a gain or a
// loss, however tight the spread: below it, a change is not worth a step.
const MinEffect = 0.01

// Row is one metric of one run, before and after.
type Row struct {
	Preset, Variant, Metric string
	A, B                    float64
	SDA, SDB                float64 // sample standard deviations; 0 when not recorded
	Delta                   float64 // (B-A)/A
	Verdict                 string
}

type metricDef struct {
	name   string
	higher bool // true when a larger number is better
	sdKey  string
	get    func(benchfmt.Run) float64
}

var compareMetrics = []metricDef{
	{"pp tok/s", true, "pp_tok_s", func(r benchfmt.Run) float64 { return r.Metrics.PPTokS }},
	{"tg tok/s", true, "tg_tok_s.d0", func(r benchfmt.Run) float64 { return r.Metrics.TGTokS.D0 }},
	{"chat tok/s", true, "chat_tok_s", func(r benchfmt.Run) float64 { return r.Metrics.ChatTokS }},
	{"ttft ms", false, "ttft_ms", func(r benchfmt.Run) float64 { return r.Metrics.TTFTMS }},
	{"mtp tok/s", true, "mtp_tok_s", func(r benchfmt.Run) float64 {
		if r.Metrics.MTP == nil {
			return 0
		}
		return r.Metrics.MTP.TokS
	}},
	batchMetric(2, false), batchMetric(2, true), batchMetric(4, false), batchMetric(4, true),
	batchMetric(8, false), batchMetric(8, true),
	{"load ms", false, "", func(r benchfmt.Run) float64 { return r.Metrics.LoadMS }},
	{"peak rss", false, "", func(r benchfmt.Run) float64 { return float64(r.Metrics.PeakRSS) }},
	{"peak cgroup", false, "", func(r benchfmt.Run) float64 { return float64(r.Metrics.PeakCgroup) }},
}

// batchMetric is the aggregate throughput of n sequences at once (the batch key
// of a dev sweep), without or with MTP.
func batchMetric(n int, mtp bool) metricDef {
	name := fmt.Sprintf("agg tok/s x%d", n)
	if mtp {
		name += " mtp"
	}
	return metricDef{name, true, "", func(r benchfmt.Run) float64 {
		for _, p := range r.Metrics.Batch {
			if p.Seqs == n {
				if mtp {
					return p.MTPAggTokS
				}
				return p.AggTokS
			}
		}
		return 0
	}}
}

// Compare pairs the runs of a and b by preset and variant and judges every
// metric against its recorded spread. With b nil it compares the variants of
// one sweep against the first run of each preset.
func Compare(a, b *benchfmt.Result) []Row {
	type key struct{ preset, variant string }
	var pairs [][2]benchfmt.Run
	if b != nil {
		byKey := map[key]benchfmt.Run{}
		for _, r := range b.Runs {
			if r.Status == benchfmt.StatusOK {
				byKey[key{r.Preset, r.Variant}] = r
			}
		}
		for _, r := range a.Runs {
			if rb, ok := byKey[key{r.Preset, r.Variant}]; ok && r.Status == benchfmt.StatusOK {
				pairs = append(pairs, [2]benchfmt.Run{r, rb})
			}
		}
	} else {
		first := map[string]benchfmt.Run{}
		for _, r := range a.Runs {
			if r.Status != benchfmt.StatusOK {
				continue
			}
			base, ok := first[r.Preset]
			if !ok {
				first[r.Preset] = r
				continue
			}
			pairs = append(pairs, [2]benchfmt.Run{base, r})
		}
	}
	var rows []Row
	for _, p := range pairs {
		ra, rb := p[0], p[1]
		variant := rb.Variant
		if b != nil && ra.Variant == rb.Variant {
			variant = ra.Variant
		}
		for _, m := range compareMetrics {
			va, vb := m.get(ra), m.get(rb)
			if va == 0 && vb == 0 {
				continue
			}
			row := Row{Preset: ra.Preset, Variant: variant, Metric: m.name, A: va, B: vb, Verdict: VerdictNone}
			if va != 0 {
				row.Delta = (vb - va) / va
			}
			if m.sdKey != "" {
				row.SDA, row.SDB = ra.Metrics.StdDev[m.sdKey], rb.Metrics.StdDev[m.sdKey]
				row.Verdict = judge(row, m.higher, max(ra.Reps, 1), max(rb.Reps, 1))
			}
			rows = append(rows, row)
		}
	}
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].Preset < rows[j].Preset })
	return rows
}

// judge compares the change with the standard error of a difference of two
// medians (1.25·sd/√n each): a change is real when it is more than two of
// those and more than MinEffect.
func judge(r Row, higher bool, na, nb int) string {
	se := 1.2533 * math.Sqrt(r.SDA*r.SDA/float64(na)+r.SDB*r.SDB/float64(nb))
	d := r.B - r.A
	if math.Abs(d) <= 2*se || math.Abs(r.Delta) < MinEffect {
		return VerdictNoise
	}
	if (d > 0) == higher {
		return VerdictGain
	}
	return VerdictLoss
}

// WriteComparison prints rows as a table.
func WriteComparison(w io.Writer, rows []Row) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "preset\tvariant\tmetric\tA\tB\tchange\tverdict")
	for _, r := range rows {
		v := r.Variant
		if v == "" {
			v = "-"
		}
		a, b := fmtNum(r.A, r.SDA), fmtNum(r.B, r.SDB)
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%+.1f%%\t%s\n", r.Preset, v, r.Metric, a, b, r.Delta*100, r.Verdict)
	}
	_ = tw.Flush()
}

func fmtNum(v, sd float64) string {
	switch {
	case v >= 1e9:
		return fmt.Sprintf("%.2f GiB", v/(1<<30))
	case v >= 1e6:
		return fmt.Sprintf("%.0f MiB", v/(1<<20))
	case sd > 0:
		return fmt.Sprintf("%.2f ±%.2f", v, sd)
	}
	return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.2f", v), "0"), ".")
}
