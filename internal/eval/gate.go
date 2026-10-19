package eval

import (
	"fmt"
	"io"
	"sort"
	"text/tabwriter"
)

// Gate verdicts.
const (
	GateSame       = "same"
	GateBetter     = "better"
	GateRegression = "regression"
	GateMissing    = "missing"
)

// GateRow is one capability of two reports.
type GateRow struct {
	Key     string // capability, with "+thinking" for the thinking runs
	A, B    Tally
	Verdict string
}

// DefaultTolerance is the drop in a capability's pass rate the gate lets
// through: the suites are small (a dozen items a capability), so one item is
// noise and two are not.
const DefaultTolerance = 0.10

// Gate compares two reports of one suite capability by capability, without
// averaging them into one number: a configuration that is faster and loses
// JSON validity must not hide behind a good Russian score. A capability is a
// regression when B passes fewer items than A by more than tol of the items,
// and by at least one.
//
// B is the candidate and A the reference. ok is false when any capability
// regressed or is missing from B.
func Gate(a, b *Report, tol float64) (rows []GateRow, ok bool, err error) {
	if a.Suite.SHA256 != b.Suite.SHA256 {
		return nil, false, fmt.Errorf("eval gate: the reports ran different suites (%s and %s)", a.Suite.Name, b.Suite.Name)
	}
	ok = true
	keys := make([]string, 0, len(a.Summary))
	for k := range a.Summary {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		ta, tb := a.Summary[k], b.Summary[k]
		row := GateRow{Key: k, A: ta, B: tb, Verdict: GateSame}
		switch {
		case tb.N == 0 && ta.N > 0:
			row.Verdict, ok = GateMissing, false
		case ta.N > 0 && float64(ta.Pass)/float64(ta.N)-float64(tb.Pass)/float64(tb.N) > tol && tb.Pass < ta.Pass:
			row.Verdict, ok = GateRegression, false
		case tb.Pass > ta.Pass:
			row.Verdict = GateBetter
		}
		rows = append(rows, row)
	}
	return rows, ok, nil
}

// WriteGate prints the rows as a table.
func WriteGate(w io.Writer, rows []GateRow) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "capability\treference\tcandidate\tverdict")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%d/%d\t%d/%d\t%s\n", r.Key, r.A.Pass, r.A.N, r.B.Pass, r.B.N, r.Verdict)
	}
	_ = tw.Flush()
}
