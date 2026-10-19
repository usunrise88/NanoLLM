package eval

import (
	"bytes"
	"strings"
	"testing"
)

func report(sha string, tallies map[string][2]int) *Report {
	r := &Report{Suite: SuiteRef{Name: "s", SHA256: sha}, Summary: map[string]Tally{}}
	for k, v := range tallies {
		r.Summary[k] = Tally{N: v[1], Pass: v[0], Rate: float64(v[0]) / float64(v[1])}
	}
	return r
}

func TestGate(t *testing.T) {
	ref := report("x", map[string][2]int{"ru-instructions": {14, 16}, "json": {10, 10}, "long-context": {9, 12}})
	// One item lost out of sixteen is noise; two out of ten is not.
	cand := report("x", map[string][2]int{"ru-instructions": {13, 16}, "json": {8, 10}, "long-context": {10, 12}})
	rows, ok, err := Gate(ref, cand, DefaultTolerance)
	if err != nil || ok {
		t.Fatalf("ok=%v err=%v", ok, err)
	}
	got := map[string]string{}
	for _, r := range rows {
		got[r.Key] = r.Verdict
	}
	if got["ru-instructions"] != GateSame || got["json"] != GateRegression || got["long-context"] != GateBetter {
		t.Errorf("verdicts %v", got)
	}
	var buf bytes.Buffer
	WriteGate(&buf, rows)
	if !strings.Contains(buf.String(), "regression") {
		t.Errorf("table:\n%s", buf.String())
	}

	same := report("x", map[string][2]int{"ru-instructions": {14, 16}, "json": {10, 10}, "long-context": {9, 12}})
	if _, ok, _ := Gate(ref, same, DefaultTolerance); !ok {
		t.Error("identical reports should pass")
	}
	if _, _, err := Gate(ref, report("y", nil), DefaultTolerance); err == nil {
		t.Error("reports of different suites should be refused")
	}
	missing := report("x", map[string][2]int{"json": {10, 10}})
	if _, ok, _ := Gate(ref, missing, DefaultTolerance); ok {
		t.Error("a capability missing from the candidate should fail the gate")
	}
}
