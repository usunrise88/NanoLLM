package main

import (
	"strings"
	"testing"
)

func TestParse(t *testing.T) {
	out := `Mean PPL(Q)/PPL(base)         :   1.001234 ±   0.000456
Mean    KLD:   0.012345 ±   0.000123
Maximum KLD:   2.500000
99.9%   KLD:   0.900000
99.0%   KLD:   0.150000
Median  KLD:   0.004000
Same top p: 97.123 ± 0.321 %
`
	n, err := Parse(strings.NewReader(out))
	if err != nil {
		t.Fatal(err)
	}
	if n["kld_mean"] != 0.012345 || n["kld_p99"] != 0.15 || n["top1_pct"] != 97.123 || n["ppl_ratio"] != 1.001234 || n["kld_median"] != 0.004 {
		t.Fatalf("%+v", n)
	}
	if _, err := Parse(strings.NewReader("nothing")); err == nil {
		t.Fatal("expected an error")
	}
}
