package aggregate

import (
	"math"
	"testing"

	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestQuantiles(t *testing.T) {
	xs := []float64{4, 1, 3, 2}
	if !near(Median(xs), 2.5) || !near(Quantile(xs, 0.25), 1.75) || !near(Quantile(xs, 0.75), 3.25) {
		t.Fatal(Median(xs), Quantile(xs, 0.25), Quantile(xs, 0.75))
	}
	if Median(nil) != 0 || Median([]float64{7}) != 7 {
		t.Fatal("degenerate")
	}
	if xs[0] != 4 {
		t.Fatal("Quantile sorted its input in place")
	}
	if !near(MAD([]float64{1, 2, 3, 4, 100}), 1) {
		t.Fatal(MAD([]float64{1, 2, 3, 4, 100}))
	}
}

func TestOutlier(t *testing.T) {
	units := []float64{50, 52, 49, 51, 50, 53}
	if Outlier(51, units) {
		t.Error("an ordinary value is an outlier")
	}
	if !Outlier(150, units) {
		t.Error("triple the cohort is not an outlier")
	}
	if Outlier(150, units[:3]) {
		t.Error("too few units to judge")
	}
	if Outlier(150, []float64{50, 50, 50, 50, 50}) {
		t.Error("no spread, no verdict")
	}
}

func TestUnitsDedupe(t *testing.T) {
	vs := []store.UnitValue{
		// One install submits three times: one unit, its median.
		{InstallID: "a", Fingerprint: "f1", Value: 10}, {InstallID: "a", Fingerprint: "f1", Value: 12}, {InstallID: "a", Fingerprint: "f1", Value: 30},
		// Fresh keys on one machine: one unit.
		{InstallID: "b", Fingerprint: "f2", Value: 100}, {InstallID: "c", Fingerprint: "f2", Value: 100}, {InstallID: "d", Fingerprint: "f2", Value: 100},
		// A real second machine.
		{InstallID: "e", Fingerprint: "f3", Value: 20},
	}
	u := Units(vs)
	if len(u) != 3 {
		t.Fatalf("want 3 units, got %v", u)
	}
	if !near(u[0], 12) || !near(u[1], 100) || !near(u[2], 20) {
		t.Fatal(u)
	}
	// An install whose fingerprint changed stays one unit.
	if n := len(Units([]store.UnitValue{{InstallID: "a", Fingerprint: "f1", Value: 1}, {InstallID: "a", Fingerprint: "f9", Value: 2}})); n != 1 {
		t.Fatal(n)
	}
}
