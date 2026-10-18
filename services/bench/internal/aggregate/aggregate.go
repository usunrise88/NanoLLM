// Package aggregate computes what the results pages show: per group, the
// median and quartiles of each metric over units, where a unit is one
// installation and one hardware fingerprint counted once (docs/BENCHMARKS.md).
// It also holds the robust statistics the outlier check uses.
package aggregate

import (
	"context"
	"math"
	"sort"
	"strings"

	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

// Quantile is the linearly interpolated q-quantile of xs (R type 7).
func Quantile(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	h := q * float64(len(s)-1)
	lo := math.Floor(h)
	i := int(lo)
	if i+1 >= len(s) {
		return s[len(s)-1]
	}
	return s[i] + (h-lo)*(s[i+1]-s[i])
}

// Median of xs.
func Median(xs []float64) float64 { return Quantile(xs, 0.5) }

// MAD is the median absolute deviation from the median.
func MAD(xs []float64) float64 {
	m := Median(xs)
	d := make([]float64, len(xs))
	for i, x := range xs {
		d[i] = math.Abs(x - m)
	}
	return Median(d)
}

// MinOutlierUnits is how many units a cohort needs before the MAD test
// means anything.
const MinOutlierUnits = 5

// OutlierZ is the modified z-score beyond which a value is an outlier
// (Iglewicz and Hoaglin).
const OutlierZ = 3.5

// Outlier reports whether x is far from the units' values: a modified
// z-score 0.6745·|x−median|/MAD above OutlierZ. With too few units, or no
// spread, nothing is an outlier.
func Outlier(x float64, units []float64) bool {
	if len(units) < MinOutlierUnits {
		return false
	}
	mad := MAD(units)
	if mad == 0 {
		return false
	}
	return 0.6745*math.Abs(x-Median(units))/mad > OutlierZ
}

// Units collapses values to one per unit. A unit is a connected set of
// installs and fingerprints: two installs on one fingerprint (a fresh key
// per run) are one unit, and so is one install whose fingerprint changed.
// Each install's values are reduced to their median first, then each
// unit's install medians to theirs.
func Units(vs []store.UnitValue) []float64 {
	parent := map[string]string{}
	var find func(string) string
	find = func(x string) string {
		if p, ok := parent[x]; ok && p != x {
			r := find(p)
			parent[x] = r
			return r
		}
		parent[x] = x
		return x
	}
	union := func(a, b string) { parent[find(a)] = find(b) }
	perInstall := map[string][]float64{}
	var installs []string
	for _, v := range vs {
		i, f := "i:"+v.InstallID, "f:"+v.Fingerprint
		union(i, f)
		if _, ok := perInstall[i]; !ok {
			installs = append(installs, i)
		}
		perInstall[i] = append(perInstall[i], v.Value)
	}
	perUnit := map[string][]float64{}
	var units []string
	for _, i := range installs {
		u := find(i)
		if _, ok := perUnit[u]; !ok {
			units = append(units, u)
		}
		perUnit[u] = append(perUnit[u], Median(perInstall[i]))
	}
	out := make([]float64, 0, len(units))
	for _, u := range units {
		out = append(out, Median(perUnit[u]))
	}
	return out
}

// Rebuild recomputes the aggregates table from the derived run rows.
func Rebuild(ctx context.Context, q store.Querier) error {
	in, err := store.AggInputs(ctx, q)
	if err != nil {
		return err
	}
	type group struct {
		row    store.AggRow
		values map[string][]store.UnitValue
	}
	groups := map[string]*group{}
	var order []string
	for _, r := range in {
		key := strings.Join([]string{r.ModelID, r.Suite, r.Preset, r.CPUSlug, r.GPUSlug, r.SystemSlug, r.OS, r.FormFactor}, "\x00")
		g, ok := groups[key]
		if !ok {
			g = &group{row: r, values: map[string][]store.UnitValue{}}
			groups[key] = g
			order = append(order, key)
		}
		g.row.Verified = g.row.Verified || r.Verified
		for m, v := range r.Metrics {
			if v > 0 {
				g.values[m] = append(g.values[m], store.UnitValue{InstallID: r.InstallID, Fingerprint: r.Fingerprint, Value: v})
			}
		}
	}
	var out []store.AggOut
	for _, key := range order {
		g := groups[key]
		for _, m := range store.AggMetrics {
			vs := g.values[m]
			if len(vs) == 0 {
				continue
			}
			u := Units(vs)
			r := g.row
			out = append(out, store.AggOut{
				ModelID: r.ModelID, Suite: r.Suite, Preset: r.Preset, CPUSlug: r.CPUSlug, GPUSlug: r.GPUSlug, SystemSlug: r.SystemSlug,
				OS: r.OS, FormFactor: r.FormFactor, CPU: r.CPU, GPU: r.GPU, System: r.System, Verified: r.Verified,
				Metric: m, N: len(u), Median: Median(u), P25: Quantile(u, 0.25), P75: Quantile(u, 0.75),
			})
		}
	}
	return store.ReplaceAggregates(ctx, q, out)
}
