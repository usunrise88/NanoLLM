package tune

import (
	"sort"
)

// The mechanism probe of MTP (docs/OPTIMIZE.md, §3.8). Whether drafting pays
// depends on two things: how many of the drafted tokens the model accepts,
// which is a property of the model and the text and nearly the same on every
// device, and what it costs the device to check several tokens in one batch,
// which differs a great deal. The second is measured in a second or two on the
// loaded model, the first comes from our own measurements, and together they
// say which draft length pays on a processor we have never seen.

// DefaultAcceptance is the chance that the draft at position j+1 is accepted
// when the j before it were, for Gemma 4 E4B's assistant on chat text under
// greedy decoding (the overall acceptance of 3 drafts per step is 0.46 on the
// RTX 3070 Ti, bench/results/rusunrisebook/optimize-stage0-2026-10-23).
var DefaultAcceptance = []float64{0.72, 0.58, 0.48, 0.40}

// DraftCostFraction is the cost of one draft round against one step of the
// target: about 2 % on CUDA, a few percent on the CPU and Vulkan, where the
// small model is not hidden by the GPU's graph.
func DraftCostFraction(deviceKind string) float64 {
	if deviceKind == "cuda" {
		return 0.02
	}
	return 0.05
}

// BatchCost is the cost of checking a batch of n tokens relative to one token,
// c(n) = t(n) / t(1), from the times measured for a few batch sizes: linear
// between them and beyond the largest.
func BatchCost(ms map[int]float64) func(n int) float64 {
	t1 := ms[1]
	if t1 <= 0 {
		return func(int) float64 { return 0 }
	}
	var ks []int
	for k, v := range ms {
		if v > 0 {
			ks = append(ks, k)
		}
	}
	sort.Ints(ks)
	return func(n int) float64 {
		if n <= 1 {
			return 1
		}
		// Find the measured sizes around n.
		lo, hi := 1, 0
		for _, k := range ks {
			if k <= n {
				lo = k
			}
			if k >= n && hi == 0 {
				hi = k
			}
		}
		at := func(k int) float64 {
			if k == 1 {
				return 1
			}
			return ms[k] / t1
		}
		switch {
		case lo == n:
			return at(n)
		case hi != 0:
			return at(lo) + (at(hi)-at(lo))*float64(n-lo)/float64(hi-lo)
		default:
			// Beyond the largest measurement: continue its slope from the
			// previous one.
			if len(ks) >= 2 {
				a, b := ks[len(ks)-2], ks[len(ks)-1]
				slope := (at(b) - at(a)) / float64(b-a)
				return at(b) + slope*float64(n-b)
			}
			return at(lo)
		}
	}
}

// ExpectedSpeedup is the expected speed-up of drafting k tokens per step over
// decoding one: the tokens a step yields on average, 1 + a1 + a1a2 + …, over its
// cost, the check of k+1 tokens plus k draft rounds (docs/OPTIMIZE.md, §3.8):
//
//	E(k) = (1 + a1 + a1·a2 + … + a1…ak) / (c(k+1) + k·d)
func ExpectedSpeedup(k int, cost func(n int) float64, accept []float64, d float64) float64 {
	if k <= 0 {
		return 1
	}
	tokens, run := 1.0, 1.0
	for j := 0; j < k; j++ {
		a := DefaultAcceptance[len(DefaultAcceptance)-1]
		if j < len(accept) {
			a = accept[j]
		}
		run *= a
		tokens += run
	}
	den := cost(k+1) + float64(k)*d
	if den <= 0 {
		return 1
	}
	return tokens / den
}

// BestDraftMax is the draft length with the highest expected speed-up up to
// maxK and that speed-up; 0 and 1 when no length beats decoding one token by
// the margin (a guess that costs more than it gives is not worth the risk of a
// slower machine).
func BestDraftMax(ms map[int]float64, accept []float64, d float64, maxK int) (k int, gain float64) {
	const margin = 1.05
	cost := BatchCost(ms)
	k, gain = 0, 1
	for n := 1; n <= maxK; n++ {
		if e := ExpectedSpeedup(n, cost, accept, d); e > gain && e >= margin {
			k, gain = n, e
		}
	}
	return k, gain
}

// AcceptanceFrom turns a variant's per-position counts (steps that tried the
// draft at j+1, steps that were accepted) into the conditional acceptance by
// position; positions with too few trials take the default.
func AcceptanceFrom(tried, accepted []int) []float64 {
	out := append([]float64(nil), DefaultAcceptance...)
	for j := range tried {
		if j < len(out) && j < len(accepted) && tried[j] >= 8 {
			out[j] = float64(accepted[j]) / float64(tried[j])
		}
	}
	return out
}
