// Package stats counts what the server does, for the operator: requests by
// outcome, tokens, cache hits and latency histograms. It has no dependencies
// and takes no locks on the request path beyond atomics, so a load test can
// run with it on and a soak test can read it once a minute.
//
// GET /api/v1/stats serves a Snapshot as JSON and GET /metrics serves it in
// the Prometheus text format; both are admin-only (internal/server).
package stats

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

// Histogram counts observations into fixed buckets. It is safe for
// concurrent use.
type Histogram struct {
	bounds []float64      // upper bounds, ascending; the last bucket is +Inf
	counts []atomic.Int64 // per bucket, not cumulative
	sum    atomic.Uint64  // float64 bits
	n      atomic.Int64
}

// NewHistogram makes a histogram with the given ascending upper bounds.
func NewHistogram(bounds ...float64) *Histogram {
	return &Histogram{bounds: append([]float64(nil), bounds...), counts: make([]atomic.Int64, len(bounds)+1)}
}

// Observe records one value.
func (h *Histogram) Observe(v float64) {
	if math.IsNaN(v) {
		return
	}
	i := sort.SearchFloat64s(h.bounds, v)
	h.counts[i].Add(1)
	h.n.Add(1)
	for {
		old := h.sum.Load()
		if h.sum.CompareAndSwap(old, math.Float64bits(math.Float64frombits(old)+v)) {
			return
		}
	}
}

// HistogramSnapshot is a histogram at an instant.
type HistogramSnapshot struct {
	Bounds []float64 `json:"bounds"`
	// Counts are cumulative, one per bound and a last one for +Inf, as
	// Prometheus expects.
	Counts []int64 `json:"counts"`
	Sum    float64 `json:"sum"`
	Count  int64   `json:"count"`
	P50    float64 `json:"p50"`
	P95    float64 `json:"p95"`
	P99    float64 `json:"p99"`
}

// Snapshot reads the histogram. The quantiles are estimates, interpolated
// inside the bucket they fall in.
func (h *Histogram) Snapshot() HistogramSnapshot {
	s := HistogramSnapshot{Bounds: append([]float64(nil), h.bounds...), Counts: make([]int64, len(h.counts))}
	var cum int64
	for i := range h.counts {
		cum += h.counts[i].Load()
		s.Counts[i] = cum
	}
	s.Count = cum
	s.Sum = math.Float64frombits(h.sum.Load())
	s.P50, s.P95, s.P99 = s.quantile(0.50), s.quantile(0.95), s.quantile(0.99)
	return s
}

func (s HistogramSnapshot) quantile(q float64) float64 {
	if s.Count == 0 {
		return 0
	}
	rank := q * float64(s.Count)
	var prev int64
	for i, c := range s.Counts {
		if float64(c) < rank {
			prev = c
			continue
		}
		lo := 0.0
		if i > 0 {
			lo = s.Bounds[i-1]
		}
		if i >= len(s.Bounds) {
			return lo // the +Inf bucket: no upper edge to interpolate to
		}
		hi := s.Bounds[i]
		if c == prev {
			return hi
		}
		return lo + (hi-lo)*(rank-float64(prev))/float64(c-prev)
	}
	return s.Bounds[len(s.Bounds)-1]
}

// Latency buckets, in milliseconds, and decode-rate buckets in tokens/s.
var (
	latencyMS = []float64{5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 120000, 300000}
	rateTokS  = []float64{1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 100, 200}
)

// Request outcomes.
const (
	OutcomeOK        = "ok"
	OutcomeError     = "error"
	OutcomeCancelled = "cancelled"
)

// Metrics is the server's counters.
type Metrics struct {
	requests sync.Map // "kind\x00outcome" → *atomic.Int64

	PromptTokens     atomic.Int64
	CachedTokens     atomic.Int64
	CompletionTokens atomic.Int64
	ReasoningTokens  atomic.Int64
	Drafted          atomic.Int64
	Accepted         atomic.Int64

	// TTFT is the time to the first token, Queue the wait for a slot, Total
	// the whole request, all in milliseconds; Decode is tokens/s per request.
	TTFT   *Histogram
	Queue  *Histogram
	Total  *Histogram
	Decode *Histogram
}

// New returns zeroed metrics.
func New() *Metrics {
	return &Metrics{
		TTFT: NewHistogram(latencyMS...), Queue: NewHistogram(latencyMS...),
		Total: NewHistogram(latencyMS...), Decode: NewHistogram(rateTokS...),
	}
}

// Count records a finished request of a kind ("chat", "complete") and its
// outcome.
func (m *Metrics) Count(kind, outcome string) {
	key := kind + "\x00" + outcome
	v, ok := m.requests.Load(key)
	if !ok {
		v, _ = m.requests.LoadOrStore(key, new(atomic.Int64))
	}
	v.(*atomic.Int64).Add(1)
}

// Request is what one finished request reports.
type Request struct {
	Kind    string
	Outcome string
	// Token counts; zero for a request that failed before generating.
	Prompt, Cached, Completion, Reasoning, Drafted, Accepted int
	// Durations in milliseconds; zero means not measured.
	TTFTMS, QueueMS, TotalMS, DecodeMS float64
}

// Observe records a finished request.
func (m *Metrics) Observe(r Request) {
	m.Count(r.Kind, r.Outcome)
	m.PromptTokens.Add(int64(r.Prompt))
	m.CachedTokens.Add(int64(r.Cached))
	m.CompletionTokens.Add(int64(r.Completion))
	m.ReasoningTokens.Add(int64(r.Reasoning))
	m.Drafted.Add(int64(r.Drafted))
	m.Accepted.Add(int64(r.Accepted))
	if r.TTFTMS > 0 {
		m.TTFT.Observe(r.TTFTMS)
	}
	if r.QueueMS > 0 {
		m.Queue.Observe(r.QueueMS)
	}
	if r.TotalMS > 0 {
		m.Total.Observe(r.TotalMS)
	}
	if r.DecodeMS > 0 && r.Completion > 1 {
		m.Decode.Observe(float64(r.Completion-1) / (r.DecodeMS / 1000))
	}
}

// Snapshot is the counters at an instant.
type Snapshot struct {
	// Requests is keyed "kind/outcome".
	Requests map[string]int64 `json:"requests"`
	Tokens   struct {
		Prompt     int64 `json:"prompt"`
		Cached     int64 `json:"cached"`
		Completion int64 `json:"completion"`
		Reasoning  int64 `json:"reasoning"`
		Drafted    int64 `json:"drafted"`
		Accepted   int64 `json:"accepted"`
	} `json:"tokens"`
	// CacheHitRate is cached prompt tokens over prompt tokens.
	CacheHitRate float64                      `json:"cache_hit_rate"`
	LatencyMS    map[string]HistogramSnapshot `json:"latency_ms"`
	DecodeTokS   HistogramSnapshot            `json:"decode_tok_s"`
}

// Snapshot reads the metrics.
func (m *Metrics) Snapshot() Snapshot {
	s := Snapshot{Requests: map[string]int64{}}
	m.requests.Range(func(k, v any) bool {
		kind, outcome, _ := strings.Cut(k.(string), "\x00")
		s.Requests[kind+"/"+outcome] = v.(*atomic.Int64).Load()
		return true
	})
	s.Tokens.Prompt, s.Tokens.Cached = m.PromptTokens.Load(), m.CachedTokens.Load()
	s.Tokens.Completion, s.Tokens.Reasoning = m.CompletionTokens.Load(), m.ReasoningTokens.Load()
	s.Tokens.Drafted, s.Tokens.Accepted = m.Drafted.Load(), m.Accepted.Load()
	if s.Tokens.Prompt > 0 {
		s.CacheHitRate = float64(s.Tokens.Cached) / float64(s.Tokens.Prompt)
	}
	s.LatencyMS = map[string]HistogramSnapshot{"ttft": m.TTFT.Snapshot(), "queue": m.Queue.Snapshot(), "total": m.Total.Snapshot()}
	s.DecodeTokS = m.Decode.Snapshot()
	return s
}

// Gauge is one labelled instantaneous value, for WritePrometheus.
type Gauge struct {
	Name   string
	Help   string
	Labels map[string]string
	Value  float64
}

// WritePrometheus renders s and the gauges in the Prometheus text format.
func WritePrometheus(w io.Writer, s Snapshot, gauges []Gauge) {
	fmt.Fprintln(w, "# HELP nanollm_requests_total Finished requests by kind and outcome.")
	fmt.Fprintln(w, "# TYPE nanollm_requests_total counter")
	keys := make([]string, 0, len(s.Requests))
	for k := range s.Requests {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		kind, outcome, _ := strings.Cut(k, "/")
		fmt.Fprintf(w, "nanollm_requests_total{kind=%q,outcome=%q} %d\n", kind, outcome, s.Requests[k])
	}
	for _, c := range []struct {
		name, help string
		v          int64
	}{
		{"nanollm_prompt_tokens_total", "Prompt tokens received.", s.Tokens.Prompt},
		{"nanollm_cached_tokens_total", "Prompt tokens served from a slot's cache.", s.Tokens.Cached},
		{"nanollm_completion_tokens_total", "Tokens generated.", s.Tokens.Completion},
		{"nanollm_reasoning_tokens_total", "Generated tokens inside the reasoning channel.", s.Tokens.Reasoning},
		{"nanollm_drafted_tokens_total", "MTP tokens drafted.", s.Tokens.Drafted},
		{"nanollm_accepted_tokens_total", "MTP drafted tokens accepted.", s.Tokens.Accepted},
	} {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s counter\n%s %d\n", c.name, c.help, c.name, c.name, c.v)
	}
	names := make([]string, 0, len(s.LatencyMS))
	for n := range s.LatencyMS {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		writeHistogram(w, "nanollm_"+n+"_milliseconds", "Latency in milliseconds.", s.LatencyMS[n])
	}
	writeHistogram(w, "nanollm_decode_tokens_per_second", "Decode rate per request.", s.DecodeTokS)

	sort.SliceStable(gauges, func(i, j int) bool { return gauges[i].Name < gauges[j].Name })
	last := ""
	for _, g := range gauges {
		if g.Name != last {
			fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n", g.Name, g.Help, g.Name)
			last = g.Name
		}
		fmt.Fprintf(w, "%s%s %g\n", g.Name, labelString(g.Labels), g.Value)
	}
}

func writeHistogram(w io.Writer, name, help string, h HistogramSnapshot) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s histogram\n", name, help, name)
	for i, c := range h.Counts {
		le := "+Inf"
		if i < len(h.Bounds) {
			le = fmt.Sprintf("%g", h.Bounds[i])
		}
		fmt.Fprintf(w, "%s_bucket{le=%q} %d\n", name, le, c)
	}
	fmt.Fprintf(w, "%s_sum %g\n%s_count %d\n", name, h.Sum, name, h.Count)
}

func labelString(l map[string]string) string {
	if len(l) == 0 {
		return ""
	}
	keys := make([]string, 0, len(l))
	for k := range l {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = fmt.Sprintf("%s=%q", k, l[k])
	}
	return "{" + strings.Join(parts, ",") + "}"
}
