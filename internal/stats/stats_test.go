package stats

import (
	"bytes"
	"math"
	"strings"
	"sync"
	"testing"
)

func TestHistogramQuantiles(t *testing.T) {
	h := NewHistogram(10, 100, 1000)
	for range 50 {
		h.Observe(5) // bucket (0,10]
	}
	for range 40 {
		h.Observe(50) // (10,100]
	}
	for range 9 {
		h.Observe(500) // (100,1000]
	}
	h.Observe(5000) // +Inf
	s := h.Snapshot()
	if s.Count != 100 || s.Counts[0] != 50 || s.Counts[1] != 90 || s.Counts[2] != 99 || s.Counts[3] != 100 {
		t.Fatalf("%+v", s)
	}
	if math.Abs(s.Sum-(250+2000+4500+5000)) > 1e-9 {
		t.Errorf("sum %v", s.Sum)
	}
	if s.P50 <= 0 || s.P50 > 10 {
		t.Errorf("p50 %v should fall in the first bucket", s.P50)
	}
	if s.P95 <= 100 || s.P95 > 1000 {
		t.Errorf("p95 %v should fall in (100,1000]", s.P95)
	}
	if empty := NewHistogram(1).Snapshot(); empty.P99 != 0 || empty.Count != 0 {
		t.Errorf("empty: %+v", empty)
	}
}

func TestMetricsConcurrent(t *testing.T) {
	m := New()
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 1000 {
				m.Observe(Request{Kind: "chat", Outcome: OutcomeOK, Prompt: 10, Cached: 4, Completion: 3, TTFTMS: 12, TotalMS: 100, DecodeMS: 50})
			}
		}()
	}
	wg.Wait()
	m.Count("chat", OutcomeCancelled)
	s := m.Snapshot()
	if s.Requests["chat/ok"] != 8000 || s.Requests["chat/cancelled"] != 1 {
		t.Errorf("requests %v", s.Requests)
	}
	if s.Tokens.Prompt != 80000 || s.Tokens.Cached != 32000 || s.CacheHitRate != 0.4 {
		t.Errorf("tokens %+v hit %v", s.Tokens, s.CacheHitRate)
	}
	if s.LatencyMS["ttft"].Count != 8000 || s.DecodeTokS.Count != 8000 {
		t.Errorf("histograms %+v %+v", s.LatencyMS["ttft"].Count, s.DecodeTokS.Count)
	}
}

func TestPrometheus(t *testing.T) {
	m := New()
	m.Observe(Request{Kind: "chat", Outcome: OutcomeOK, Prompt: 5, Completion: 2, TTFTMS: 30})
	var buf bytes.Buffer
	WritePrometheus(&buf, m.Snapshot(), []Gauge{
		{Name: "nanollm_slots_busy", Help: "Busy slots.", Labels: map[string]string{"model": "tiny"}, Value: 1},
		{Name: "nanollm_slots_busy", Help: "Busy slots.", Labels: map[string]string{"model": "other"}, Value: 0},
	})
	out := buf.String()
	for _, want := range []string{
		`nanollm_requests_total{kind="chat",outcome="ok"} 1`,
		"nanollm_prompt_tokens_total 5",
		`nanollm_ttft_milliseconds_bucket{le="50"} 1`,
		`nanollm_ttft_milliseconds_bucket{le="+Inf"} 1`,
		"nanollm_ttft_milliseconds_count 1",
		`nanollm_slots_busy{model="tiny"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	if strings.Count(out, "# TYPE nanollm_slots_busy gauge") != 1 {
		t.Errorf("a gauge family is declared once:\n%s", out)
	}
}
