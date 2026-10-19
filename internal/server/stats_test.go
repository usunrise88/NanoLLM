package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/pool"
	"github.com/usunrise88/nanollm/internal/sched"
	"github.com/usunrise88/nanollm/internal/stats"
)

// serve drives a handler without a listener.
func serve(h http.Handler, method, path, token string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, nil)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// traffic leaves the runtime with two answers (the second reusing the first's
// prompt), a failed request and a cancelled one.
func traffic(t *testing.T, rt *Runtime) {
	t.Helper()
	chat(t, rt, "alpha", "a prompt long enough to be cached [[fake:echo]]")
	chat(t, rt, "alpha", "a prompt long enough to be cached [[fake:echo]]")
	if _, err := rt.Engine.Chat(context.Background(), core.ChatRequest{Messages: []core.Message{{Role: core.RoleUser, Content: "[[fake:error]]"}}}, nil); err == nil {
		t.Fatal("a scripted failure succeeded")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := rt.Engine.Chat(ctx, core.ChatRequest{Messages: []core.Message{{Role: core.RoleUser, Content: "never runs"}}}, nil); err == nil {
		t.Fatal("a cancelled request succeeded")
	}
	eventually(t, "the model to go idle", func() bool { return resident(rt)["alpha"].Busy == 0 })
}

func TestStatsReport(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	traffic(t, rt)
	rep := rt.Stats()
	if rep.Process.Goroutines <= 0 || rep.Process.GoHeap == 0 || rep.Process.UptimeS < 0 {
		t.Errorf("process %+v", rep.Process)
	}
	m := rep.Metrics
	if m.Requests["chat/ok"] != 2 || m.Requests["chat/error"] != 1 || m.Requests["chat/cancelled"] != 1 {
		t.Errorf("requests %v", m.Requests)
	}
	if m.Tokens.Prompt == 0 || m.Tokens.Completion == 0 || m.Tokens.Cached == 0 || m.CacheHitRate <= 0 || m.CacheHitRate > 1 {
		t.Errorf("tokens %+v, cache hit rate %v", m.Tokens, m.CacheHitRate)
	}
	if m.LatencyMS["total"].Count != 4 {
		t.Errorf("total latency count %d, want every finished request", m.LatencyMS["total"].Count)
	}
	if len(rep.Instances) != 1 {
		t.Fatalf("instances %+v", rep.Instances)
	}
	in := rep.Instances[0]
	if in.Key.Model != "alpha" || in.Loop.Slots != 1 || in.Loop.Busy != 0 || in.Loop.Pending != 0 || in.Loop.Steps == 0 || in.Loop.StepTokens < in.Loop.Steps || in.Refs != 0 {
		t.Errorf("instance %+v", in)
	}
}

func TestStatsEndpointsAreAdminOnly(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	traffic(t, rt)
	mux := http.NewServeMux()
	rt.MountStats(mux)
	keys, err := httpx.NewKeyStore([]httpx.KeySpec{{Name: "root", Secret: "root-secret", Admin: true}, {Name: "user", Secret: "user-secret"}})
	if err != nil {
		t.Fatal(err)
	}
	h := httpx.Auth(keys)(mux)
	for _, path := range []string{"/api/v1/stats", "/metrics"} {
		for token, want := range map[string]int{"": http.StatusUnauthorized, "user-secret": http.StatusForbidden, "root-secret": http.StatusOK} {
			if got := serve(h, http.MethodGet, path, token).Code; got != want {
				t.Errorf("GET %s with key %q: %d, want %d", path, token, got, want)
			}
		}
		if got := serve(h, http.MethodPost, path, "root-secret").Code; got != http.StatusMethodNotAllowed {
			t.Errorf("POST %s: %d, want 405", path, got)
		}
		// Open mode has no keys at all, and the loopback rule guards it.
		if got := serve(mux, http.MethodGet, path, "").Code; got != http.StatusOK {
			t.Errorf("GET %s in open mode: %d", path, got)
		}
	}
}

func TestStatsJSON(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	traffic(t, rt)
	mux := http.NewServeMux()
	rt.MountStats(mux)
	rec := serve(mux, http.MethodGet, "/api/v1/stats", "")
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("content type %q", ct)
	}
	// The field names are the contract the soak test and dashboards read.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"process", "metrics", "instances"} {
		if _, ok := raw[k]; !ok {
			t.Errorf("no %q in %s", k, rec.Body.String())
		}
	}
	var rep struct {
		Process   map[string]any `json:"process"`
		Metrics   stats.Snapshot `json:"metrics"`
		Instances []struct {
			Key  pool.Key        `json:"key"`
			Refs int             `json:"refs"`
			Loop sched.LoopStats `json:"loop"`
		} `json:"instances"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"uptime_s", "goroutines", "rss", "rss_peak", "locked", "go_heap", "gc_cycles"} {
		if _, ok := rep.Process[k]; !ok {
			t.Errorf("no process.%s", k)
		}
	}
	if rep.Metrics.Requests["chat/ok"] != 2 || rep.Metrics.LatencyMS["ttft"].Count == 0 || len(rep.Metrics.DecodeTokS.Bounds) == 0 {
		t.Errorf("metrics %+v", rep.Metrics)
	}
	if len(rep.Instances) != 1 || rep.Instances[0].Key.Model != "alpha" || rep.Instances[0].Loop.Slots != 1 {
		t.Errorf("instances %+v", rep.Instances)
	}
}

var (
	sampleLine = regexp.MustCompile(`^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(?:[a-zA-Z_][a-zA-Z0-9_]*="(?:[^"\\]|\\.)*",?)*\})? (\S+)$`)
	leLabel    = regexp.MustCompile(`le="([^"]+)"`)
)

func TestPrometheusText(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	traffic(t, rt)
	mux := http.NewServeMux()
	rt.MountStats(mux)
	rec := serve(mux, http.MethodGet, "/metrics", "")
	if ct := rec.Header().Get("Content-Type"); ct != "text/plain; version=0.0.4; charset=utf-8" {
		t.Fatalf("content type %q", ct)
	}
	body := rec.Body.String()
	if !strings.HasSuffix(body, "\n") {
		t.Error("the exposition does not end with a newline")
	}

	types := map[string]string{} // family → type
	helped := map[string]bool{}
	samples := map[string]float64{} // name{labels} → value
	buckets := map[string][]float64{}
	for _, line := range strings.Split(strings.TrimSuffix(body, "\n"), "\n") {
		switch {
		case strings.HasPrefix(line, "# HELP "):
			helped[strings.Fields(line)[2]] = true
		case strings.HasPrefix(line, "# TYPE "):
			f := strings.Fields(line)
			if _, dup := types[f[2]]; dup {
				t.Errorf("family %s declared twice", f[2])
			}
			types[f[2]] = f[3]
		default:
			m := sampleLine.FindStringSubmatch(line)
			if m == nil {
				t.Errorf("not a sample line: %q", line)
				continue
			}
			v, err := strconv.ParseFloat(m[3], 64)
			if err != nil {
				t.Errorf("value of %q: %v", line, err)
			}
			samples[m[1]+m[2]] = v
			family := strings.TrimSuffix(strings.TrimSuffix(strings.TrimSuffix(m[1], "_bucket"), "_sum"), "_count")
			if types[m[1]] == "" && types[family] == "" {
				t.Errorf("sample %q before its # TYPE", line)
			}
			if strings.HasSuffix(m[1], "_bucket") {
				if le := leLabel.FindStringSubmatch(m[2]); le == nil {
					t.Errorf("bucket without le: %q", line)
				} else {
					buckets[family] = append(buckets[family], v)
				}
			}
		}
	}
	for fam := range types {
		if !helped[fam] {
			t.Errorf("family %s has no # HELP", fam)
		}
	}

	for sample, want := range map[string]float64{
		`nanollm_requests_total{kind="chat",outcome="ok"}`:        2,
		`nanollm_requests_total{kind="chat",outcome="error"}`:     1,
		`nanollm_requests_total{kind="chat",outcome="cancelled"}`: 1,
		`nanollm_slots{device="cpu",model="alpha"}`:               1,
		`nanollm_slots_busy{device="cpu",model="alpha"}`:          0,
		`nanollm_pending_requests{device="cpu",model="alpha"}`:    0,
		`nanollm_model_references{device="cpu",model="alpha"}`:    0,
	} {
		if got, ok := samples[sample]; !ok || got != want {
			t.Errorf("%s = %v (present %v), want %v", sample, got, ok, want)
		}
	}
	for _, name := range []string{
		"nanollm_uptime_seconds", "nanollm_goroutines", "nanollm_rss_bytes", "nanollm_rss_peak_bytes",
		"nanollm_locked_bytes", "nanollm_go_heap_bytes", "nanollm_prompt_tokens_total", "nanollm_cached_tokens_total",
		"nanollm_completion_tokens_total", "nanollm_drafted_tokens_total",
	} {
		if _, ok := samples[name]; !ok {
			t.Errorf("no %s in the exposition", name)
		}
	}
	if samples["nanollm_prompt_tokens_total"] == 0 || samples["nanollm_cached_tokens_total"] == 0 {
		t.Errorf("token counters: prompt %v, cached %v", samples["nanollm_prompt_tokens_total"], samples["nanollm_cached_tokens_total"])
	}
	if samples[`nanollm_steps_total{device="cpu",model="alpha"}`] == 0 || samples[`nanollm_step_tokens_total{device="cpu",model="alpha"}`] < samples[`nanollm_steps_total{device="cpu",model="alpha"}`] {
		t.Errorf("step counters: %v steps, %v tokens", samples[`nanollm_steps_total{device="cpu",model="alpha"}`], samples[`nanollm_step_tokens_total{device="cpu",model="alpha"}`])
	}

	// Histograms: cumulative buckets, +Inf equal to _count.
	for _, fam := range []string{"nanollm_ttft_milliseconds", "nanollm_queue_milliseconds", "nanollm_total_milliseconds", "nanollm_decode_tokens_per_second"} {
		if types[fam] != "histogram" {
			t.Errorf("%s is %q, want a histogram", fam, types[fam])
			continue
		}
		b := buckets[fam]
		if len(b) < 2 {
			t.Errorf("%s has %d buckets", fam, len(b))
			continue
		}
		for i := 1; i < len(b); i++ {
			if b[i] < b[i-1] {
				t.Errorf("%s buckets are not cumulative: %v", fam, b)
				break
			}
		}
		if inf, count := samples[fam+`_bucket{le="+Inf"}`], samples[fam+"_count"]; inf != count {
			t.Errorf("%s: +Inf bucket %v, count %v", fam, inf, count)
		}
	}
	if samples["nanollm_total_milliseconds_count"] != 4 {
		t.Errorf("total latency observations %v, want 4", samples["nanollm_total_milliseconds_count"])
	}
}

// Instances of one gauge family share its # HELP and # TYPE, and a label set
// is rendered the same whatever order the map yields it in.
func TestGaugesPerInstance(t *testing.T) {
	rep := StatsReport{
		Process: stats.Process{UptimeS: 12.5, Goroutines: 7, RSS: 1 << 20},
		Instances: []pool.Status{
			{Key: pool.Key{Model: "alpha", Device: "cpu"}, Refs: 2, Loop: sched.LoopStats{Slots: 4, Busy: 3, Pending: 1, Steps: 10, StepTokens: 99}},
			{Key: pool.Key{Model: "beta", Device: "gpu0"}, Loop: sched.LoopStats{Slots: 1}},
		},
	}
	var sb strings.Builder
	stats.WritePrometheus(&sb, stats.New().Snapshot(), gauges(rep))
	out := sb.String()
	for _, want := range []string{
		"nanollm_uptime_seconds 12.5\n",
		"nanollm_goroutines 7\n",
		"nanollm_rss_bytes 1.048576e+06\n",
		"nanollm_slots{device=\"cpu\",model=\"alpha\"} 4\n",
		"nanollm_slots{device=\"gpu0\",model=\"beta\"} 1\n",
		"nanollm_slots_busy{device=\"cpu\",model=\"alpha\"} 3\n",
		"nanollm_pending_requests{device=\"cpu\",model=\"alpha\"} 1\n",
		"nanollm_step_tokens_total{device=\"cpu\",model=\"alpha\"} 99\n",
		"nanollm_model_references{device=\"cpu\",model=\"alpha\"} 2\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in\n%s", want, out)
		}
	}
	if n := strings.Count(out, "# TYPE nanollm_slots gauge"); n != 1 {
		t.Errorf("# TYPE nanollm_slots appears %d times", n)
	}
	// A server with nothing loaded renders no per-model samples at all.
	sb.Reset()
	stats.WritePrometheus(&sb, stats.New().Snapshot(), gauges(StatsReport{}))
	if strings.Contains(sb.String(), "nanollm_slots") {
		t.Errorf("per-model gauges without a model:\n%s", sb.String())
	}
}
