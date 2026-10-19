package server

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/bench"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/httpx"
)

// The benchmark routes need the llama.cpp backend to start a run, so what runs
// here is everything around that: the refusals before a run starts, and the
// event stream and the submission of runs the test puts in place itself.

func post(h http.Handler, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func problem(t *testing.T, rec *httptest.ResponseRecorder) httpx.Problem {
	t.Helper()
	var p httpx.Problem
	if err := json.Unmarshal(rec.Body.Bytes(), &p); err != nil {
		t.Fatalf("not a problem: %q", rec.Body.String())
	}
	return p
}

func TestBenchStartRefusals(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	mux := http.NewServeMux()
	rt.MountBench(mux, BenchInfo{})
	for _, tc := range []struct {
		name, body string
		status     int
		code, para string
	}{
		{"not JSON", `{oops`, http.StatusBadRequest, "invalid_request", ""},
		{"an unknown suite", `{"suite":"bogus"}`, http.StatusBadRequest, "invalid_request", "suite"},
		{"an unknown model", `{"model":"gamma"}`, http.StatusNotFound, "model_not_found", "model"},
		// The fake backend is all this runtime has.
		{"no llama.cpp backend", `{}`, http.StatusConflict, "capability_unavailable", ""},
		{"no llama.cpp backend for the dev suite", `{"suite":"dev","quick":false}`, http.StatusConflict, "capability_unavailable", ""},
	} {
		rec := post(mux, "/api/v1/bench", tc.body)
		p := problem(t, rec)
		if rec.Code != tc.status || p.Code != tc.code || p.Param != tc.para {
			t.Errorf("%s: %d %+v, want %d %s %q", tc.name, rec.Code, p, tc.status, tc.code, tc.para)
		}
	}
	// Nothing was started: no run exists to follow.
	if rec := serve(mux, http.MethodGet, "/api/v1/bench/0123/events", ""); rec.Code != http.StatusNotFound || problem(t, rec).Code != "job_not_found" {
		t.Errorf("events of a run that never started: %d %s", rec.Code, rec.Body.String())
	}
}

// benchRoutes mounts the routes that act on a run on an API the test holds,
// so it can put runs in it.
func benchRoutes(rt *Runtime) (*http.ServeMux, *benchAPI) {
	api := &benchAPI{rt: rt, runs: map[string]*benchRun{}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/bench/{id}/events", api.events)
	mux.HandleFunc("POST /api/v1/bench/{id}/submit", api.submit)
	return mux, api
}

func newRun(api *benchAPI, id string) *benchRun {
	run := &benchRun{id: id, notify: make(chan struct{})}
	api.mu.Lock()
	api.runs[id] = run
	api.mu.Unlock()
	return run
}

var eventName = regexp.MustCompile(`(?m)^event: (\w+)$`)

func TestBenchEventsReplayAFinishedRun(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	mux, api := benchRoutes(rt)
	run := newRun(api, "done1")
	run.push("progress", map[string]any{"preset": "ram-8g", "status": "pending"})
	run.push("progress", map[string]any{"preset": "ram-8g", "status": "running", "percent": 50})
	run.push("result", map[string]any{"ok": true})
	run.push("done", map[string]any{})
	rec := serve(mux, http.MethodGet, "/api/v1/bench/done1/events", "")
	if rec.Code != http.StatusOK || !strings.HasPrefix(rec.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("%d %q", rec.Code, rec.Header().Get("Content-Type"))
	}
	var names []string
	for _, m := range eventName.FindAllStringSubmatch(rec.Body.String(), -1) {
		names = append(names, m[1])
	}
	if got := strings.Join(names, ","); got != "progress,progress,result,done" {
		t.Fatalf("events %s in\n%s", got, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"percent":50`) {
		t.Errorf("payloads lost:\n%s", rec.Body.String())
	}
	// A run that ended in an error ends the stream too.
	failed := newRun(api, "failed1")
	failed.push("error", map[string]any{"code": "internal"})
	if rec := serve(mux, http.MethodGet, "/api/v1/bench/failed1/events", ""); !strings.Contains(rec.Body.String(), "event: error") {
		t.Errorf("a failed run's stream:\n%s", rec.Body.String())
	}
}

func TestBenchEventsFollowARunningOne(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	mux, api := benchRoutes(rt)
	run := newRun(api, "live1")
	run.push("progress", map[string]any{"status": "pending"})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/api/v1/bench/live1/events", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	lines := make(chan string)
	go func() {
		defer close(lines)
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			if m := eventName.FindStringSubmatch(sc.Text()); m != nil {
				lines <- m[1]
			}
		}
	}()
	next := func() string {
		select {
		case n, ok := <-lines:
			if !ok {
				return "<end>"
			}
			return n
		case <-ctx.Done():
			t.Fatal("timed out waiting for an event")
			return ""
		}
	}
	if got := next(); got != "progress" {
		t.Fatalf("first event %q", got)
	}
	// Events pushed while the client waits arrive; "done" ends the stream.
	run.push("progress", map[string]any{"status": "running"})
	if got := next(); got != "progress" {
		t.Fatalf("second event %q", got)
	}
	run.push("result", map[string]any{})
	run.push("done", map[string]any{})
	if a, b, c := next(), next(), next(); a != "result" || b != "done" || c != "<end>" {
		t.Fatalf("the rest of the stream: %q %q %q", a, b, c)
	}
}

func TestBenchEventsEndWhenTheClientLeaves(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	mux, api := benchRoutes(rt)
	newRun(api, "idle1").push("progress", map[string]any{})
	srv := httptest.NewServer(mux)
	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+"/api/v1/bench/idle1/events", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	resp.Body.Close()
	closed := make(chan struct{})
	go func() { srv.Close(); close(closed) }() // waits for the handler to return
	select {
	case <-closed:
	case <-time.After(10 * time.Second):
		t.Fatal("the event handler did not return when the client left")
	}
}

func TestBenchSubmitGuards(t *testing.T) {
	cases := []struct {
		name   string
		cfg    func(*config.Config)
		run    func(*benchRun)
		body   string
		status int
		code   string
		para   string
	}{
		{"not JSON", nil, nil, `{oops`, http.StatusBadRequest, "invalid_request", ""},
		{"a stale consent text", nil, nil, `{"consent_version":"2000-01"}`, http.StatusBadRequest, "invalid_request", "consent_version"},
		{"submissions switched off", func(c *config.Config) { c.Bench.Submit = "never" }, nil,
			`{"consent_version":"` + bench.ConsentVersion + `"}`, http.StatusForbidden, "forbidden", ""},
		{"a run still going", nil, nil, `{"consent_version":"` + bench.ConsentVersion + `","anonymous":true}`, http.StatusBadRequest, "invalid_request", ""},
		// A result measured on the fake backend never goes to the public
		// service, whatever the consent.
		{"a fake result for the public service", nil, func(r *benchRun) {
			r.result = &benchfmt.Result{Suite: benchfmt.Suite{ID: bench.SuiteStandard}, Client: benchfmt.Client{GGMLCPUVariant: benchfmt.FakeCPUVariant}}
		}, `{"consent_version":"` + bench.ConsentVersion + `","anonymous":true}`, http.StatusServiceUnavailable, "model_unavailable", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var mutate []func(*config.Config)
			if tc.cfg != nil {
				mutate = append(mutate, tc.cfg)
			}
			rt := newRuntime(t, 1, fake.Options{}, mutate...)
			mux, api := benchRoutes(rt)
			run := newRun(api, "r1")
			if tc.run != nil {
				tc.run(run)
			}
			rec := post(mux, "/api/v1/bench/r1/submit", tc.body)
			p := problem(t, rec)
			if rec.Code != tc.status || p.Code != tc.code || p.Param != tc.para {
				t.Fatalf("%d %+v, want %d %s %q", rec.Code, p, tc.status, tc.code, tc.para)
			}
		})
	}
	rt := newRuntime(t, 1, fake.Options{})
	mux, _ := benchRoutes(rt)
	if rec := post(mux, "/api/v1/bench/missing/submit", `{}`); rec.Code != http.StatusNotFound || problem(t, rec).Code != "job_not_found" {
		t.Errorf("an unknown run: %d %s", rec.Code, rec.Body.String())
	}
}
