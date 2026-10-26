package server

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/tune"
)

// gatedRunner holds every probe until the test lets it go, so that a run can be
// observed while it is in progress.
type gatedRunner struct {
	inner tune.Runner
	gate  chan struct{}
	once  sync.Once
	seen  chan struct{}
}

func (g *gatedRunner) Probe(ctx context.Context, spec tune.ProbeSpec) (tune.ProbeResult, error) {
	g.once.Do(func() { close(g.seen) })
	select {
	case <-g.gate:
	case <-ctx.Done():
	}
	return g.inner.Probe(ctx, spec)
}

type optimizeRig struct {
	rt     *Runtime
	api    *optimizeAPI
	mux    *http.ServeMux
	runner *gatedRunner
	cfg    string
}

func newOptimizeRig(t *testing.T) *optimizeRig {
	t.Helper()
	cat, err := registry.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "nanollm.yaml")
	if err := os.WriteFile(cfgPath, []byte("server:\n  data_dir: "+quote(dir)+"\nbackends:\n  lib_dir: ./lib\n  enabled: [fake]\nauth:\n  mode: open\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := config.Default()
	cfg.Server.DataDir = dir
	cfg.Models.Default, cfg.Models.Preload = "gemma-4-e4b-qat", nil
	cfg.Backends.Enabled = []string{config.BackendFake}
	cfg.Devices.Default = config.DeviceCPU
	cfg.Pool = config.Pool{MaxResidentModels: 1, MaxPending: 8}
	fb := fake.New(fake.Options{
		PrefillRate: 40000, DecodeRate: 500, MaxDraft: 3, Acceptance: 0.6, NoGPU: true,
		BatchCost: func(tokens, outputs int) time.Duration {
			return time.Duration(10_000+400*(tokens-1)) * time.Microsecond
		},
	})
	rt := New(context.Background(), Options{Config: cfg, Catalog: cat, Backends: []backend.Backend{fb}, Log: slog.New(slog.NewTextHandler(io.Discard, nil))})
	t.Cleanup(rt.Close)
	gr := &gatedRunner{inner: tune.InProcessRunner{Backend: fb}, gate: make(chan struct{}), seen: make(chan struct{})}
	api := &optimizeAPI{rt: rt, runs: newRuns[*optRun](), info: OptimizeInfo{Version: "test", LibsRelease: "b1-n1", ConfigPath: cfgPath, Runner: gr, Test: true}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/optimize", api.status)
	mux.HandleFunc("POST /api/v1/optimize", api.start)
	mux.HandleFunc("GET /api/v1/optimize/{id}/events", api.events)
	mux.HandleFunc("GET /api/v1/optimize/{id}", api.get)
	mux.HandleFunc("POST /api/v1/optimize/{id}/apply", api.apply)
	return &optimizeRig{rt: rt, api: api, mux: mux, runner: gr, cfg: cfgPath}
}

func quote(s string) string { b, _ := json.Marshal(s); return string(b) }

func (r *optimizeRig) do(method, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	rec := httptest.NewRecorder()
	r.mux.ServeHTTP(rec, req)
	return rec
}

func (r *optimizeRig) waitState(t *testing.T, id, want string) map[string]any {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		rec := r.do("GET", "/api/v1/optimize/"+id, "")
		var out map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("%v: %s", err, rec.Body.String())
		}
		if out["state"] == want {
			return out
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("run %s did not reach %s", id, want)
	return nil
}

func TestOptimizeRejectsWhatIsNotAMode(t *testing.T) {
	r := newOptimizeRig(t)
	for _, body := range []string{`{"mode":"turbo"}`, `{"mode":""}`, `not json`} {
		rec := r.do("POST", "/api/v1/optimize", body)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: %d %s", body, rec.Code, rec.Body.String())
		}
	}
	if rec := r.do("GET", "/api/v1/optimize/nope", ""); rec.Code != http.StatusNotFound {
		t.Errorf("an unknown run: %d", rec.Code)
	}
}

func TestOptimizeStatusOfAServerWithoutTheAutotuner(t *testing.T) {
	r := newOptimizeRig(t)
	rec := r.do("GET", "/api/v1/optimize", "")
	var st optimizeStatus
	if err := json.Unmarshal(rec.Body.Bytes(), &st); err != nil || rec.Code != 200 {
		t.Fatalf("%d %v %s", rec.Code, err, rec.Body.String())
	}
	if st.Mode != "" || st.Stale || len(st.Modes) != 3 || st.Profile != nil {
		t.Fatalf("%+v", st)
	}
}

func TestOptimizeStatusSaysWhenTheProfileIsStale(t *testing.T) {
	r := newOptimizeRig(t)
	k, _ := tune.ConstraintsFor(tune.Optimal, tune.Cores{Physical: 8, Logical: 16}, 20<<30)
	r.rt.SetTune(&tune.Active{Mode: tune.Optimal, Constraints: k, Stale: true, Changes: []tune.Change{{Field: "driver", Old: "581.57", New: "590.12"}}})
	var st optimizeStatus
	if err := json.Unmarshal(r.do("GET", "/api/v1/optimize", "").Body.Bytes(), &st); err != nil {
		t.Fatal(err)
	}
	if !st.Stale || st.Suggest == nil || st.Suggest.Mode != tune.Optimal || len(st.Changes) != 1 || !strings.Contains(st.Message, "581.57") {
		t.Fatalf("%+v", st)
	}
}

func TestOptimizeDryRunMeasuresNothing(t *testing.T) {
	r := newOptimizeRig(t)
	rec := r.do("POST", "/api/v1/optimize", `{"mode":"optimal","dry_run":true}`)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("%d %s", rec.Code, rec.Body.String())
	}
	var started struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &started)
	out := r.waitState(t, started.ID, "done")
	if out["report"] == nil || out["profile"] != nil || out["dry_run"] != true {
		t.Fatalf("%v", out)
	}
	select {
	case <-r.runner.seen:
		t.Fatal("a dry run loaded a model")
	default:
	}
	if rec := r.do("POST", "/api/v1/optimize/"+started.ID+"/apply", ""); rec.Code != http.StatusBadRequest {
		t.Errorf("applying a dry run: %d", rec.Code)
	}
}

// The whole path on the scripted backend: a run, the model held back while it
// measures, the second run refused, the events, apply.
func TestOptimizeRunHoldApplyReload(t *testing.T) {
	r := newOptimizeRig(t)
	// A model that is resident before the run.
	if _, err := r.rt.Load(context.Background(), "gemma-4-e4b-qat"); err != nil {
		t.Fatal(err)
	}
	if len(r.rt.Pool.Status()) != 1 {
		t.Fatal("the model did not load")
	}

	rec := r.do("POST", "/api/v1/optimize", `{"mode":"optimal"}`)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("%d %s", rec.Code, rec.Body.String())
	}
	var started struct {
		ID        string `json:"id"`
		EstimateS int    `json:"estimate_s"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &started)
	if started.ID == "" || started.EstimateS <= 0 {
		t.Fatalf("%s", rec.Body.String())
	}
	<-r.runner.seen // a probe is in progress

	// One measurement at a time.
	if rec := r.do("POST", "/api/v1/optimize", `{"mode":"silent"}`); rec.Code != http.StatusConflict {
		t.Errorf("a second run: %d %s", rec.Code, rec.Body.String())
	}
	// The model is held: a request gets 503 and is told when to come back, and
	// the resident copy is gone.
	if len(r.rt.Pool.Status()) != 0 {
		t.Errorf("the resident model stayed in memory during the measurement: %v", r.rt.Pool.Status())
	}
	_, err := r.rt.Resolve(context.Background(), "gemma-4-e4b-qat", "")
	ce := core.AsError(err)
	if ce.Code != core.CodeModelUnavailable || ce.RetryAfter <= 0 {
		t.Fatalf("a request during the run: %v (retry after %v)", err, ce.RetryAfter)
	}
	prec := httptest.NewRecorder()
	httpx.WriteProblem(prec, err)
	if prec.Code != http.StatusServiceUnavailable || prec.Header().Get("Retry-After") == "" {
		t.Errorf("the problem answer: %d, Retry-After %q", prec.Code, prec.Header().Get("Retry-After"))
	}
	var status optimizeStatus
	_ = json.Unmarshal(r.do("GET", "/api/v1/optimize", "").Body.Bytes(), &status)
	if status.Running == "" {
		t.Error("the status does not say a measurement runs")
	}

	close(r.runner.gate)
	out := r.waitState(t, started.ID, "done")
	if out["profile"] == nil {
		t.Fatalf("%v", out)
	}
	if _, err := r.rt.Resolve(context.Background(), "gemma-4-e4b-qat", ""); err != nil {
		t.Fatalf("the model is still held after the run: %v", err)
	}

	// The events replay in order to a client that comes late.
	sse := r.do("GET", "/api/v1/optimize/"+started.ID+"/events", "").Body.String()
	prev := -1
	for _, name := range []string{"event: survey", "event: candidate", "event: probe_start", "event: probe", "event: decision", "event: done"} {
		i := strings.Index(sse, name)
		if i < 0 || i < prev {
			t.Fatalf("%q is missing or out of order in\n%s", name, sse)
		}
		prev = i
	}

	// Apply: the profile is saved, the mode written, and the runtime and the model
	// take it on.
	rec = r.do("POST", "/api/v1/optimize/"+started.ID+"/apply", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("apply: %d %s", rec.Code, rec.Body.String())
	}
	var ar struct {
		Profile       string `json:"profile"`
		ConfigWritten bool   `json:"config_written"`
		Reloaded      bool   `json:"reloaded"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &ar)
	if !ar.ConfigWritten || !ar.Reloaded || !strings.HasPrefix(ar.Profile, "optimal@") {
		t.Fatalf("%s", rec.Body.String())
	}
	b, _ := os.ReadFile(r.cfg)
	if !strings.Contains(string(b), "mode: optimal") {
		t.Errorf("the configuration does not name the mode:\n%s", b)
	}
	act := r.rt.Tune()
	if act == nil || act.Mode != tune.Optimal || act.Profile == nil || act.ID() != ar.Profile {
		t.Fatalf("the runtime did not take the profile on: %+v", act)
	}
	if rec := r.do("POST", "/api/v1/optimize/"+started.ID+"/apply", ""); rec.Code != http.StatusConflict {
		t.Errorf("applying twice: %d", rec.Code)
	}
}
