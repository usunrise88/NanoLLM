package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/bench"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/registry"
)

// The benchmark over HTTP, for the UI's Bench screen (docs/BENCHMARKS.md):
//
//	POST /api/v1/bench {model?, suite, quick}      → {id, presets, estimate_s?}
//	GET  /api/v1/bench/{id}/events                 SSE: progress, result, done, error
//	POST /api/v1/bench/{id}/submit {consent_version, anonymous} → {id, url}
//
// One run at a time: workers load their own copies of the model, so the
// model's resident instances are unloaded first, and two runs would measure
// each other.

// BenchInfo is what the runtime needs to know to run the benchmark.
type BenchInfo struct {
	LibDir  string
	Version string
	Tag     string
	Commit  string
}

type benchRun struct {
	id     string
	mu     sync.Mutex
	events []benchEvent
	notify chan struct{}
	done   bool
	result *benchfmt.Result
}

type benchEvent struct {
	name    string
	payload any
}

func (b *benchRun) push(name string, payload any) {
	b.mu.Lock()
	b.events = append(b.events, benchEvent{name, payload})
	if name == "done" || name == "error" {
		b.done = true
	}
	old := b.notify
	b.notify = make(chan struct{})
	b.mu.Unlock()
	close(old)
}

type benchAPI struct {
	rt   *Runtime
	info BenchInfo

	mu      sync.Mutex
	runs    map[string]*benchRun
	running bool
}

// MountBench adds the benchmark routes, behind the admin check.
func (r *Runtime) MountBench(mux *http.ServeMux, info BenchInfo) {
	api := &benchAPI{rt: r, info: info, runs: map[string]*benchRun{}}
	admin := httpx.RequireAdmin
	mux.Handle("POST /api/v1/bench", admin(http.HandlerFunc(api.start)))
	mux.Handle("GET /api/v1/bench/{id}/events", admin(http.HandlerFunc(api.events)))
	mux.Handle("POST /api/v1/bench/{id}/submit", admin(http.HandlerFunc(api.submit)))
}

func (a *benchAPI) start(w http.ResponseWriter, req *http.Request) {
	var body struct {
		Model string `json:"model"`
		Suite string `json:"suite"`
		Quick *bool  `json:"quick"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "the request body is not valid JSON"))
		return
	}
	if body.Suite == "" {
		body.Suite = bench.SuiteStandard
	}
	if body.Suite != bench.SuiteStandard && body.Suite != bench.SuiteDev {
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "suite must be standard or dev").WithParam("suite"))
		return
	}
	if body.Model == "" {
		body.Model = a.rt.cfg.Models.Default
	}
	m, err := a.rt.manifest(body.Model)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	var llama backend.Backend
	for _, b := range a.rt.backends {
		if b.Name() == config.BackendLlamaCpp {
			llama = b
		}
	}
	if llama == nil || a.rt.store == nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeCapabilityUnavailable, "the benchmark needs the llama.cpp backend"))
		return
	}
	modelPath, drafterPath, err := a.rt.store.Ready(m)
	if err != nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeModelUnavailable, "%v", err))
		return
	}
	if a.rt.cfg.Models.Overrides[m.ID].Speculative.Type == config.SpecNone {
		drafterPath = ""
	}
	opts := bench.Options{
		Manifest: m, ModelPath: modelPath, DrafterPath: drafterPath, Backend: llama, LibDir: a.info.LibDir,
		Relaid: func() (registry.Relaid, error) { return a.rt.store.Relaid(m) },
		Suite:  body.Suite, Full: body.Quick != nil && !*body.Quick,
		Version: a.info.Version, Tag: a.info.Tag, Commit: a.info.Commit,
	}
	presets, err := bench.ChoosePresets(req.Context(), opts)
	if err != nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeInternal, "%v", err))
		return
	}
	opts.Presets = presets

	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		httpx.WriteProblem(w, core.Errorf(core.CodeQueueFull, "a benchmark is already running"))
		return
	}
	a.running = true
	var b [8]byte
	_, _ = rand.Read(b[:])
	run := &benchRun{id: hex.EncodeToString(b[:]), notify: make(chan struct{})}
	a.runs[run.id] = run
	a.mu.Unlock()

	// The workers need the memory the resident model holds.
	for _, k := range a.rt.keys(m.ID) {
		_ = a.rt.Pool.Unload(k)
	}
	for _, p := range presets {
		run.push("progress", map[string]any{"preset": p, "status": "pending"})
	}
	opts.Events = func(e bench.Event) {
		switch e.Type {
		case "start":
			run.push("progress", map[string]any{"preset": e.Preset, "status": "running", "percent": 100 * e.Index / max(e.Total, 1)})
		case "result":
			run.push("progress", map[string]any{"preset": e.Preset, "status": e.Run.Status, "run": e.Run})
		}
	}
	go func() {
		defer func() {
			a.mu.Lock()
			a.running = false
			a.mu.Unlock()
		}()
		res, err := bench.Run(context.Background(), opts)
		if err != nil {
			run.push("error", httpx.ProblemOf(core.Errorf(core.CodeInternal, "%v", err)))
			return
		}
		run.mu.Lock()
		run.result = res
		run.mu.Unlock()
		run.push("result", res)
		run.push("done", map[string]any{})
	}()
	httpx.WriteJSON(w, http.StatusAccepted, map[string]any{"id": run.id, "presets": presets})
}

func (a *benchAPI) find(w http.ResponseWriter, id string) *benchRun {
	a.mu.Lock()
	defer a.mu.Unlock()
	run, ok := a.runs[id]
	if !ok {
		httpx.WriteProblem(w, core.Errorf(core.CodeJobNotFound, "no benchmark run %q", id))
		return nil
	}
	return run
}

// events replays everything so far, then follows the run.
func (a *benchAPI) events(w http.ResponseWriter, req *http.Request) {
	run := a.find(w, req.PathValue("id"))
	if run == nil {
		return
	}
	st, err := httpx.NewStream(w)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	defer st.Close()
	sent := 0
	for {
		run.mu.Lock()
		pending := run.events[sent:]
		done, wait := run.done, run.notify
		run.mu.Unlock()
		for _, e := range pending {
			if st.Send("", e.name, e.payload) != nil {
				return
			}
			sent++
		}
		if done {
			return
		}
		select {
		case <-wait:
		case <-st.Heartbeat():
			if st.Comment("keep-alive") != nil {
				return
			}
		case <-req.Context().Done():
			return
		}
	}
}

func (a *benchAPI) submit(w http.ResponseWriter, req *http.Request) {
	run := a.find(w, req.PathValue("id"))
	if run == nil {
		return
	}
	var body struct {
		ConsentVersion string `json:"consent_version"`
		Anonymous      bool   `json:"anonymous"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "the request body is not valid JSON"))
		return
	}
	if body.ConsentVersion != bench.ConsentVersion {
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "consent %q is not the current text (%s); reload the page", body.ConsentVersion, bench.ConsentVersion).WithParam("consent_version"))
		return
	}
	cfg := a.rt.cfg.Bench
	if cfg.Submit == "never" {
		httpx.WriteProblem(w, core.Errorf(core.CodeForbidden, "bench.submit is never on this server"))
		return
	}
	run.mu.Lock()
	res := run.result
	run.mu.Unlock()
	if res == nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "the run has no result yet"))
		return
	}
	id := bench.Anonymous()
	if !body.Anonymous && !cfg.Anonymous {
		var err error
		if id, err = bench.LoadIdentity(a.rt.cfg.Server.DataDir); err != nil {
			httpx.WriteProblem(w, core.Errorf(core.CodeInternal, "%v", err))
			return
		}
	}
	ctx, cancel := context.WithTimeout(req.Context(), time.Minute)
	defer cancel()
	c := &bench.Client{BaseURL: cfg.SubmitURL}
	rcpt, err := c.Submit(ctx, res, id)
	if err != nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeModelUnavailable, "the benchmark service: %v", err))
		return
	}
	httpx.WriteJSON(w, http.StatusOK, rcpt)
}
