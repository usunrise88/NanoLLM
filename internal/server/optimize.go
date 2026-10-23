package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/tune"
)

// The autotuner over HTTP (docs/OPTIMIZE.md, §4.2), for external systems, the
// installers and the UI:
//
//	GET  /api/v1/optimize                      the mode, the profile, is it stale, the problems
//	POST /api/v1/optimize {mode, model?, dry_run?}  → 202 {id, estimate_s}
//	GET  /api/v1/optimize/{id}/events          SSE: survey, problem, candidate, probe_start, probe, decision, done, error
//	GET  /api/v1/optimize/{id}                 the state and the result
//	POST /api/v1/optimize/{id}/apply           make the profile the one in force
//
// One measurement at a time on the machine (a benchmark counts). While a run
// measures, the model is held back: requests to it get 503 with a Retry-After,
// and the job queue waits for it. The measurement runs inside the service, as
// the account and with the devices the service runs with.

// OptimizeInfo is what the runtime needs to run the autotuner.
type OptimizeInfo struct {
	LibDir      string
	Version     string
	LibsRelease string
	LlamaTag    string
	LlamaCommit string
	// ConfigPath is the configuration file tune.mode is written to; empty
	// leaves it alone.
	ConfigPath string
	// Runner measures; nil runs the worker processes of this binary.
	Runner tune.Runner
	// Test leaves out the questions put to the OS (the tests run on a scripted
	// backend and need only its devices).
	Test bool
}

// estimateSeconds is what POST answers for how long a run takes, by mode.
var estimateSeconds = map[tune.Mode]int{tune.Silent: 60, tune.Optimal: 45, tune.Powerful: 60}

type optRun struct {
	id string
	*eventLog
	mode   tune.Mode
	model  string
	dryRun bool
	start  time.Time

	state   string // running | done | error
	result  *tune.Result
	report  *tune.Report
	profile *tune.Profile
	err     error
	applied bool
}

type optimizeAPI struct {
	rt   *Runtime
	info OptimizeInfo
	runs *runs[*optRun]
}

// MountOptimize adds the autotuner's routes, behind the admin check.
func (r *Runtime) MountOptimize(mux *http.ServeMux, info OptimizeInfo) {
	api := &optimizeAPI{rt: r, info: info, runs: newRuns[*optRun]()}
	admin := httpx.RequireAdmin
	mux.Handle("GET /api/v1/optimize", admin(http.HandlerFunc(api.status)))
	mux.Handle("POST /api/v1/optimize", admin(http.HandlerFunc(api.start)))
	mux.Handle("GET /api/v1/optimize/{id}/events", admin(http.HandlerFunc(api.events)))
	mux.Handle("GET /api/v1/optimize/{id}", admin(http.HandlerFunc(api.get)))
	mux.Handle("POST /api/v1/optimize/{id}/apply", admin(http.HandlerFunc(api.apply)))
}

// optimizeStatus is what GET /api/v1/optimize answers.
type optimizeStatus struct {
	Mode        string            `json:"mode"`
	Modes       []modeSummary     `json:"modes"`
	Constraints *tune.Constraints `json:"constraints,omitempty"`
	Profile     *profileSummary   `json:"profile,omitempty"`
	Stale       bool              `json:"stale"`
	Missing     bool              `json:"missing,omitempty"`
	Changes     []tune.Change     `json:"changes,omitempty"`
	Suggest     *tune.Suggest     `json:"suggest,omitempty"`
	Message     string            `json:"message,omitempty"`
	Problems    []tune.Problem    `json:"problems,omitempty"`
	Running     string            `json:"running,omitempty"`
}

type modeSummary struct {
	Mode    string `json:"mode"`
	Summary string `json:"summary"`
}

type profileSummary struct {
	ID       string    `json:"id"`
	Created  time.Time `json:"created"`
	Device   any       `json:"device"`
	PlanMode any       `json:"plan_mode"`
	NCtx     int       `json:"n_ctx"`
	MTP      any       `json:"mtp"`
	AnswerS  float64   `json:"answer_s"`
	Measured int64     `json:"measured_bytes"`
}

func modes() []modeSummary {
	var out []modeSummary
	for _, m := range tune.Modes {
		out = append(out, modeSummary{string(m), m.Summary()})
	}
	return out
}

func (a *optimizeAPI) status(w http.ResponseWriter, req *http.Request) {
	st := optimizeStatus{Modes: modes()}
	act := a.rt.Tune()
	if act != nil {
		st.Mode = string(act.Mode)
		st.Constraints = &act.Constraints
		st.Problems = act.Problems
		st.Message = act.Message()
		s := act.Staleness()
		st.Stale, st.Missing, st.Changes, st.Suggest = s.Stale, s.Missing, s.Changes, s.Suggest
		if p := act.Profile; p != nil {
			st.Profile = &profileSummary{
				ID: p.ID(), Created: p.Created, Device: p.Chosen.Device.Value, PlanMode: p.Chosen.PlanMode.Value, NCtx: p.Chosen.NCtx,
				MTP: p.Chosen.MTP.Value, AnswerS: p.Answer.AnswerS, Measured: p.Memory.MeasuredBytes,
			}
		}
	}
	a.rt.mu.Lock()
	st.Running = a.rt.measuring
	a.rt.mu.Unlock()
	httpx.WriteJSON(w, http.StatusOK, st)
}

func (a *optimizeAPI) start(w http.ResponseWriter, req *http.Request) {
	var body struct {
		Mode   string `json:"mode"`
		Model  string `json:"model"`
		DryRun bool   `json:"dry_run"`
	}
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "the request body is not valid JSON"))
		return
	}
	mode, err := tune.ParseMode(body.Mode)
	if err != nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "%v", err).WithParam("mode"))
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
	opts, err := a.options(m.ID, mode)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}

	end, err := a.rt.beginMeasure("an optimization")
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	var b [8]byte
	_, _ = rand.Read(b[:])
	run := &optRun{id: hex.EncodeToString(b[:]), eventLog: newEventLog(), mode: mode, model: m.ID, dryRun: body.DryRun, start: time.Now(), state: "running"}
	a.runs.add(run.id, run)

	go func() {
		defer end()
		a.execute(run, opts)
	}()
	est := estimateSeconds[mode]
	if body.DryRun {
		est = 10
	}
	httpx.WriteJSON(w, http.StatusAccepted, map[string]any{"id": run.id, "estimate_s": est})
}

// options builds what the autotuner looks at from the runtime.
func (a *optimizeAPI) options(model string, mode tune.Mode) (tune.Options, error) {
	m, err := a.rt.manifest(model)
	if err != nil {
		return tune.Options{}, err
	}
	var backends []backend.Backend
	for _, b := range a.rt.backends {
		if b.Name() == config.BackendLlamaCpp || b.Name() == config.BackendFake {
			backends = append(backends, b)
		}
	}
	if len(backends) == 0 {
		return tune.Options{}, core.Errorf(core.CodeCapabilityUnavailable, "the autotuner needs the llama.cpp backend")
	}
	modelPath, drafterPath := "fake.gguf", "fake-drafter.gguf"
	if backends[0].Name() != config.BackendFake {
		if a.rt.store == nil {
			return tune.Options{}, core.Errorf(core.CodeModelUnavailable, "no model store is configured")
		}
		if modelPath, drafterPath, err = a.rt.store.Ready(m); err != nil {
			return tune.Options{}, core.Errorf(core.CodeModelUnavailable, "%v", err)
		}
	}
	ov := a.rt.cfg.Models.Overrides[m.ID]
	if ov.Speculative.Type == config.SpecNone {
		drafterPath = ""
	}
	return tune.Options{
		Mode: mode,
		Survey: tune.SurveyOptions{
			Backends: backends, LibDir: a.info.LibDir, Manifest: m, ModelPath: modelPath, DrafterPath: drafterPath,
			Version: a.info.Version, LibsRelease: a.info.LibsRelease, LlamaTag: a.info.LlamaTag, LlamaCommit: a.info.LlamaCommit,
			SkipDisk: a.info.Test, SkipMachine: a.info.Test,
		},
		Plan: tune.PlanInput{
			Manifest: m, Override: ov, ModelPath: modelPath, DrafterPath: drafterPath,
			Threads: a.rt.cfg.Devices.CPUThreads, ThreadsBatch: a.rt.cfg.Devices.CPUThreadsBatch,
		},
	}, nil
}

// execute runs the autotuner for a run and records how it ended.
func (a *optimizeAPI) execute(run *optRun, opts tune.Options) {
	ctx := context.Background()
	progress := func(e tune.Event) { run.push(e.Type, e) }
	finish := func(state string, err error) {
		run.mu.Lock()
		run.state, run.err = state, err
		run.mu.Unlock()
	}
	if run.dryRun {
		rep, err := tune.Look(ctx, opts)
		if err != nil {
			finish("error", err)
			run.push("error", httpx.ProblemOf(core.Errorf(core.CodeInternal, "%v", err)))
			return
		}
		run.mu.Lock()
		run.report = rep
		run.mu.Unlock()
		finish("done", nil)
		run.push("done", map[string]any{"report": rep})
		return
	}
	// The workers load their own copies of the model: the resident ones go, and
	// requests are held back until the run ends.
	release := a.rt.Hold(run.model, 15*time.Minute)
	defer release()
	for _, k := range a.rt.keys(run.model) {
		_ = a.rt.Pool.Unload(k)
	}
	runner := a.info.Runner
	if runner == nil {
		runner = tune.SpawnRunner{}
	}
	res, err := tune.Run(ctx, tune.RunOptions{Options: opts, Runner: runner, Progress: progress})
	if err != nil {
		finish("error", err)
		run.push("error", httpx.ProblemOf(core.Errorf(core.CodeInternal, "%v", err)))
		return
	}
	run.mu.Lock()
	run.result, run.report, run.profile = res, res.Report, res.Profile
	run.mu.Unlock()
	finish("done", nil)
	run.push("done", map[string]any{"profile": res.Profile})
}

func (a *optimizeAPI) find(w http.ResponseWriter, id string) *optRun {
	run, ok := a.runs.get(id)
	if !ok {
		httpx.WriteProblem(w, core.Errorf(core.CodeJobNotFound, "no optimization run %q", id))
		return nil
	}
	return run
}

func (a *optimizeAPI) events(w http.ResponseWriter, req *http.Request) {
	if run := a.find(w, req.PathValue("id")); run != nil {
		run.stream(w, req)
	}
}

func (a *optimizeAPI) get(w http.ResponseWriter, req *http.Request) {
	run := a.find(w, req.PathValue("id"))
	if run == nil {
		return
	}
	run.mu.Lock()
	out := map[string]any{"id": run.id, "state": run.state, "mode": run.mode, "model": run.model, "dry_run": run.dryRun, "applied": run.applied}
	if run.profile != nil {
		out["profile"] = run.profile
	}
	if run.report != nil && run.dryRun {
		out["report"] = run.report
	}
	if run.err != nil {
		out["error"] = httpx.ProblemOf(core.Errorf(core.CodeInternal, "%v", run.err))
	}
	run.mu.Unlock()
	httpx.WriteJSON(w, http.StatusOK, out)
}

// apply makes the profile of a finished run the one in force: it saves it,
// writes tune.mode into the configuration, installs it in the runtime and
// reloads the model by the new plan.
func (a *optimizeAPI) apply(w http.ResponseWriter, req *http.Request) {
	run := a.find(w, req.PathValue("id"))
	if run == nil {
		return
	}
	run.mu.Lock()
	prof, applied, dry := run.profile, run.applied, run.dryRun
	run.mu.Unlock()
	switch {
	case dry:
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "a dry run measured nothing, so it has nothing to apply"))
		return
	case prof == nil:
		httpx.WriteProblem(w, core.Errorf(core.CodeConflict, "the run has no result yet"))
		return
	case applied:
		httpx.WriteProblem(w, core.Errorf(core.CodeConflict, "the profile was applied already"))
		return
	}
	cfgPath := a.info.ConfigPath
	if cfgPath != "" {
		if _, err := os.Stat(cfgPath); err != nil {
			cfgPath = ""
		}
	}
	ar, err := tune.Apply(prof, tune.ApplyOptions{Store: tune.Store{Dir: a.rt.cfg.TunePath()}, ConfigPath: cfgPath})
	if err != nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeInternal, "%v", err))
		return
	}
	// The runtime takes the mode and the profile on without a restart: the
	// configuration it holds is the old one, so the mode is set on a copy.
	cfg := a.rt.cfg
	cfg.Tune.Mode = string(prof.Mode)
	m, _ := a.rt.manifest(run.model)
	act, lerr := tune.LoadActive(req.Context(), tune.ActiveInput{
		Config: cfg, Manifest: m, Devices: a.rt.ProbedDevices(), LibDir: a.info.LibDir,
		LibsRelease: a.info.LibsRelease, LlamaCommit: a.info.LlamaCommit,
	})
	reloaded := false
	if lerr == nil && act != nil {
		a.rt.SetTune(act)
		if _, err := a.rt.Reload(req.Context(), run.model); err == nil {
			reloaded = true
		}
	}
	run.mu.Lock()
	run.applied = true
	run.mu.Unlock()
	httpx.WriteJSON(w, http.StatusOK, map[string]any{
		"profile": prof.ID(), "config_written": ar.ConfigWritten, "reloaded": reloaded,
		"restart_required": ar.RestartRequired, "note": ar.Note,
	})
}
