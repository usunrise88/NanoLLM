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
	"github.com/usunrise88/nanollm/internal/registry"
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
//
// The server never starts a run by itself, with one exception: nanollm
// managed, which an application runs in the background and no person tunes,
// starts runs through the Optimizer and has them applied as they end
// (docs/OPTIMIZE.md, §3.7).

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
	// Context ends the runs in progress, and kills their workers, when the
	// server stops; nil lets them run to the end.
	Context context.Context

	// OnStart is told of a run that measures as it starts, OnDone as it ends:
	// with the profile, or with the error. A run that applies itself has
	// applied the profile by then. Neither is told of a dry run.
	OnStart func(OptimizeRun)
	OnDone  func(OptimizeRun, *tune.Profile, error)
	// OnApplied is told of a profile made the one in force, by an apply
	// request or by a run that applies itself: nanollm managed, which has no
	// configuration file, records the mode of an apply request in its
	// settings.
	OnApplied func(OptimizeRun, *tune.Profile)
}

// OptimizeRun is what the hooks and Optimizer.Start tell of a run.
type OptimizeRun struct {
	ID        string    `json:"run"`
	Mode      tune.Mode `json:"mode"`
	Model     string    `json:"model"`
	Reason    string    `json:"reason,omitempty"`
	EstimateS int       `json:"estimate_s"`
	// Apply says the run applies itself as it ends.
	Apply bool `json:"-"`
}

// ReasonAPI is the reason of a run started by POST /api/v1/optimize.
const ReasonAPI = "api"

// StartOptions say what a run measures and what happens when it ends.
type StartOptions struct {
	Mode tune.Mode
	// Model is a catalog id; empty is the default model.
	Model string
	// DryRun looks at the machine and measures nothing.
	DryRun bool
	// Apply makes the profile the one in force as the run ends, as an apply
	// request would.
	Apply bool
	// Reason says why the run was started, for the hooks.
	Reason string
}

// estimateSeconds is what POST answers for how long a run takes, by mode.
var estimateSeconds = map[tune.Mode]int{tune.Silent: 60, tune.Optimal: 45, tune.Powerful: 60}

type optRun struct {
	id string
	*eventLog
	info      OptimizeRun
	mode      tune.Mode
	model     string
	dryRun    bool
	autoApply bool
	start     time.Time
	// finished is closed when the run has ended and the machine is free.
	finished chan struct{}

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

// MountOptimize adds the autotuner's routes, behind the admin check, and
// returns the same autotuner for use without HTTP.
func (r *Runtime) MountOptimize(mux *http.ServeMux, info OptimizeInfo) *Optimizer {
	api := &optimizeAPI{rt: r, info: info, runs: newRuns[*optRun]()}
	admin := httpx.RequireAdmin
	mux.Handle("GET /api/v1/optimize", admin(http.HandlerFunc(api.status)))
	mux.Handle("POST /api/v1/optimize", admin(http.HandlerFunc(api.start)))
	mux.Handle("GET /api/v1/optimize/{id}/events", admin(http.HandlerFunc(api.events)))
	mux.Handle("GET /api/v1/optimize/{id}", admin(http.HandlerFunc(api.get)))
	mux.Handle("POST /api/v1/optimize/{id}/apply", admin(http.HandlerFunc(api.apply)))
	return &Optimizer{api: api}
}

// Optimizer runs the autotuner without HTTP: nanollm managed measures at start
// and after a change of its settings, and applies what it measured.
type Optimizer struct{ api *optimizeAPI }

// Start begins a run and returns at once; the run goes on in the background,
// and its events are on GET /api/v1/optimize/{id}/events like any other's.
func (o *Optimizer) Start(so StartOptions) (OptimizeRun, error) { return o.api.begin(so) }

// Wait blocks until the run has ended or ctx is done, and returns how it ended.
func (o *Optimizer) Wait(ctx context.Context, id string) error {
	run, ok := o.api.runs.get(id)
	if !ok {
		return core.Errorf(core.CodeJobNotFound, "no optimization run %q", id)
	}
	select {
	case <-run.finished:
	case <-ctx.Done():
		return ctx.Err()
	}
	run.mu.Lock()
	defer run.mu.Unlock()
	return run.err
}

// Look is what the mode holds for the model now, its limits and the profile if
// a current one exists, without installing it.
func (o *Optimizer) Look(ctx context.Context, mode tune.Mode, model string) (*tune.Active, error) {
	return o.api.look(ctx, mode, model)
}

// Activate makes mode the runtime's mode for the model, with the profile
// measured for it if a current one exists, and reloads the model by it if the
// model is resident. It measures nothing.
func (o *Optimizer) Activate(ctx context.Context, mode tune.Mode, model string) (*tune.Active, error) {
	model = o.api.modelOr(model)
	act, err := o.api.activate(ctx, mode, model)
	if act == nil || len(o.api.rt.keys(model)) == 0 {
		return act, err
	}
	if _, rerr := o.api.rt.Reload(ctx, model); rerr != nil {
		return act, rerr
	}
	return act, err
}

// MachineKey is tune.MachineKey for the model on the runtime's devices.
func (o *Optimizer) MachineKey(ctx context.Context, mode tune.Mode, model string) (tune.Key, error) {
	m, err := o.api.rt.manifest(o.api.modelOr(model))
	if err != nil {
		return tune.Key{}, err
	}
	return tune.MachineKey(ctx, mode, o.api.input(m)), nil
}

// optimizeStatus is what GET /api/v1/optimize answers.
type optimizeStatus struct {
	Mode        string            `json:"mode"`
	Modes       []modeSummary     `json:"modes"`
	Constraints *tune.Constraints `json:"constraints,omitempty"`
	Profile     *tune.Summary     `json:"profile,omitempty"`
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
		st.Profile = act.Profile.Summary()
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
	run, err := a.begin(StartOptions{Mode: mode, Model: body.Model, DryRun: body.DryRun, Reason: ReasonAPI})
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusAccepted, map[string]any{"id": run.ID, "estimate_s": run.EstimateS})
}

// begin claims the machine and starts a run in the background.
func (a *optimizeAPI) begin(so StartOptions) (OptimizeRun, error) {
	if !so.Mode.Valid() {
		return OptimizeRun{}, core.Errorf(core.CodeInvalidRequest, "mode %q is not silent, optimal or powerful", so.Mode).WithParam("mode")
	}
	m, err := a.rt.manifest(a.modelOr(so.Model))
	if err != nil {
		return OptimizeRun{}, err
	}
	opts, err := a.options(m.ID, so.Mode)
	if err != nil {
		return OptimizeRun{}, err
	}

	end, err := a.rt.beginMeasure("an optimization")
	if err != nil {
		return OptimizeRun{}, err
	}
	var b [8]byte
	_, _ = rand.Read(b[:])
	info := OptimizeRun{ID: hex.EncodeToString(b[:]), Mode: so.Mode, Model: m.ID, Reason: so.Reason, EstimateS: estimateSeconds[so.Mode], Apply: so.Apply && !so.DryRun}
	if so.DryRun {
		info.EstimateS = 10
	}
	run := &optRun{
		id: info.ID, eventLog: newEventLog(), info: info, mode: so.Mode, model: m.ID, dryRun: so.DryRun, autoApply: info.Apply,
		start: time.Now(), finished: make(chan struct{}), state: "running",
	}
	a.runs.add(run.id, run)
	if !so.DryRun && a.info.OnStart != nil {
		a.info.OnStart(info)
	}
	go func() {
		prof, err := a.execute(run, opts)
		// The machine is free before OnDone, so that the hook may start the
		// next run.
		end()
		if !run.dryRun && a.info.OnDone != nil {
			a.info.OnDone(info, prof, err)
		}
		close(run.finished)
	}()
	return info, nil
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

// execute runs the autotuner for a run and records how it ended. A run that
// applies itself is done once the model is reloaded by the new profile.
func (a *optimizeAPI) execute(run *optRun, opts tune.Options) (*tune.Profile, error) {
	ctx := a.info.Context
	if ctx == nil {
		ctx = context.Background()
	}
	fail := func(err error) (*tune.Profile, error) {
		run.mu.Lock()
		run.state, run.err = "error", err
		run.mu.Unlock()
		run.push("error", httpx.ProblemOf(core.Errorf(core.CodeInternal, "%v", err)))
		return nil, err
	}
	if run.dryRun {
		rep, err := tune.Look(ctx, opts)
		if err != nil {
			return fail(err)
		}
		run.mu.Lock()
		run.report, run.state = rep, "done"
		run.mu.Unlock()
		run.push("done", map[string]any{"report": rep})
		return nil, nil
	}
	res, installed, err := a.measure(ctx, run, opts)
	if err != nil {
		return fail(err)
	}
	done := map[string]any{"profile": res.Profile}
	if run.autoApply {
		reloaded := false
		if installed {
			if _, err := a.rt.Reload(ctx, run.model); err != nil {
				a.rt.log.Warn("optimize: the model did not reload by the new profile", "model", run.model, "err", err)
			} else {
				reloaded = true
			}
		}
		done["applied"], done["reloaded"] = true, reloaded
	}
	run.mu.Lock()
	run.result, run.report, run.profile, run.state = res, res.Report, res.Profile, "done"
	run.mu.Unlock()
	run.push("done", done)
	return res.Profile, nil
}

// measure runs the autotuner while the model is held: the workers load their
// own copies of it, so the resident ones go, and requests are held back until
// the run ends. A run that applies itself installs the profile before the hold
// ends, so that the first request after it places by the new profile; whether
// the runtime took it on is installed.
func (a *optimizeAPI) measure(ctx context.Context, run *optRun, opts tune.Options) (res *tune.Result, installed bool, err error) {
	release := a.rt.Hold(run.model, 15*time.Minute)
	defer release()
	for _, k := range a.rt.keys(run.model) {
		_ = a.rt.Pool.Unload(k)
	}
	runner := a.info.Runner
	if runner == nil {
		runner = tune.SpawnRunner{}
	}
	res, err = tune.Run(ctx, tune.RunOptions{Options: opts, Runner: runner, Progress: func(e tune.Event) { run.push(e.Type, e) }})
	if err != nil || !run.autoApply {
		return res, false, err
	}
	if _, installed, err = a.install(ctx, run, res.Profile); err != nil {
		return nil, false, err
	}
	return res, installed, nil
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
	prof, dry := run.profile, run.dryRun
	run.mu.Unlock()
	switch {
	case dry:
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "a dry run measured nothing, so it has nothing to apply"))
		return
	case prof == nil:
		httpx.WriteProblem(w, core.Errorf(core.CodeConflict, "the run has no result yet"))
		return
	}
	ar, installed, err := a.install(req.Context(), run, prof)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	reloaded := false
	if installed {
		if _, err := a.rt.Reload(req.Context(), run.model); err == nil {
			reloaded = true
		}
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]any{
		"profile": prof.ID(), "config_written": ar.ConfigWritten, "reloaded": reloaded,
		"restart_required": ar.RestartRequired, "note": ar.Note,
	})
}

// install makes a measured profile the one in force, once per run: it saves
// it, writes tune.mode into the configuration file when there is one, and
// installs the mode and the profile in the runtime (the configuration the
// runtime holds is the old one, so the mode is set on a copy). Whether the
// runtime took it on is installed; the caller reloads the model.
func (a *optimizeAPI) install(ctx context.Context, run *optRun, prof *tune.Profile) (ar tune.ApplyResult, installed bool, err error) {
	run.mu.Lock()
	if run.applied {
		run.mu.Unlock()
		return ar, false, core.Errorf(core.CodeConflict, "the profile was applied already")
	}
	run.applied = true
	run.mu.Unlock()
	cfgPath := a.info.ConfigPath
	if cfgPath != "" {
		if _, err := os.Stat(cfgPath); err != nil {
			cfgPath = ""
		}
	}
	ar, err = tune.Apply(prof, tune.ApplyOptions{Store: tune.Store{Dir: a.rt.cfg.TunePath()}, ConfigPath: cfgPath})
	if err != nil {
		run.mu.Lock()
		run.applied = false
		run.mu.Unlock()
		return ar, false, core.Errorf(core.CodeInternal, "%v", err)
	}
	act, lerr := a.activate(ctx, prof.Mode, run.model)
	if a.info.OnApplied != nil {
		a.info.OnApplied(run.info, prof)
	}
	return ar, lerr == nil && act != nil, nil
}

// input is what tune.LoadActive and tune.MachineKey look at for a model.
func (a *optimizeAPI) input(m registry.Manifest) tune.ActiveInput {
	return tune.ActiveInput{
		Config: a.rt.cfg, Manifest: m, Devices: a.rt.ProbedDevices(), LibDir: a.info.LibDir,
		LibsRelease: a.info.LibsRelease, LlamaCommit: a.info.LlamaCommit,
	}
}

// modelOr is the model, or the default model for none.
func (a *optimizeAPI) modelOr(model string) string {
	if model == "" {
		return a.rt.DefaultModel()
	}
	return model
}

// look reads what the mode holds for the model: its limits and the profile.
func (a *optimizeAPI) look(ctx context.Context, mode tune.Mode, model string) (*tune.Active, error) {
	m, err := a.rt.manifest(a.modelOr(model))
	if err != nil {
		return nil, err
	}
	in := a.input(m)
	in.Config.Tune.Mode = string(mode)
	return tune.LoadActive(ctx, in)
}

// activate installs what look reads. The limits of a mode hold even when its
// profile cannot be read; the error says why it is not in force.
func (a *optimizeAPI) activate(ctx context.Context, mode tune.Mode, model string) (*tune.Active, error) {
	act, err := a.look(ctx, mode, model)
	if act != nil || err == nil {
		a.rt.SetTune(act)
	}
	return act, err
}
