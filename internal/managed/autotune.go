package managed

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/posture"
	"github.com/usunrise88/nanollm/internal/tune"
)

// RunInfo is a measurement as the server's autotuner tells of it.
type RunInfo struct {
	ID        string
	Mode      tune.Mode
	Model     string
	Reason    string
	EstimateS int
	// Apply says the run applies its profile itself as it ends; a run started
	// by POST /api/v1/optimize waits for an apply request.
	Apply bool
}

// Tuner is the server's autotuner as this package uses it: server.Optimizer,
// adapted by the command so that this package does not import the server.
type Tuner interface {
	// Current is the mode and profile in force now.
	Current() *tune.Active
	// Look is what the mode holds for the model now, without installing it.
	Look(ctx context.Context, mode tune.Mode, model string) (*tune.Active, error)
	// Activate installs the mode and its current profile, and reloads the
	// model by them if it is resident.
	Activate(ctx context.Context, mode tune.Mode, model string) (*tune.Active, error)
	// MachineKey is the key a measurement would have now.
	MachineKey(ctx context.Context, mode tune.Mode, model string) (tune.Key, error)
	// Start begins a measurement that applies itself as it ends.
	Start(mode tune.Mode, model, reason string) (RunInfo, error)
}

// Reasons a measurement is started for, as tune_started says.
const (
	// ReasonMissing: at the start, the mode has no profile for the model.
	ReasonMissing = "missing"
	// ReasonStale: at the start, the profile no longer matches the machine.
	ReasonStale = "stale"
	// ReasonDownload: the model has just been downloaded.
	ReasonDownload = "download"
	// ReasonSettings: the application changed the mode or the model.
	ReasonSettings = "settings"
	// ReasonRequested: the application asked (POST /api/v1/managed/tune).
	ReasonRequested = "requested"
)

// Autotune states, as GET /api/v1/managed reports them.
const (
	// TuneOff: the settings switched measuring off, and no profile is in force.
	TuneOff = "off"
	// TuneWaitingModel: the model is not on disk; the measurement follows its
	// download.
	TuneWaitingModel = "waiting_model"
	// TuneRunning: a measurement is in progress, and requests to the model get
	// 503 with a Retry-After until it ends.
	TuneRunning = "running"
	// TuneReady: a current profile of the mode and model is in force.
	TuneReady = "ready"
	// TuneMissing and TuneStale: the mode has no current profile, and a
	// measurement is due.
	TuneMissing = "missing"
	TuneStale   = "stale"
	// TuneFailed: the last measurement failed; the mode's limits hold until
	// something changes or the application asks again.
	TuneFailed = "failed"
)

// TuneState is the autotuner as GET /api/v1/managed reports it.
type TuneState struct {
	State    string         `json:"state"`
	Mode     tune.Mode      `json:"mode"`
	Model    string         `json:"model"`
	Run      string         `json:"run,omitempty"`
	Profile  *tune.Summary  `json:"profile,omitempty"`
	Changes  []tune.Change  `json:"changes,omitempty"`
	Problems []tune.Problem `json:"problems,omitempty"`
	Message  string         `json:"message,omitempty"`
}

// AutotuneOptions builds an Autotune.
type AutotuneOptions struct {
	Tuner    Tuner
	Events   *Events
	Failures Failures
	// Settings are the settings in force now.
	Settings func() Settings
	// Present reports whether a model is on disk.
	Present func(id string) bool
	// Posture sets the priority the process runs at in a mode;
	// posture.ErrRestart says it cannot be raised while the process runs.
	Posture func(tune.Mode) error
	// Load makes the model resident, as the preload of a start would.
	Load func(ctx context.Context, model string) error
	Log  *slog.Logger
}

// Autotune keeps the mode and the model of the settings in force, and
// measures when the mode has no current profile for the model. One goroutine
// (Run) does the work, so that settings changed in quick succession, a
// download that finishes and a measurement that ends are taken one at a time.
type Autotune struct {
	o    AutotuneOptions
	wake chan struct{}

	mu sync.Mutex
	// reason is why the next look was asked for; empty at the start.
	reason string
	// run is the measurement in progress.
	run *RunInfo
	// failed is why the last measurement of a model and mode failed, while
	// that stands, by pair.
	failed map[string]string
	// skipped are the pairs whose measurement a start skipped for a recorded
	// failure, said once each.
	skipped map[string]bool
	// manual are the pairs the application measured itself through
	// POST /api/v1/optimize: their result waits for its apply request, and
	// the loop does not measure them meanwhile.
	manual map[string]bool
	// restart says the mode wants a higher priority than this process can
	// take back; announced, that it was said.
	restart, announced bool
	// active is what was last put in force.
	active activeKey
	// ctx is the instance's: once it is done, a measurement that ends is a
	// stop, not a failure.
	ctx context.Context
}

type activeKey struct {
	mode    tune.Mode
	model   string
	profile string
}

// NewAutotune builds the loop; Run starts it.
func NewAutotune(o AutotuneOptions) *Autotune {
	if o.Log == nil {
		o.Log = slog.Default()
	}
	return &Autotune{o: o, wake: make(chan struct{}, 1), failed: map[string]string{}, skipped: map[string]bool{}, manual: map[string]bool{}}
}

// pair names a model and mode in failed and skipped.
func pair(model string, mode tune.Mode) string { return model + "/" + string(mode) }

// Kick asks the loop to look at the settings again: to put their mode and
// model in force and to measure if the mode has no current profile. reason is
// what a measurement it starts says it was started for.
func (a *Autotune) Kick(reason string) {
	a.mu.Lock()
	if reason != "" {
		a.reason = reason
	}
	a.mu.Unlock()
	select {
	case a.wake <- struct{}{}:
	default:
	}
}

// Prepare puts the mode of the settings in force before the server answers,
// so that the first request places by it and the first status is right. It
// loads nothing and measures nothing: Run does, after Kick.
func (a *Autotune) Prepare(ctx context.Context) {
	a.mu.Lock()
	a.ctx = ctx
	a.mu.Unlock()
	s := a.o.Settings()
	mode, model := s.TuneMode(), s.Model()
	a.posture(mode)
	act, err := a.o.Tuner.Activate(ctx, mode, model)
	if err != nil {
		a.o.Log.Warn("managed: the profile cannot be used", "mode", mode, "model", model, "err", err)
	}
	a.mu.Lock()
	a.active = activeKey{mode: mode, model: model, profile: act.ID()}
	a.mu.Unlock()
	if act == nil || act.Profile != nil || !s.AutotuneOn() {
		return
	}
	if key, err := a.o.Tuner.MachineKey(ctx, mode, model); err == nil {
		if rec, ok := a.o.Failures.Matches(model, mode, key); ok {
			a.mu.Lock()
			a.failed[pair(model, mode)] = rec.Error
			a.mu.Unlock()
		}
	}
}

// Run is the loop. It returns when ctx is done.
func (a *Autotune) Run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-a.wake:
		}
		a.settle(ctx)
	}
}

func (a *Autotune) settle(ctx context.Context) {
	a.mu.Lock()
	if a.run != nil {
		// The end of the measurement looks again, with the reason kept.
		a.mu.Unlock()
		return
	}
	reason := a.reason
	a.reason = ""
	a.mu.Unlock()

	s := a.o.Settings()
	mode, model := s.TuneMode(), s.Model()
	a.posture(mode)
	act, err := a.o.Tuner.Look(ctx, mode, model)
	if err != nil {
		a.o.Log.Warn("managed: the profile cannot be used", "mode", mode, "model", model, "err", err)
	}
	a.mu.Lock()
	manual := a.manual[pair(model, mode)]
	a.mu.Unlock()
	if s.AutotuneOn() && !manual && a.o.Present(model) && act != nil && act.Profile == nil && a.measure(ctx, mode, model, reason, act) {
		return
	}
	want := activeKey{mode: mode, model: model, profile: act.ID()}
	a.mu.Lock()
	same := a.active == want
	a.mu.Unlock()
	if !same {
		got, err := a.o.Tuner.Activate(ctx, mode, model)
		if err != nil {
			a.o.Log.Warn("managed: the mode is not fully in force", "mode", mode, "model", model, "err", err)
		}
		a.mu.Lock()
		a.active = activeKey{mode: mode, model: model, profile: got.ID()}
		a.mu.Unlock()
	}
	if a.o.Present(model) {
		// What a start's preload does; a model already resident stays as it is.
		if err := a.o.Load(ctx, model); err != nil {
			a.o.Log.Warn("managed: the model did not load", "model", model, "err", err)
		}
	}
}

// measure starts a measurement unless the last one failed on this very
// machine, and reports whether one runs.
func (a *Autotune) measure(ctx context.Context, mode tune.Mode, model, reason string, act *tune.Active) bool {
	if key, err := a.o.Tuner.MachineKey(ctx, mode, model); err == nil {
		if rec, ok := a.o.Failures.Matches(model, mode, key); ok {
			p := pair(model, mode)
			a.mu.Lock()
			first := !a.skipped[p]
			a.skipped[p], a.failed[p] = true, rec.Error
			a.mu.Unlock()
			if first {
				a.o.Events.Warn(WarnTuneSkipped, "the last measurement failed on this machine and build; the mode's limits hold without a profile",
					map[string]any{"mode": mode, "model": model, "error": rec.Error, "failed_at": rec.At})
			}
			return false
		}
	}
	if reason == "" {
		reason = ReasonMissing
		if act.Stale {
			reason = ReasonStale
		}
	}
	if _, err := a.o.Tuner.Start(mode, model, reason); err != nil {
		// Another measurement (a benchmark) has the machine; this one waits for
		// the next look.
		a.o.Log.Warn("managed: the measurement did not start", "mode", mode, "model", model, "err", err)
		return false
	}
	return true
}

// Request starts a measurement of the settings' mode and model now, whether
// or not the mode has a current profile and whatever the last one did.
func (a *Autotune) Request(ctx context.Context) (RunInfo, error) {
	s := a.o.Settings()
	mode, model := s.TuneMode(), s.Model()
	if !a.o.Present(model) {
		return RunInfo{}, core.Errorf(core.CodeConflict, "model %s is not on disk; download it first", model).WithParam("default_model")
	}
	a.posture(mode)
	return a.o.Tuner.Start(mode, model, ReasonRequested)
}

// Started is the server's OnStart: every measurement, the application's own
// through POST /api/v1/optimize included.
func (a *Autotune) Started(run RunInfo) {
	a.mu.Lock()
	a.run = &run
	a.mu.Unlock()
	a.o.Events.Emit(EventTuneStarted, map[string]any{
		"run": run.ID, "mode": run.Mode, "model": run.Model, "reason": run.Reason, "estimate_s": run.EstimateS,
	})
}

// Done is the server's OnDone. A failure is recorded against the machine as it
// is, so that the next start does not measure again until something changes.
func (a *Autotune) Done(run RunInfo, prof *tune.Profile, err error) {
	ctx := context.Background()
	a.mu.Lock()
	a.run = nil
	stopping := a.ctx != nil && a.ctx.Err() != nil
	a.mu.Unlock()
	if stopping {
		// The instance stops, and the measurement with it: nothing failed.
		return
	}
	if err != nil {
		if key, kerr := a.o.Tuner.MachineKey(ctx, run.Mode, run.Model); kerr == nil {
			if rerr := a.o.Failures.Record(run.Model, run.Mode, key, err); rerr != nil {
				a.o.Log.Warn("managed: the failure cannot be recorded", "err", rerr)
			}
		}
		p := pair(run.Model, run.Mode)
		a.mu.Lock()
		a.failed[p] = err.Error()
		a.skipped[p] = true // said now, as tune_failed
		a.mu.Unlock()
		a.o.Log.Warn("managed: the measurement failed", "run", run.ID, "mode", run.Mode, "model", run.Model, "err", err)
		a.o.Events.Warn(WarnTuneFailed, err.Error(), map[string]any{"run": run.ID, "mode": run.Mode, "model": run.Model})
	} else {
		a.o.Failures.Clear(run.Model, run.Mode)
		p := pair(run.Model, run.Mode)
		a.mu.Lock()
		delete(a.failed, p)
		delete(a.skipped, p)
		if !run.Apply {
			a.manual[p] = true
		}
		a.mu.Unlock()
		fields := map[string]any{"run": run.ID, "mode": run.Mode, "model": run.Model, "applied": run.Apply}
		if sum := prof.Summary(); sum != nil {
			fields["profile"], fields["device"], fields["plan_mode"] = sum.ID, sum.Device, sum.PlanMode
			fields["answer_s"], fields["measured_bytes"] = sum.AnswerS, sum.Measured
		}
		a.o.Events.Emit(EventTuneDone, fields)
	}
	// The settings may have changed while it measured, and the model was
	// unloaded for it: look again.
	a.Kick("")
}

// WaitIdle waits up to max for the measurement in progress to end: at a stop,
// its workers are killed with the instance's context, and must be gone before
// the process is.
func (a *Autotune) WaitIdle(max time.Duration) {
	deadline := time.Now().Add(max)
	for time.Now().Before(deadline) {
		a.mu.Lock()
		idle := a.run == nil
		a.mu.Unlock()
		if idle {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// Applied is the server's OnApplied: the profile of a run is in force now.
// It reports whether the run was an apply request of the application's
// (POST /api/v1/optimize/{id}/apply), whose mode the settings should take on.
func (a *Autotune) Applied(run RunInfo, prof *tune.Profile) bool {
	a.mu.Lock()
	a.active = activeKey{mode: prof.Mode, model: run.Model, profile: prof.ID()}
	delete(a.manual, pair(run.Model, prof.Mode))
	a.mu.Unlock()
	return !run.Apply
}

// State is the autotuner as the status reports it.
func (a *Autotune) State() TuneState {
	s := a.o.Settings()
	st := TuneState{Mode: s.TuneMode(), Model: s.Model()}
	a.mu.Lock()
	run, failed := a.run, a.failed[pair(st.Model, st.Mode)]
	a.mu.Unlock()
	cur := a.o.Tuner.Current()
	inForce := cur != nil && cur.Mode == st.Mode && a.activeModel() == st.Model
	if inForce {
		st.Problems, st.Changes = cur.Problems, cur.Changes
		st.Profile = cur.Profile.Summary()
	}
	switch {
	case run != nil:
		st.State, st.Run, st.Mode, st.Model = TuneRunning, run.ID, run.Mode, run.Model
	case inForce && cur.Profile != nil:
		st.State = TuneReady
	case !a.o.Present(st.Model):
		st.State, st.Message = TuneWaitingModel, "the model is not on disk; the measurement follows its download"
		if !s.AutotuneOn() {
			st.State, st.Message = TuneOff, ""
		}
	case failed != "":
		st.State, st.Message = TuneFailed, failed
	case !s.AutotuneOn():
		st.State = TuneOff
	case inForce && cur.Stale:
		st.State, st.Message = TuneStale, cur.Message()
	default:
		st.State = TuneMissing
	}
	return st
}

func (a *Autotune) activeModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.active.model
}

// RestartReasons are what this process cannot take on while it runs: the
// priority of a mode above the quiet one, on a system where a process cannot
// raise its own priority again.
func (a *Autotune) RestartReasons() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.restart {
		return []string{"priority"}
	}
	return nil
}

func (a *Autotune) posture(mode tune.Mode) {
	err := a.o.Posture(mode)
	if err != nil && !errors.Is(err, posture.ErrRestart) {
		a.o.Log.Warn("managed: cannot set the priority", "mode", mode, "err", err)
	}
	a.mu.Lock()
	a.restart = errors.Is(err, posture.ErrRestart)
	say := a.restart && !a.announced
	a.announced = a.restart
	a.mu.Unlock()
	if say {
		a.o.Events.Emit(EventRestartRequired, map[string]any{"reasons": []string{"priority"}})
	}
}
