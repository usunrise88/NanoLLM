package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/managed"
	"github.com/usunrise88/nanollm/internal/posture"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/server"
	"github.com/usunrise88/nanollm/internal/tune"
)

// managedGrace is how long in-flight requests get when the application goes:
// a child that lingers keeps the application's next start waiting on the
// lock, and queued jobs survive in the database anyway.
const managedGrace = 5 * time.Second

// The tests replace these: managedRunner measures in place of the worker
// processes (a test binary cannot be started as `nanollm optimize worker`),
// and managedPosture records the priority instead of setting the test
// process's own.
var (
	managedRunner  tune.Runner
	managedPosture = posture.ApplyLevel
)

// managedCommand runs NanoLLM as the background process of an application:
// see internal/managed for the contract and nanollm-embedded.md for the
// application's side of it.
//
// Every way it can fail to start is said on stdout as a fatal event before it
// returns, because the application reads that and nothing else.
func managedCommand(ctx context.Context, args []string, stdin io.Reader, stdout io.Writer) error {
	events := managed.NewEvents(stdout)
	fail := func(code string, err error) error {
		events.Fatal(code, err.Error())
		return err
	}

	fs := flag.NewFlagSet("managed", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	dataDir := fs.String("data-dir", "", "where settings, models, profiles and the log live (default: the user's application data directory)")
	parentPID := fs.Int("parent-pid", 0, "stop when this process exits")
	watchStdin := fs.Bool("watch-stdin", true, "stop when standard input closes")
	lowPriority := fs.Bool("low-priority", true, "run below normal priority in every mode, not only the quiet one")
	ui := fs.Bool("ui", false, "serve the web UI at /ui")
	testBackend := fs.String("test-backend", "", "run on the scripted backend (fake), for tests")
	if _, err := parseFlags(fs, args); err != nil {
		return fail("invalid_arguments", err)
	}
	if *testBackend != "" && *testBackend != config.BackendFake {
		return fail("invalid_arguments", fmt.Errorf("-test-backend must be %s", config.BackendFake))
	}

	managed.IgnoreBrokenPipe()

	dir := *dataDir
	if dir == "" {
		d, err := managed.DefaultDataDir()
		if err != nil {
			return fail("data_dir_unavailable", err)
		}
		dir = d
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return fail("data_dir_unavailable", err)
	}
	paths := managed.PathsFor(dir)
	if err := paths.Create(); err != nil {
		return fail("data_dir_unavailable", err)
	}
	lock, err := managed.Lock(paths.Lock)
	if errors.Is(err, managed.ErrLocked) {
		return fail("data_dir_locked", fmt.Errorf("%w: %s", err, dir))
	}
	if err != nil {
		return fail("data_dir_unavailable", err)
	}
	defer lock.Close()

	logw, closeLog, err := openLog(paths.Log)
	if err != nil {
		return fail("data_dir_unavailable", err)
	}
	defer closeLog()

	settings, configured, err := managed.LoadSettings(paths.Settings)
	if err != nil {
		// Unreadable settings must not keep the application from starting
		// its engine: it is set up again, and the old file is kept to look at.
		bad := paths.Settings + ".invalid"
		_ = os.Rename(paths.Settings, bad)
		events.Warn(managed.WarnSettingsReset, fmt.Sprintf("the settings could not be read and were moved to %s: %v", bad, err), nil)
		settings, configured = managed.Settings{}, false
	}

	token, err := config.NewKeySecret()
	if err != nil {
		return fail("internal", err)
	}
	cfg, err := managed.NewPlan(settings, managed.Options{Paths: paths, Token: token, UI: *ui, Backend: *testBackend})
	if err != nil {
		return fail("settings_invalid", err)
	}
	// The level is a variable: the application changes it without a restart.
	level := new(slog.LevelVar)
	level.Set(logLevel(cfg.Log.Level))
	log := slog.New(slog.NewJSONHandler(logw, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(log)
	log.Info("starting nanollm managed", "version", version, "data_dir", dir, "configured", configured,
		"mode", settings.TuneMode(), "model", settings.Model(), "low_priority", *lowPriority)

	// Set before the backends start their threads, so that they inherit it.
	setPosture := func(mode tune.Mode) error { return managedPosture(managedLevel(mode, *lowPriority)) }
	if err := setPosture(settings.TuneMode()); err != nil {
		log.Warn("managed: cannot set the priority", "err", err)
	}

	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	ctx, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()
	// Set before the cancels above run on the way out: the context ending
	// because this function returned — after a fatal event, say — is not a
	// stop anybody asked for, and must not be reported as one.
	var returned atomic.Bool
	defer returned.Store(true)
	stop := func(reason string) { cancel(errors.New(reason)) }
	if *watchStdin {
		managed.WatchStdin(stdin, stop)
	}
	if *parentPID > 0 {
		if err := managed.WatchParent(ctx, *parentPID, stop); err != nil {
			return fail("parent_gone", err)
		}
	}
	go func() {
		<-ctx.Done()
		if returned.Load() {
			return
		}
		reason := "signal"
		if cause := context.Cause(ctx); cause != nil && !errors.Is(cause, context.Canceled) {
			reason = cause.Error()
		}
		log.Info("stopping", "reason", reason)
		events.Emit(managed.EventStopping, map[string]any{"reason": reason})
	}()

	// The port is the system's choice: a fixed one is taken on some machine
	// sooner or later, and on the loopback interface no firewall asks.
	ln, err := (&net.ListenConfig{}).Listen(ctx, "tcp", "127.0.0.1:0")
	if err != nil {
		return fail("listen_failed", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port

	store := newStore(cfg)
	var (
		ctrl *managed.Controller
		auto *managed.Autotune
	)
	err = runServer(ctx, cfg, log, runOptions{
		listener:      ln,
		middleware:    []httpx.Middleware{httpx.RequireHost(httpx.LoopbackHosts(strconv.Itoa(port))...)},
		store:         store,
		ownTune:       true,
		shutdownGrace: managedGrace,
		closeLate:     true,
		admin: func(a core.Admin) core.Admin {
			return notifyingAdmin{Admin: a, downloaded: func(id string) { ctrl.Downloaded(id) }}
		},
		optimize: func(info *server.OptimizeInfo) {
			if managedRunner != nil {
				info.Runner = managedRunner
			}
			info.Test = *testBackend != ""
			info.OnStart = func(run server.OptimizeRun) { auto.Started(runInfo(run)) }
			info.OnDone = func(run server.OptimizeRun, p *tune.Profile, err error) { auto.Done(runInfo(run), p, err) }
			info.OnApplied = func(run server.OptimizeRun, p *tune.Profile) {
				// The application applied a run of its own: its mode becomes the
				// setting, as tune.mode in the file is the service's.
				if auto.Applied(runInfo(run), p) && run.Model == ctrl.Settings().Model() {
					if err := ctrl.AdoptMode(p.Mode); err != nil {
						log.Warn("managed: the mode cannot be saved", "mode", p.Mode, "err", err)
					}
					auto.Kick("")
				}
			}
		},
		mount: func(mux *http.ServeMux, rt *server.Runtime, opt *server.Optimizer) {
			auto = managed.NewAutotune(managed.AutotuneOptions{
				Tuner:    optimizerTuner{opt: opt, rt: rt},
				Events:   events,
				Failures: managed.Failures{Dir: paths.Tune, Version: version},
				Settings: func() managed.Settings { return ctrl.Settings() },
				Present:  rt.Present,
				Posture:  setPosture,
				Load: func(ctx context.Context, id string) error {
					_, err := rt.Load(ctx, id)
					return err
				},
				Log: log,
			})
			ctrl = managed.NewController(managed.ControllerOptions{
				Version:    version,
				Paths:      paths,
				Settings:   settings,
				Configured: configured,
				Lookup: func(id string) (registry.Manifest, error) {
					cat, err := registry.Builtin()
					if err != nil {
						return registry.Manifest{}, err
					}
					m, ok := registry.Find(cat, id)
					if !ok {
						return registry.Manifest{}, core.Errorf(core.CodeModelNotFound, "no model %q in the catalog", id)
					}
					return m, nil
				},
				Present: rt.Present,
				Loaded:  func() bool { ok, _ := rt.Ready(); return ok },
				Apply: func(prev, next managed.Settings) {
					applyManagedSettings(rt, store, level, prev, next)
				},
				Autotune: auto,
			})
			ctrl.Mount(mux)
			auto.Prepare(ctx)
			go auto.Run(ctx)
		},
		serving: func(addr net.Addr) {
			st := ctrl.Status()
			log.Info("managed instance ready", "addr", addr.String(), "configured", st.Configured,
				"missing_models", st.MissingModels, "tune", st.Features.Autotune.State)
			events.Emit(managed.EventReady, map[string]any{
				"url":              "http://" + addr.String(),
				"port":             port,
				"token":            token,
				"pid":              os.Getpid(),
				"version":          version,
				"api_version":      managed.APIVersion,
				"data_dir":         dir,
				"configured":       st.Configured,
				"restart_required": st.RestartRequired,
				"missing_models":   st.MissingModels,
				"tune":             map[string]any{"state": st.Features.Autotune.State, "mode": st.Features.Autotune.Mode},
			})
			// After ready: the application hears of a measurement as it starts.
			auto.Kick("")
		},
		stopped: func() {
			// A measurement in progress ends with the context; its workers are
			// killed, and must be before this process goes.
			if auto != nil {
				auto.WaitIdle(managedGrace)
			}
		},
	})
	if err != nil {
		return fail("server_failed", err)
	}
	return nil
}

// managedLevel is the priority a managed instance runs at in a mode: the
// quiet mode is quiet whatever the flag says, as it is for the service.
func managedLevel(mode tune.Mode, low bool) posture.Level {
	switch {
	case mode == tune.Silent:
		return posture.Quiet
	case low:
		return posture.Below
	}
	return posture.Normal
}

func logLevel(s string) slog.Level {
	switch s {
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	}
	return slog.LevelInfo
}

// applyManagedSettings takes on, in the running process, what changed apart
// from the mode and the model's placement, which the autotuner puts in force.
func applyManagedSettings(rt *server.Runtime, store *registry.Store, level *slog.LevelVar, prev, next managed.Settings) {
	if prev.Model() != next.Model() {
		rt.SetDefaultModel(next.Model())
	}
	if prev.Thinking != next.Thinking {
		rt.Engine.SetThinking(next.Thinking)
	}
	if !slices.Equal(prev.Mirrors, next.Mirrors) && store.Downloader != nil {
		store.Downloader.SetMirrors(next.Mirrors)
	}
	if prev.LogLevel != next.LogLevel {
		level.Set(logLevel(next.LogLevel))
	}
}

// notifyingAdmin tells the managed controller of a download that finished:
// the default model's arrival is the moment to measure.
type notifyingAdmin struct {
	core.Admin
	downloaded func(id string)
}

func (a notifyingAdmin) Download(ctx context.Context, id string, progress func(core.DownloadProgress)) error {
	err := a.Admin.Download(ctx, id, progress)
	if err == nil {
		a.downloaded(id)
	}
	return err
}

// errNoAutotuner is what the autotuner answers when the native libraries
// cannot be located, and so it was never mounted.
var errNoAutotuner = core.Errorf(core.CodeCapabilityUnavailable, "the autotuner is unavailable: the native libraries cannot be located")

// optimizerTuner is the server's autotuner as internal/managed uses it.
type optimizerTuner struct {
	opt *server.Optimizer
	rt  *server.Runtime
}

func (t optimizerTuner) Current() *tune.Active { return t.rt.Tune() }

func (t optimizerTuner) Look(ctx context.Context, mode tune.Mode, model string) (*tune.Active, error) {
	if t.opt == nil {
		return nil, errNoAutotuner
	}
	return t.opt.Look(ctx, mode, model)
}

func (t optimizerTuner) Activate(ctx context.Context, mode tune.Mode, model string) (*tune.Active, error) {
	if t.opt == nil {
		return nil, errNoAutotuner
	}
	return t.opt.Activate(ctx, mode, model)
}

func (t optimizerTuner) MachineKey(ctx context.Context, mode tune.Mode, model string) (tune.Key, error) {
	if t.opt == nil {
		return tune.Key{}, errNoAutotuner
	}
	return t.opt.MachineKey(ctx, mode, model)
}

func (t optimizerTuner) Start(mode tune.Mode, model, reason string) (managed.RunInfo, error) {
	if t.opt == nil {
		return managed.RunInfo{}, errNoAutotuner
	}
	run, err := t.opt.Start(server.StartOptions{Mode: mode, Model: model, Apply: true, Reason: reason})
	return runInfo(run), err
}

func runInfo(r server.OptimizeRun) managed.RunInfo {
	return managed.RunInfo{ID: r.ID, Mode: r.Mode, Model: r.Model, Reason: r.Reason, EstimateS: r.EstimateS, Apply: r.Apply}
}
