package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/usunrise88/nanollm/internal/api/adapter"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/llamacpp"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/queue"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/server"
	"github.com/usunrise88/nanollm/internal/ui"
)

// shutdownGrace is how long in-flight requests get after SIGTERM.
const shutdownGrace = 30 * time.Second

// serve runs the server from a console: SIGINT and SIGTERM shut it down, and
// a second signal during a stuck shutdown terminates the process.
func serve(args []string) error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	return serveContext(ctx, stop, args)
}

// serveContext runs the server until ctx is done. A console run cancels it on
// a signal, the Windows service on a stop request from the control manager.
// release, when not nil, is called once shutdown begins, so that the signal
// handler can be given back to the default disposition.
func serveContext(ctx context.Context, release func(), args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	cfgPath := configFlag(fs)
	addr := fs.String("addr", "", "listen address (overrides config)")
	logFile := fs.String("log-file", "", "append the log to this file instead of stderr (overrides log.file)")
	if _, err := parseFlags(fs, args); err != nil {
		return err
	}

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	// Flags win over everything, so they are applied after Load and the
	// result validated again: -addr 0.0.0.0:8080 in open mode must still fail.
	if *addr != "" {
		cfg.Server.Addr = *addr
		if err := cfg.Validate(); err != nil {
			return err
		}
	}

	if *logFile != "" {
		cfg.Log.File = *logFile
	}

	// Before anything is opened: a placement re-executes the process.
	if err := placeProcess(cfg.Devices.NUMA); err != nil {
		slog.Warn("devices.numa", "value", cfg.Devices.NUMA, "err", err)
	}

	logw, closeLog, err := openLog(cfg.Log.File)
	if err != nil {
		return fmt.Errorf("open the log file: %w", err)
	}
	defer closeLog()
	log := newLogger(cfg.Log, logw)
	// The backends and the queue log through the default logger; without
	// this a service, which has no stderr, would lose their lines.
	slog.SetDefault(log)
	log.Info("starting nanollm", "version", version, "llama_cpp", llamacpp.PinnedTag,
		"addr", cfg.Server.Addr, "dialects", cfg.API.Dialects, "backends", cfg.Backends.Enabled)

	backends := openBackends(cfg, log)
	defer func() {
		for _, b := range backends {
			_ = b.Close()
		}
	}()
	loaded := make([]map[string]string, 0, len(backends))
	for _, b := range backends {
		loaded = append(loaded, map[string]string{"name": b.Name(), "version": b.Version()})
	}

	catalog, err := registry.Builtin()
	if err != nil {
		return err
	}
	rt := server.New(ctx, server.Options{
		Config: cfg, Catalog: catalog, Store: newStore(cfg), Backends: backends, Log: log,
	})
	defer rt.Close()
	go rt.Preload(ctx)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]any{
			"status": "ok", "version": version, "llama_cpp": llamacpp.PinnedTag, "backends": loaded,
		})
	})
	// Not ready until the default model is loaded (SPEC §12), and the answer
	// says why: a load balancer only needs the status, an operator needs the
	// reason.
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) {
		if ok, reason := rt.Ready(); !ok {
			httpx.WriteJSON(w, http.StatusServiceUnavailable, map[string]any{
				"status": "not_ready", "reason": reason, "model": cfg.Models.Default,
			})
			return
		}
		httpx.WriteJSON(w, http.StatusOK, map[string]any{"status": "ready", "model": cfg.Models.Default})
	})

	snapshot := func() any {
		out, err := redactedConfig(cfg)
		if err != nil {
			log.Error("cannot render the configuration snapshot", "err", err)
			return map[string]string{"error": "the configuration could not be rendered"}
		}
		return out
	}
	// A queue that cannot open its database (no data_dir yet, a read-only
	// disk) leaves the jobs routes unavailable; chat and completions still
	// serve, as with a backend that fails to open.
	var jobs any
	if cfg.Queue.Enabled {
		q, err := openQueue(cfg, rt.Engine, log)
		if err != nil {
			log.Error("job queue unavailable", "err", err)
		} else {
			defer func() { _ = q.Close() }()
			jobs = q
		}
	}
	if err := adapter.MountAll(mux, cfg.API.Dialects, adapter.Deps{
		Service:        rt.Engine,
		Admin:          rt,
		DefaultModel:   cfg.Models.Default,
		Catalog:        catalog,
		ConfigSnapshot: snapshot,
		Extra:          jobs,
	}); err != nil {
		return err
	}

	rt.MountStats(mux)
	if libDir, err := backend.ResolveLibDir(cfg.Backends.LibDir); err == nil {
		rt.MountBench(mux, server.BenchInfo{LibDir: libDir, Version: version, Tag: llamacpp.PinnedTag, Commit: llamacpp.PinnedCommit})
	}

	public := []string{"/healthz", "/readyz"}
	if cfg.UI.Enabled {
		if h, err := ui.Handler(); err != nil {
			log.Warn("ui not mounted", "err", err)
		} else {
			mux.Handle(ui.Path, h)
			// A browser does not send a bearer token when it loads a script
			// tag; the SPA asks for a key on its first 401 instead (SPEC §11).
			// "/ui" too, so the mux's redirect to "/ui/" is not a 401.
			public = append(public, ui.Path, strings.TrimSuffix(ui.Path, "/"))
			log.Info("ui mounted", "path", ui.Path)
		}
	}

	mw := []httpx.Middleware{
		httpx.WithRequestID(), httpx.Recover(log), httpx.SecurityHeaders(), httpx.AccessLog(log),
		httpx.LimitBody(cfg.Server.MaxBody),
	}
	if cfg.Auth.Mode == "open" {
		log.Warn("authentication disabled", "addr", cfg.Server.Addr,
			"note", "open mode is only permitted on a loopback address")
		// Whoever is on the loopback is the interactive user.
		mw = append(mw, func(next http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				next.ServeHTTP(w, r.WithContext(core.WithInteractive(r.Context(), true)))
			})
		})
	} else {
		keys, err := httpx.NewKeyStore(keySpecs(cfg.Auth.Keys))
		if err != nil {
			return err
		}
		mw = append(mw, httpx.Auth(keys, public...), httpx.RateLimit(httpx.NewRateLimiter()))
		log.Info("authentication enabled", "keys", keys.Names(), "public", public)
	}

	srv := &http.Server{
		Addr:              cfg.Server.Addr,
		Handler:           httpx.Chain(mux, mw...),
		ReadHeaderTimeout: 10 * time.Second,
	}
	errCh := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		// Logged as well as returned: a service's stderr goes nowhere, and
		// "address already in use" is the commonest reason it stops.
		log.Error("server stopped", "err", err)
		return err
	case <-ctx.Done():
		log.Info("shutting down", "grace", shutdownGrace)
	}
	// A second signal during a stuck shutdown falls through to the default
	// disposition and terminates.
	if release != nil {
		release()
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownGrace)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}

// newStore is the on-disk model store the configuration describes.
func openQueue(cfg config.Config, svc core.Service, log *slog.Logger) (*queue.Queue, error) {
	if err := os.MkdirAll(cfg.Server.DataDir, 0o750); err != nil {
		return nil, err
	}
	return queue.Open(svc, queue.Options{
		Path:          filepath.Join(cfg.Server.DataDir, "nanollm.db"),
		MaxPending:    cfg.Queue.MaxPending,
		Concurrency:   cfg.Queue.Concurrency,
		WebhookSecret: cfg.Queue.WebhookSecret,
		RetentionDays: cfg.Queue.RetentionDays,
		Log:           log,
	})
}

func newStore(cfg config.Config) *registry.Store {
	return &registry.Store{
		Root:          cfg.ModelsPath(),
		Downloader:    &registry.Downloader{Mirrors: cfg.Registry.Mirrors},
		StrictLicense: cfg.Registry.StrictLicense,
	}
}

// openBackends opens every enabled backend that can be opened. A backend whose
// libraries are missing is logged and skipped, never fatal (SPEC §3): a
// machine without the CUDA add-on still serves on Vulkan or the CPU.
func openBackends(cfg config.Config, log *slog.Logger) []backend.Backend {
	libDir, err := backend.ResolveLibDir(cfg.Backends.LibDir)
	if err != nil {
		log.Error("cannot resolve backends.lib_dir", "err", err)
		return nil
	}
	var out []backend.Backend
	for _, name := range cfg.Backends.Enabled {
		b, err := backend.Open(name, libDir)
		if err != nil {
			log.Warn("backend unavailable", "backend", name, "err", err)
			continue
		}
		log.Info("backend opened", "backend", name, "version", b.Version(), "lib_dir", libDir)
		out = append(out, b)
	}
	return out
}

// keySpecs adapts configured keys to what the key store takes.
func keySpecs(keys []config.APIKey) []httpx.KeySpec {
	out := make([]httpx.KeySpec, 0, len(keys))
	for _, k := range keys {
		out = append(out, httpx.KeySpec{
			Name: k.Name, Secret: k.Secret, Admin: k.Admin, Interactive: k.Interactive, RPS: k.RPS,
		})
	}
	return out
}

// redactedConfig is the effective configuration as /api/v1/config serves it:
// every secret replaced, and decoded back into a generic tree so the JSON
// field names match the YAML a person would write.
func redactedConfig(cfg config.Config) (any, error) {
	keys := make([]config.APIKey, len(cfg.Auth.Keys))
	copy(keys, cfg.Auth.Keys)
	for i := range keys {
		keys[i].Secret = "redacted"
	}
	cfg.Auth.Keys = keys
	if cfg.Queue.WebhookSecret != "" {
		cfg.Queue.WebhookSecret = "redacted"
	}

	b, err := yaml.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := yaml.Unmarshal(b, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func newLogger(c config.Log, w io.Writer) *slog.Logger {
	level := slog.LevelInfo
	switch c.Level {
	case "debug":
		level = slog.LevelDebug
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	}
	opts := &slog.HandlerOptions{Level: level}
	if c.Format == "text" {
		return slog.New(slog.NewTextHandler(w, opts))
	}
	return slog.New(slog.NewJSONHandler(w, opts))
}
