package server

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/pool"
	"github.com/usunrise88/nanollm/internal/registry"
)

// The runtime over the fake backend: the catalog, the device policy, the
// placement and the pool, with no HTTP and no files. The pin lifecycle is
// what keeps a model a request-free server still answers from memory.

// newRuntime is a runtime holding at most resident models, with the catalog
// alpha (the default) and beta, on the fake backend's CPU.
func newRuntime(t *testing.T, resident int, fo fake.Options, mutate ...func(*config.Config)) *Runtime {
	t.Helper()
	cfg := config.Default()
	cfg.Models.Default, cfg.Models.Preload = "alpha", nil
	cfg.Models.Overrides = map[string]config.ModelOverride{}
	cfg.Backends.Enabled = []string{config.BackendFake}
	cfg.Devices.Default = config.DeviceCPU
	cfg.Pool = config.Pool{MaxResidentModels: resident, MaxPending: 8}
	for _, f := range mutate {
		f(&cfg)
	}
	fo.NoGPU = true
	rt := New(context.Background(), Options{
		Config:   cfg,
		Catalog:  []registry.Manifest{{ID: "alpha", Family: "fake"}, {ID: "beta", Family: "fake"}},
		Backends: []backend.Backend{fake.New(fo)},
		Log:      slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	t.Cleanup(rt.Close)
	return rt
}

func chat(t *testing.T, rt *Runtime, model, text string) *core.ChatResult {
	t.Helper()
	res, err := rt.Engine.Chat(context.Background(), core.ChatRequest{
		Generation: core.Generation{Model: model},
		Messages:   []core.Message{{Role: core.RoleUser, Content: text}},
	}, nil)
	if err != nil {
		t.Fatalf("chat on %s: %v", model, err)
	}
	return res
}

// resident maps each loaded model to its instance's status.
func resident(rt *Runtime) map[string]pool.Status {
	out := map[string]pool.Status{}
	for _, s := range rt.Pool.Status() {
		out[s.Key.Model] = s
	}
	return out
}

func info(t *testing.T, rt *Runtime, id string) core.ModelInfo {
	t.Helper()
	all, err := rt.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range all {
		if m.ID == id {
			return m
		}
	}
	t.Fatalf("no model %q in %+v", id, all)
	return core.ModelInfo{}
}

func code(err error) core.Code {
	var ce *core.Error
	if errors.As(err, &ce) {
		return ce.Code
	}
	return ""
}

func TestModelsAndLoad(t *testing.T) {
	rt := newRuntime(t, 2, fake.Options{})
	ctx := context.Background()
	if ok, why := rt.Ready(); ok || why == "" {
		t.Fatalf("ready before anything is loaded: %v %q", ok, why)
	}
	a := info(t, rt, "alpha")
	if !a.Default || a.Pinned || len(a.Loaded) != 0 || info(t, rt, "beta").Default {
		t.Fatalf("before loading: %+v", a)
	}
	got, err := rt.Load(ctx, "alpha")
	if err != nil || len(got.Loaded) != 1 {
		t.Fatalf("load: %+v, %v", got, err)
	}
	if ok, why := rt.Ready(); !ok {
		t.Fatalf("not ready with the default model loaded: %q", why)
	}
	if _, err := rt.Load(ctx, "gamma"); code(err) != core.CodeModelNotFound {
		t.Fatalf("an unknown model: %v", err)
	}
	// Loading again reuses the instance, the fake placement is the default's.
	if _, err := rt.Load(ctx, "alpha"); err != nil || len(resident(rt)) != 1 {
		t.Fatalf("a second load: %v, %d resident", err, len(resident(rt)))
	}
	st := resident(rt)["alpha"]
	if st.Key.Backend != config.BackendFake || st.Key.Device != "cpu" || st.Key.NCtx != 4096 || st.Loop.Slots != 1 {
		t.Fatalf("placement %+v", st)
	}
	if err := rt.Unload(ctx, "alpha"); err != nil {
		t.Fatal(err)
	}
	if ok, _ := rt.Ready(); ok || len(resident(rt)) != 0 {
		t.Fatal("still resident after Unload")
	}
	if err := rt.Unload(ctx, "alpha"); code(err) != core.CodeInvalidRequest {
		t.Fatalf("unloading what is not loaded: %v", err)
	}
	if err := rt.Unload(ctx, "gamma"); code(err) != core.CodeModelNotFound {
		t.Fatalf("unloading an unknown model: %v", err)
	}
}

// A pinned model is never evicted for another: with room for one model, the
// second cannot load until the first is unpinned.
func TestPinnedModelIsNotEvicted(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	ctx := context.Background()
	if _, err := rt.Load(ctx, "alpha"); err != nil {
		t.Fatal(err)
	}
	if err := rt.Pin(ctx, "alpha", true); err != nil {
		t.Fatal(err)
	}
	if !info(t, rt, "alpha").Pinned || !resident(rt)["alpha"].Pinned {
		t.Fatal("the pin is not reported")
	}
	if _, err := rt.Load(ctx, "beta"); code(err) != core.CodeModelUnavailable {
		t.Fatalf("loading past a pinned model: %v", err)
	}
	if _, err := rt.Engine.Chat(ctx, core.ChatRequest{Generation: core.Generation{Model: "beta"}, Messages: []core.Message{{Role: core.RoleUser, Content: "x"}}}, nil); code(err) != core.CodeModelUnavailable {
		t.Fatalf("a request for a model that cannot be made room for: %v", err)
	}
	if _, ok := resident(rt)["alpha"]; !ok || len(resident(rt)) != 1 {
		t.Fatalf("the pinned model left: %v", resident(rt))
	}
	// It still serves.
	if res := chat(t, rt, "alpha", "still here [[fake:echo]]"); res.Message.Content != "still here" {
		t.Fatalf("%q", res.Message.Content)
	}

	if err := rt.Pin(ctx, "alpha", false); err != nil {
		t.Fatal(err)
	}
	if info(t, rt, "alpha").Pinned {
		t.Fatal("still pinned after unpinning")
	}
	if _, err := rt.Load(ctx, "beta"); err != nil {
		t.Fatalf("an unpinned model should make room: %v", err)
	}
	if r := resident(rt); len(r) != 1 || r["beta"].Key.Model != "beta" {
		t.Fatalf("resident after the eviction: %v", r)
	}
	if err := rt.Pin(ctx, "gamma", true); code(err) != core.CodeModelNotFound {
		t.Fatalf("pinning an unknown model: %v", err)
	}
}

// A model in use cannot be evicted either way; once its request ends, an
// unpinned one can.
func TestPinnedWhileBusy(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{DecodeRate: 200})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := rt.Engine.Chat(ctx, core.ChatRequest{Messages: []core.Message{{Role: core.RoleUser, Content: "busy [[fake:loop]]"}}}, nil)
		done <- err
	}()
	eventually(t, "the request to hold the model", func() bool { return resident(rt)["alpha"].Refs == 1 })
	if _, err := rt.Load(ctx, "beta"); code(err) != core.CodeModelUnavailable {
		t.Fatalf("loading past a model in use: %v", err)
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("the request ended with %v", err)
	}
	eventually(t, "the model to be released", func() bool { return resident(rt)["alpha"].Refs == 0 })
	if _, err := rt.Load(context.Background(), "beta"); err != nil {
		t.Fatalf("after the request ended: %v", err)
	}
}

// Pin is for the model, not for one instance: asked before the model is
// loaded, it holds for the instance a later load makes, whether the admin
// loads it or its first request does.
func TestPinBeforeTheModelLoads(t *testing.T) {
	for _, tc := range []struct {
		name string
		load func(t *testing.T, rt *Runtime)
	}{
		{"loaded by the admin", func(t *testing.T, rt *Runtime) {
			if _, err := rt.Load(context.Background(), "alpha"); err != nil {
				t.Fatal(err)
			}
		}},
		{"loaded by a request", func(t *testing.T, rt *Runtime) { chat(t, rt, "alpha", "first request [[fake:echo]]") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rt := newRuntime(t, 1, fake.Options{})
			if err := rt.Pin(context.Background(), "alpha", true); err != nil {
				t.Fatal(err)
			}
			if len(resident(rt)) != 0 {
				t.Fatal("pinning loaded the model")
			}
			tc.load(t, rt)
			if !resident(rt)["alpha"].Pinned || !info(t, rt, "alpha").Pinned {
				t.Fatalf("the instance loaded after the pin is not pinned: %+v", resident(rt)["alpha"])
			}
			if _, err := rt.Load(context.Background(), "beta"); code(err) != core.CodeModelUnavailable {
				t.Fatalf("a pinned model was evicted: %v", err)
			}
		})
	}
}

// A reload hands the pin to the new instance and unloads the old one; with a
// changed placement the old instance drains while the new one is pinned.
func TestReloadKeepsThePin(t *testing.T) {
	overrides := map[string]config.ModelOverride{}
	rt := newRuntime(t, 2, fake.Options{DecodeRate: 200}, func(c *config.Config) { c.Models.Overrides = overrides })
	ctx := context.Background()
	if _, err := rt.Load(ctx, "alpha"); err != nil {
		t.Fatal(err)
	}
	if err := rt.Pin(ctx, "alpha", true); err != nil {
		t.Fatal(err)
	}
	before := resident(rt)["alpha"]
	got, err := rt.Reload(ctx, "alpha")
	if err != nil || len(got.Loaded) != 1 || !got.Pinned {
		t.Fatalf("reload: %+v, %v", got, err)
	}
	after := resident(rt)["alpha"]
	if !after.Pinned || !after.LoadedAt.After(before.LoadedAt) {
		t.Fatalf("the reloaded instance: pinned=%v, loaded %v then %v", after.Pinned, before.LoadedAt, after.LoadedAt)
	}

	// A different placement is a different key. A request is running on the
	// old instance, which drains; the new one takes over, pinned.
	reqCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := rt.Engine.Chat(reqCtx, core.ChatRequest{Messages: []core.Message{{Role: core.RoleUser, Content: "draining [[fake:loop]]"}}}, nil)
		done <- err
	}()
	eventually(t, "the request to hold the old instance", func() bool { return resident(rt)["alpha"].Refs == 1 })
	overrides["alpha"] = config.ModelOverride{NCtx: config.Auto{N: 8192}}
	if _, err := rt.Reload(ctx, "alpha"); err != nil {
		t.Fatal(err)
	}
	var old, fresh *pool.Status
	for _, s := range rt.Pool.Status() {
		switch s.Key.NCtx {
		case 4096:
			old = &s
		case 8192:
			fresh = &s
		}
	}
	if old == nil || !old.Retired || old.Refs != 1 {
		t.Fatalf("the old instance should be draining its request: %+v", old)
	}
	if fresh == nil || fresh.Retired || !fresh.Pinned {
		t.Fatalf("the new instance should be live and pinned: %+v", fresh)
	}
	cancel()
	<-done
	eventually(t, "the old instance to unload", func() bool {
		for _, s := range rt.Pool.Status() {
			if s.Key.NCtx == 4096 {
				return false
			}
		}
		return true
	})
	if len(rt.Pool.Status()) != 1 || !info(t, rt, "alpha").Pinned {
		t.Fatalf("after the drain: %+v", rt.Pool.Status())
	}
}

// An admin's Unload goes through a pin (the pin guards against eviction, not
// against the operator), and the pin is still the model's when it loads again.
func TestUnloadPinnedModel(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	ctx := context.Background()
	if _, err := rt.Load(ctx, "alpha"); err != nil {
		t.Fatal(err)
	}
	if err := rt.Pin(ctx, "alpha", true); err != nil {
		t.Fatal(err)
	}
	if err := rt.Unload(ctx, "alpha"); err != nil {
		t.Fatal(err)
	}
	if len(resident(rt)) != 0 {
		t.Fatal("a pinned model survived an explicit unload")
	}
	if _, err := rt.Load(ctx, "alpha"); err != nil || !resident(rt)["alpha"].Pinned {
		t.Fatalf("the pin was forgotten across an unload: %v", err)
	}
}

func TestPreloadLoadsTheListedModels(t *testing.T) {
	rt := newRuntime(t, 2, fake.Options{}, func(c *config.Config) { c.Models.Preload = []string{"gamma", "beta"} })
	rt.Preload(context.Background())
	// The unknown model is logged and skipped; the next one still loads.
	if r := resident(rt); len(r) != 1 || r["beta"].Key.Model != "beta" {
		t.Fatalf("resident after preload: %v", r)
	}
}

func TestPickDevice(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	if d, err := rt.pickDevice(""); err != nil || d.Selector() != "fake:cpu" {
		t.Fatalf("the configured default: %v %v", d, err)
	}
	// A device that does not exist falls back to the CPU unless fallback is off.
	if d, err := rt.pickDevice("fake:gpu9"); err != nil || d.Type != backend.CPU {
		t.Fatalf("fallback: %v %v", d, err)
	}
	rt.cfg.Devices.Fallback = false
	if _, err := rt.pickDevice("fake:gpu9"); code(err) != core.CodeDeviceUnavailable {
		t.Fatalf("no fallback: %v", err)
	}
	if _, err := rt.Engine.Chat(context.Background(), core.ChatRequest{
		Generation: core.Generation{Device: "gpu"},
		Messages:   []core.Message{{Role: core.RoleUser, Content: "x"}},
	}, nil); code(err) != core.CodeDeviceUnavailable {
		t.Fatalf("a request for a missing device class: %v", err)
	}
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}
