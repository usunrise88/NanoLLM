package server

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

func defaultModel(t *testing.T) registry.Manifest {
	t.Helper()
	cat, err := registry.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := registry.Find(cat, config.DefaultModel)
	if !ok {
		t.Fatalf("the catalog has no %s", config.DefaultModel)
	}
	return m
}

func cpuRequest(m registry.Manifest) PlaceRequest {
	return PlaceRequest{
		Manifest: m, Backend: config.BackendLlamaCpp,
		Device:    backend.Device{Backend: config.BackendLlamaCpp, ID: "cpu", Type: backend.CPU, Name: "CPU", MemoryTotal: 32 << 30, MemoryFree: 24 << 30},
		ModelPath: "/models/model.gguf", DrafterPath: "/models/drafter.gguf",
		Threads: 6, ThreadsBatch: 12,
	}
}

func TestPlannerPlacerPlacesTheCatalogModel(t *testing.T) {
	m := defaultModel(t)
	req := cpuRequest(m)
	req.Override.Preset = "ram-8g"
	pl, err := PlannerPlacer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	p, ok := pl.Plan.(*plan.Plan)
	if !ok || !p.OK() {
		t.Fatalf("plan %T %+v", pl.Plan, pl.Plan)
	}
	k, s := pl.Key, pl.Spec
	if k.Model != m.ID || k.Backend != config.BackendLlamaCpp || k.Device != "cpu" || k.Preset != "ram-8g" || k.Mode != string(p.Mode) || k.NCtx != p.NCtx {
		t.Errorf("key %+v for plan mode %s, %d context", k, p.Mode, p.NCtx)
	}
	if s.Path != "/models/model.gguf" || s.Device != "llamacpp:cpu" || s.NCtx != p.NCtx || s.Slots < 1 || s.Threads != 6 || s.ThreadsBatch != 12 {
		t.Errorf("spec %+v", s)
	}
	// The drafter only counts when its file is there.
	req.DrafterPath = ""
	if pl, err = PlannerPlacer(context.Background(), req); err != nil || pl.Spec.DrafterPath != "" {
		t.Errorf("no drafter file: %v, spec drafter %q", err, pl.Spec.DrafterPath)
	}
	// Without threads in the request the spec still gets a thread count.
	req.Threads, req.ThreadsBatch = 0, 0
	if pl, err = PlannerPlacer(context.Background(), req); err != nil || pl.Spec.Threads <= 0 || pl.Spec.ThreadsBatch <= 0 {
		t.Errorf("default threads: %v, %d and %d", err, pl.Spec.Threads, pl.Spec.ThreadsBatch)
	}
	// A request the same twice is the same key: one instance in the pool.
	a, _ := PlannerPlacer(context.Background(), cpuRequest(m))
	b, _ := PlannerPlacer(context.Background(), cpuRequest(m))
	if a.Key != b.Key {
		t.Errorf("keys differ: %+v %+v", a.Key, b.Key)
	}
}

// With no preset in the configuration the budget comes from what the device
// has free: three quarters of it on a CPU, rounded down to half a GiB.
func TestPlannerPlacerDerivesThePreset(t *testing.T) {
	req := cpuRequest(defaultModel(t))
	pl, err := PlannerPlacer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if pl.Key.Preset != "ram-18g" {
		t.Errorf("derived preset %q, want ram-18g (3/4 of 24 GiB)", pl.Key.Preset)
	}
	req.Device.MemoryFree, req.Device.MemoryTotal = 0, 0
	if _, err := PlannerPlacer(context.Background(), req); err == nil || !strings.Contains(err.Error(), "reports no memory") {
		t.Errorf("a device with no memory figure: %v", err)
	}
}

func TestPlannerPlacerRefusals(t *testing.T) {
	m := defaultModel(t)
	for _, tc := range []struct {
		name string
		edit func(*PlaceRequest)
		want string
	}{
		{"an unknown preset", func(r *PlaceRequest) { r.Override.Preset = "ram-7g" }, "unknown preset"},
		{"an unknown mode", func(r *PlaceRequest) { r.Override.Preset, r.Override.Mode = "ram-8g", "M99" }, "M99"},
		{"a model with no measured inventory", func(r *PlaceRequest) { r.Manifest = registry.Manifest{ID: "unmeasured"}; r.Override.Preset = "ram-8g" }, ""},
	} {
		req := cpuRequest(m)
		tc.edit(&req)
		_, err := PlannerPlacer(context.Background(), req)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want an error with %q", tc.name, err, tc.want)
		}
	}
	// A mode that cannot fit the preset is refused with the preset named and
	// the shortfall; the planner's own pick for the preset would map the file.
	req := cpuRequest(m)
	req.Override.Preset, req.Override.Mode = "ram-1g", "M6"
	if _, err := PlannerPlacer(context.Background(), req); err == nil || !strings.Contains(err.Error(), "preset ram-1g") {
		t.Errorf("weights in RAM against a 1 GiB preset: %v", err)
	}
}

// The fake backend never goes through the planner: it has no inventory and
// the default placement is all it needs.
func TestPlannerPlacerLeavesTheFakeBackendAlone(t *testing.T) {
	req := PlaceRequest{
		Manifest: registry.Manifest{ID: "alpha", Drafter: &registry.Drafter{}}, Backend: config.BackendFake,
		Device:   backend.Device{Backend: config.BackendFake, ID: "cpu", Type: backend.CPU},
		Override: config.ModelOverride{NCtx: config.Auto{N: 2048}, Slots: config.Auto{N: 3}},
	}
	pl, err := PlannerPlacer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if pl.Key.NCtx != 2048 || pl.Spec.NCtx != 2048 || pl.Spec.Slots != 3 || pl.Spec.DrafterPath != "fake-drafter" || pl.Spec.DraftMax != 3 || pl.Spec.GPULayers != 0 {
		t.Errorf("placement %+v", pl)
	}
	req.Override.Speculative.Type = config.SpecNone
	if pl, _ = PlannerPlacer(context.Background(), req); pl.Spec.DrafterPath != "" {
		t.Errorf("speculative none still has a drafter: %+v", pl.Spec)
	}
	req.Device.Type = backend.GPU
	if pl, _ = PlannerPlacer(context.Background(), req); pl.Spec.GPULayers != -1 {
		t.Errorf("a GPU device offloads everything, got %d layers", pl.Spec.GPULayers)
	}
}

// A placement that fails reaches the client as model_unavailable naming the
// model and device, unless the placer already speaks the API's errors.
func TestPlacementFailures(t *testing.T) {
	for _, tc := range []struct {
		name  string
		place Placer
		want  core.Code
		text  string
	}{
		{"a plain error", func(context.Context, PlaceRequest) (Placement, error) { return Placement{}, errors.New("no room") },
			core.CodeModelUnavailable, "model alpha does not fit fake:cpu: no room"},
		{"an API error", func(context.Context, PlaceRequest) (Placement, error) {
			return Placement{}, core.Errorf(core.CodeCapabilityUnavailable, "no NPU")
		}, core.CodeCapabilityUnavailable, "no NPU"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := config.Default()
			cfg.Models.Default, cfg.Models.Preload = "alpha", nil
			cfg.Devices.Default = config.DeviceCPU
			rt := New(context.Background(), Options{
				Config: cfg, Catalog: []registry.Manifest{{ID: "alpha"}},
				Backends: []backend.Backend{fake.New(fake.Options{NoGPU: true})}, Place: tc.place,
				Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
			})
			t.Cleanup(rt.Close)
			_, err := rt.Engine.Chat(context.Background(), core.ChatRequest{Messages: []core.Message{{Role: core.RoleUser, Content: "x"}}}, nil)
			var ce *core.Error
			if !errors.As(err, &ce) || ce.Code != tc.want || !strings.Contains(ce.Message, tc.text) {
				t.Fatalf("got %v, want %s %q", err, tc.want, tc.text)
			}
			if len(rt.Pool.Status()) != 0 {
				t.Fatal("a failed placement left an instance")
			}
		})
	}
}

// With no backend the runtime says so instead of picking nothing quietly.
func TestNoBackends(t *testing.T) {
	cfg := config.Default()
	cfg.Models.Default, cfg.Models.Preload = "alpha", nil
	rt := New(context.Background(), Options{Config: cfg, Catalog: []registry.Manifest{{ID: "alpha"}}, Log: slog.New(slog.NewTextHandler(io.Discard, nil))})
	t.Cleanup(rt.Close)
	_, err := rt.Load(context.Background(), "alpha")
	var ce *core.Error
	if !errors.As(err, &ce) || ce.Code != core.CodeModelUnavailable {
		t.Fatalf("got %v", err)
	}
	if devs, err := rt.Devices(context.Background()); err != nil || len(devs) != 0 {
		t.Fatalf("devices %v, %v", devs, err)
	}
}

func TestDevicesListsWhatTheBackendsReport(t *testing.T) {
	rt := newRuntime(t, 1, fake.Options{})
	devs, err := rt.Devices(context.Background())
	if err != nil || len(devs) != 1 {
		t.Fatalf("devices %+v, %v", devs, err)
	}
	if d := devs[0]; d.Selector != "fake:cpu" || d.Type != "cpu" || d.MemoryTotal == 0 {
		t.Errorf("device %+v", d)
	}
}

// MTP is off by default on the CPU (the stage 2 sweep found no gain there) and
// a configuration can turn it on; a GPU device keeps its default.
func TestPlannerPlacerLeavesMTPOffOnTheCPU(t *testing.T) {
	m := defaultModel(t)
	req := cpuRequest(m)
	req.Override.Preset = "ram-8g"
	pl, err := PlannerPlacer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	p := pl.Plan.(*plan.Plan)
	if p.MTP.Enabled || pl.Spec.DrafterPath != "" || p.MTP.Reason != CPUMTPOffReason {
		t.Errorf("default on the CPU: MTP %+v, drafter %q", p.MTP, pl.Spec.DrafterPath)
	}
	req.Override.Speculative.Type = config.SpecMTP
	if pl, err = PlannerPlacer(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if p = pl.Plan.(*plan.Plan); !p.MTP.Enabled || pl.Spec.DrafterPath != "/models/drafter.gguf" {
		t.Errorf("speculative.type: mtp: MTP %+v, drafter %q", p.MTP, pl.Spec.DrafterPath)
	}
}

// OPT-8: models.overrides.<id>.stream reaches the load spec of a plan that
// leaves layers in the file, and a plan that does not has no stream spec.
func TestPlannerPlacerCarriesTheStreamSettings(t *testing.T) {
	m := defaultModel(t)
	req := cpuRequest(m)
	req.Override.Stream = config.Stream{Ahead: 3, Keep: 5, Evict: true}

	req.Override.Preset = "ram-8g"
	pl, err := PlannerPlacer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if pl.Spec.Stream != nil {
		t.Errorf("ram-8g holds the layers in memory but has a stream spec: %+v", pl.Spec.Stream)
	}

	req.Override.Preset = "ram-2g"
	if pl, err = PlannerPlacer(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	st := pl.Spec.Stream
	if st == nil || st.Ahead != 3 || st.Keep != 5 || !st.Evict || len(st.Ranges) == 0 {
		t.Errorf("ram-2g stream spec %+v, want ahead 3, keep 5, evict, with ranges", st)
	}
}

// models.overrides.<id>.repack: off keeps the weights as the file has them in
// every CPU mode (the quality gate of docs/QUALITY-STAGE2.md needs it); auto
// and no key leave the planner's choice alone. The planner's own choice
// depends on the platform, so only the off case is asserted outright.
func TestPlannerPlacerRepackKey(t *testing.T) {
	m := defaultModel(t)
	for _, preset := range []string{"ram-8g", "ram-4g"} {
		req := cpuRequest(m)
		req.Override.Preset = preset
		def, err := PlannerPlacer(context.Background(), req)
		if err != nil {
			t.Fatal(err)
		}
		req.Override.Repack = "auto"
		auto, err := PlannerPlacer(context.Background(), req)
		if err != nil {
			t.Fatal(err)
		}
		if auto.Spec.Repack != def.Spec.Repack {
			t.Errorf("%s: repack: auto changed the planner's choice (%v → %v)", preset, def.Spec.Repack, auto.Spec.Repack)
		}
		req.Override.Repack = "off"
		off, err := PlannerPlacer(context.Background(), req)
		if err != nil {
			t.Fatal(err)
		}
		if off.Spec.Repack {
			t.Errorf("%s: repack: off still repacks", preset)
		}
		req.Override.Repack = "on"
		on, err := PlannerPlacer(context.Background(), req)
		if err != nil {
			t.Fatal(err)
		}
		if !on.Spec.Repack {
			t.Errorf("%s: repack: on does not repack", preset)
		}
	}
}

// MTP on the CPU: on by default where the device lists the multi-row Q4_0 kernels
// (draft_max 2, p_min 0.75), off elsewhere with the reason, and the configuration
// wins either way.
func TestPlannerPlacerMTPDefaultOnCPU(t *testing.T) {
	m := defaultModel(t)
	run := func(feat bool, typ string, nmax int) *plan.Plan {
		t.Helper()
		req := cpuRequest(m)
		req.Override.Preset = "ram-8g"
		req.Override.Speculative = config.Speculative{Type: typ, NMax: nmax}
		if feat {
			req.Device.Features = []string{backend.FeatureMultiRowQ4_0}
		}
		pl, err := PlannerPlacer(context.Background(), req)
		if err != nil {
			t.Fatal(err)
		}
		return pl.Plan.(*plan.Plan)
	}
	if p := run(false, "", 0); p.MTP.Enabled || p.MTP.Reason == "" {
		t.Errorf("no multi-row kernels: MTP %+v, want off with a reason", p.MTP)
	}
	p := run(true, "", 0)
	if !p.MTP.Enabled || p.MTP.DraftMax != CPUMTPDraftMax || p.MTP.DraftPMin != CPUMTPDraftPMin {
		t.Errorf("multi-row kernels: MTP %+v, want on with draft_max %d, p_min %v", p.MTP, CPUMTPDraftMax, CPUMTPDraftPMin)
	}
	if p := run(true, config.SpecNone, 0); p.MTP.Enabled {
		t.Errorf("speculative.type: none with the kernels: MTP %+v", p.MTP)
	}
	if p := run(true, config.SpecMTP, 3); !p.MTP.Enabled || p.MTP.DraftMax != 3 {
		t.Errorf("speculative.type: mtp, n_max 3: MTP %+v", p.MTP)
	}
	if p := run(true, config.SpecMTP, 3); p.MTP.VerifyRows != 4 {
		t.Errorf("speculative.type: mtp, n_max 3 with the kernels: verify_rows %d, want 4 (drafts while one sequence decodes)", p.MTP.VerifyRows)
	}
	if p := run(true, "", 0); p.MTP.VerifyRows != 1+CPUMTPDraftMax {
		t.Errorf("the default: verify_rows %d, want %d", p.MTP.VerifyRows, 1+CPUMTPDraftMax)
	}
	if p := run(false, config.SpecMTP, 0); p.MTP.VerifyRows != 0 {
		t.Errorf("without the kernels there is no bound: verify_rows %d", p.MTP.VerifyRows)
	}
	if p := run(false, config.SpecMTP, 0); !p.MTP.Enabled {
		t.Errorf("speculative.type: mtp without the kernels: MTP %+v, want on (the configuration wins)", p.MTP)
	}
}
