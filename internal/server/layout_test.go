package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/pool"
	"github.com/usunrise88/nanollm/internal/registry"
)

// OPT-7 in the server: models.overrides.<id>.layout: nano loads the relaid
// copy and plans from its own offsets.

// nanoOfCatalog is E4B's inventory laid out by the nano policy, as the store
// would hand it out for a relaid copy at path.
func nanoOfCatalog(t *testing.T, m registry.Manifest, path string) registry.Relaid {
	t.Helper()
	orig, ok := m.FileInventory(registry.RoleModel)
	if !ok {
		t.Fatal("the catalog model has no inventory")
	}
	l, err := gguf.PlanInventory(orig, gguf.DefaultRelayoutAlignment)
	if err != nil {
		t.Fatal(err)
	}
	inv := l.Inventory(orig)
	inv.Role = registry.RoleModel
	return registry.Relaid{Path: path, Inventory: inv}
}

func TestPlannerPlacerPlansFromTheRelaidInventory(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("M8 is not proven on macOS (range-lock), so the planner refuses ram-2g there")
	}
	m := defaultModel(t)
	req := cpuRequest(m)
	req.Override.Preset, req.Override.Mode = "ram-2g", "M8"
	orig, err := PlannerPlacer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}

	rl := nanoOfCatalog(t, m, "/models/model.nano.gguf")
	req.ModelPath, req.Manifest, req.Layout = rl.Path, m.WithRelaid(rl), registry.LayoutNano
	nano, err := PlannerPlacer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	po, pn := orig.Plan.(*plan.Plan), nano.Plan.(*plan.Plan)
	if po.Layout != "" || pn.Layout != gguf.LayoutNano {
		t.Errorf("plan layouts %q and %q", po.Layout, pn.Layout)
	}
	// The loader reads the copy, with the copy's ranges: a layer is one.
	if nano.Spec.Path != rl.Path || orig.Spec.Path == rl.Path {
		t.Errorf("paths %q and %q", orig.Spec.Path, nano.Spec.Path)
	}
	if nano.Spec.Stream == nil || len(nano.Spec.Stream.Ranges) != 42 || len(orig.Spec.Stream.Ranges) != 126 {
		t.Errorf("stream ranges: %d, then %d", len(orig.Spec.Stream.Ranges), len(nano.Spec.Stream.Ranges))
	}
	if len(pn.LockRanges) != 1 || len(po.LockRanges) <= 1 {
		t.Errorf("lock ranges: %d, then %d", len(po.LockRanges), len(pn.LockRanges))
	}
	// The same weights in another order are another instance of the pool.
	if orig.Key.Layout != "" || nano.Key.Layout != registry.LayoutNano || orig.Key == nano.Key ||
		!strings.Contains(nano.Key.String(), "layout=nano") || strings.Contains(orig.Key.String(), "layout") {
		t.Errorf("keys %v and %v", orig.Key, nano.Key)
	}
	if orig.Key.Model != nano.Key.Model || orig.Key.Revision != nano.Key.Revision || orig.Key.Mode != nano.Key.Mode {
		t.Errorf("the keys differ in more than the layout: %+v, %+v", orig.Key, nano.Key)
	}
}

// asLlama makes the fake backend answer as llama.cpp, so the runtime takes the
// path of a model with files.
type asLlama struct{ backend.Backend }

func (asLlama) Name() string { return config.BackendLlamaCpp }

func (a asLlama) Devices(ctx context.Context) ([]backend.Device, error) {
	devs, err := a.Backend.Devices(ctx)
	for i := range devs {
		devs[i].Backend = config.BackendLlamaCpp
	}
	return devs, err
}

func TestRuntimePlacementResolvesTheLayout(t *testing.T) {
	// The toy gemma4 as a pinned catalog entry in a store.
	b, err := gguf.TinyGemma4().FillPattern().Bytes()
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	m := registry.Manifest{ID: "tiny", Arch: "gemma4", Files: []registry.File{{
		Role: registry.RoleModel, Repo: "x/tiny", Path: "tiny.gguf", Revision: strings.Repeat("a", 40),
		Size: int64(len(b)), SHA256: hex.EncodeToString(sum[:]),
	}}}
	store := &registry.Store{Root: t.TempDir()}
	src, err := store.FilePath(m, registry.RoleModel)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(src), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(src, b, 0o644); err != nil {
		t.Fatal(err)
	}
	inv, _, err := registry.InspectFile(src, registry.RoleModel)
	if err != nil {
		t.Fatal(err)
	}
	m.Inventory = []gguf.Inventory{inv}
	if err := store.Pull(context.Background(), m, nil); err != nil { // verifies the copy in place
		t.Fatal(err)
	}

	var got PlaceRequest
	cfg := config.Default()
	cfg.Models.Default, cfg.Models.Preload = "tiny", nil
	cfg.Devices.Default = config.DeviceCPU
	cfg.Models.Overrides = map[string]config.ModelOverride{"tiny": {Layout: config.LayoutNano}}
	rt := New(context.Background(), Options{
		Config: cfg, Catalog: []registry.Manifest{m}, Store: store,
		Backends: []backend.Backend{asLlama{fake.New(fake.Options{NoGPU: true})}},
		Place: func(_ context.Context, req PlaceRequest) (Placement, error) {
			got = req
			return Placement{Key: pool.Key{Model: req.Manifest.ID, Backend: req.Backend, Layout: req.Layout}}, nil
		},
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	t.Cleanup(rt.Close)
	ctx := context.Background()

	// No copy: a clear refusal that says what to run, and nothing is planned.
	_, err = rt.placement(ctx, m, "")
	if err == nil || !strings.Contains(err.Error(), "`nanollm models relayout tiny`") || got.Manifest.ID != "" {
		t.Fatalf("nano without the copy: %v (planned %q)", err, got.Manifest.ID)
	}
	if code(err) != core.CodeModelUnavailable {
		t.Errorf("code %v for %v", code(err), err)
	}

	// With the copy: its path, its inventory, its own pool key.
	if _, err := gguf.Relayout(ctx, src, gguf.NanoPath(src), gguf.RelayoutOptions{SourceSHA256: m.Files[0].SHA256}); err != nil {
		t.Fatal(err)
	}
	pl, err := rt.placement(ctx, m, "")
	if err != nil {
		t.Fatal(err)
	}
	fi, _ := got.Manifest.FileInventory(registry.RoleModel)
	if got.ModelPath != gguf.NanoPath(src) || got.Layout != registry.LayoutNano || fi.Layout != gguf.LayoutNano ||
		fi.Alignment != gguf.DefaultRelayoutAlignment || pl.Key.Layout != registry.LayoutNano {
		t.Errorf("request: path %s layout %q inventory %q/%d, key %+v", got.ModelPath, got.Layout, fi.Layout, fi.Alignment, pl.Key)
	}
	if len(got.Manifest.Inventory) != 1 || m.Inventory[0].Layout != "" {
		t.Error("the catalog's inventory was replaced instead of the request's")
	}

	// original, or nothing, is the file as published.
	for _, layout := range []string{"", config.LayoutOriginal} {
		rt.cfg.Models.Overrides = map[string]config.ModelOverride{"tiny": {Layout: layout}}
		if _, err := rt.placement(ctx, m, ""); err != nil {
			t.Fatal(err)
		}
		if got.ModelPath != src || got.Layout != "" || got.Manifest.Inventory[0].Layout != "" {
			t.Errorf("layout %q: path %s, request layout %q", layout, got.ModelPath, got.Layout)
		}
	}
}
