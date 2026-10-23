package server

import (
	"context"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/tune"
)

func tuned(mode tune.Mode, cores int, mutate func(*tune.Active)) *tune.Active {
	k, _ := tune.ConstraintsFor(mode, tune.Cores{Physical: cores, Logical: cores * 2}, 20<<30)
	a := &tune.Active{Mode: mode, Constraints: k}
	if mutate != nil {
		mutate(a)
	}
	return a
}

func placed(t *testing.T, req PlaceRequest, a *tune.Active) Placement {
	t.Helper()
	req.Tune, req.TuneID = a.PlaceTune(req.Device.Selector()), a.ID()
	pl, err := PlannerPlacer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	return pl
}

func cudaRequest(t *testing.T) PlaceRequest {
	m := defaultModel(t)
	return PlaceRequest{
		Manifest: m, Backend: config.BackendLlamaCpp,
		Device: backend.Device{Backend: config.BackendLlamaCpp, ID: "cuda0", Type: backend.GPU, Name: "RTX", MemoryTotal: 8 << 30, MemoryFree: 7 << 30,
			Features: []string{backend.FeatureCudaNano, backend.FeatureBackendSampling}},
		ModelPath: "/models/model.gguf", DrafterPath: "/models/drafter.gguf",
	}
}

// The three modes give three placements of one model on one machine, and three
// pool keys, so that switching the mode reloads the model.
func TestModesPlaceAndKeyDifferently(t *testing.T) {
	req := cudaRequest(t)
	free := placed(t, req, nil)
	opt := placed(t, req, tuned(tune.Optimal, 14, nil))
	pow := placed(t, req, tuned(tune.Powerful, 14, nil))
	if free.Key.Tune != "" || opt.Key.Tune != "optimal" || pow.Key.Tune != "powerful" {
		t.Fatalf("keys %q %q %q", free.Key.Tune, opt.Key.Tune, pow.Key.Tune)
	}
	po, pf := opt.Plan.(*plan.Plan), free.Plan.(*plan.Plan)
	if po.Mode == pf.Mode || po.Mode != plan.VRAMPLEDisk {
		t.Fatalf("optimal %s, uncapped %s", po.Mode, pf.Mode)
	}
	if opt.Key == pow.Key || opt.Key == free.Key {
		t.Fatal("two modes share one key")
	}
	if pf.NCtx < po.NCtx {
		t.Fatalf("the uncapped plan has a smaller context (%d) than the capped one (%d)", pf.NCtx, po.NCtx)
	}
}

func TestSilentModeSetsTheCPUTeamAndTheDevice(t *testing.T) {
	a := tuned(tune.Silent, 14, nil)
	if got := a.DefaultDevice("auto"); got != "cpu" {
		t.Fatalf("default device %q", got)
	}
	if got := a.DefaultDevice("llamacpp:cuda0"); got != "" {
		t.Fatalf("an explicit devices.default must win, got %q", got)
	}
	req := cpuRequest(defaultModel(t))
	req.Threads, req.ThreadsBatch = 0, 0
	pl := placed(t, req, a)
	if pl.Spec.Threads != 2 || pl.Spec.ThreadsBatch != 2 {
		t.Fatalf("threads %d/%d", pl.Spec.Threads, pl.Spec.ThreadsBatch)
	}
	if p := pl.Plan.(*plan.Plan); p.NCtx < 4608 || p.Sources["threads"] != "mode" {
		t.Fatalf("ctx %d sources %v", p.NCtx, p.Sources)
	}
}

// A stale or missing profile leaves the server on the limits of the mode, and
// says why.
func TestAStaleProfileFallsBackToTheModeLimits(t *testing.T) {
	stale := tuned(tune.Optimal, 14, func(a *tune.Active) {
		a.Stale, a.Changes = true, []tune.Change{{Field: "driver", Old: "581.57", New: "590.12"}}
	})
	if stale.Profile != nil || stale.ID() != "optimal" {
		t.Fatalf("id %q", stale.ID())
	}
	if msg := stale.Message(); msg == "" || !contains(msg, "581.57 → 590.12") {
		t.Fatalf("message %q", msg)
	}
	if s := stale.Staleness(); !s.Stale || s.Suggest == nil || s.Suggest.Mode != tune.Optimal {
		t.Fatalf("staleness %+v", s)
	}
	missing := tuned(tune.Silent, 4, func(a *tune.Active) { a.Missing = true })
	if missing.Message() == "" || !missing.Staleness().Missing {
		t.Fatalf("missing: %q %+v", missing.Message(), missing.Staleness())
	}
	var off *tune.Active
	if off.ID() != "" || off.Message() != "" || off.Staleness().Stale || off.DefaultDevice("auto") != "" {
		t.Fatal("a nil Active must be inert")
	}
	if !off.PlaceTune("llamacpp:cpu").Zero() {
		t.Fatal("a nil Active contributes something")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
