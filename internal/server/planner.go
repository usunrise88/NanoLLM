package server

import (
	"context"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/placement"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

// The CPU default of MTP is in package plan, shared with the benchmark; these
// are its names here.
const (
	CPUMTPOffReason = plan.CPUMTPOffReason
	CPUMTPDraftMax  = plan.CPUMTPDraftMax
	CPUMTPDraftPMin = plan.CPUMTPDraftPMin
)

// DefaultNGramDraftMax is how many tokens a prompt-lookup draft carries when
// the configuration does not say: provisional until the sweep of docs/STAGE2.md
// 2.7 sets it.
const DefaultNGramDraftMax = placement.DefaultNGramDraftMax

// PlannerPlacer places a model with the planner, from the catalog entry's
// measured inventory. The preset is models.overrides.<id>.preset; without
// one it is derived from what the chosen device has free, so a default
// installation uses the machine it is on without being told its size. The
// work is package placement's, which the autotuner shares.
func PlannerPlacer(ctx context.Context, req PlaceRequest) (Placement, error) {
	if req.Backend == config.BackendFake {
		return DefaultPlacer(ctx, req)
	}
	res, err := placement.Resolve(placement.Input{
		Manifest: req.Manifest, Device: req.Device, Override: req.Override,
		ModelPath: req.ModelPath, DrafterPath: req.DrafterPath,
		Threads: req.Threads, ThreadsBatch: req.ThreadsBatch, Tune: req.Tune,
	})
	if err != nil {
		return Placement{}, err
	}
	f, _ := req.Manifest.File(registry.RoleModel)
	key := planKey(req, f.Revision, string(res.Plan.Mode), res.Preset.Name, res.Plan.NCtx)
	return Placement{Key: key, Spec: res.Spec, Plan: res.Plan}, nil
}
