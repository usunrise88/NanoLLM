package server

import (
	"context"
	"fmt"
	"runtime"
	"strings"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

// PlannerPlacer places a model with the planner, from the catalog entry's
// measured inventory. The preset is models.overrides.<id>.preset; without
// one it is derived from what the chosen device has free, so a default
// installation uses the machine it is on without being told its size.
// CPUMTPOffReason is the plan's reason for the default of MTP on the CPU.
const CPUMTPOffReason = "off by default on the CPU: the measured speed-up is at best 1.04x on a 16-core EPYC 7351P (threshold 1.2x, docs/STAGE2.md 2.7); models.<id>.speculative.type: mtp turns it on"

func PlannerPlacer(ctx context.Context, req PlaceRequest) (Placement, error) {
	if req.Backend == config.BackendFake {
		return DefaultPlacer(ctx, req)
	}
	model, err := plan.ModelFrom(req.Manifest.ID, req.Manifest.Inventory)
	if err != nil {
		return Placement{}, err
	}
	ov := req.Override
	var preset plan.Preset
	if ov.Preset != "" {
		if preset, err = plan.LookupPreset(ov.Preset); err != nil {
			return Placement{}, err
		}
	} else if preset, err = plan.DerivedPreset(req.Device); err != nil {
		return Placement{}, err
	}
	mode, err := plan.ParseMode(ov.Mode)
	if err != nil {
		return Placement{}, err
	}
	r := plan.Request{
		Model: model, Preset: preset, Mode: mode, KVType: ov.KVType, Proven: plan.ProvenFor(runtime.GOOS),
		NCtx: ov.NCtx.N, Slots: ov.Slots.N,
		NoMTP: ov.Speculative.Type == config.SpecNone, DraftMax: ov.Speculative.NMax, DraftPMin: ov.Speculative.PMin,
	}
	// MTP is off by default on the CPU: the stage 2 sweep (docs/STAGE2.md 2.7)
	// found at best 1.04x on a 16-core EPYC (threshold 1.2x), and the drafter
	// costs memory. speculative.type: mtp turns it on.
	if ov.Speculative.Type == "" && req.Device.Type == backend.CPU {
		r.NoMTP, r.NoMTPReason = true, CPUMTPOffReason
	}
	if req.Device.Type != backend.CPU {
		r.Device = &plan.Device{
			Selector: req.Device.Selector(), Backend: strings.TrimRight(req.Device.ID, "0123456789"),
			Memory: plan.DeviceMemory(req.Device), Unified: req.Device.Unified,
		}
	}
	p, err := plan.Make(r)
	if err != nil {
		return Placement{}, err
	}
	if !p.OK() {
		return Placement{}, fmt.Errorf("preset %s: %s", preset.Name, p.Refused)
	}
	if req.DrafterPath == "" {
		p.MTP.Enabled = false
	}
	spec, err := p.LoadSpec(req.ModelPath, req.DrafterPath)
	if err != nil {
		return Placement{}, err
	}
	if p.Device == nil {
		spec.Device = req.Device.Selector()
	}
	spec.Threads, spec.ThreadsBatch = req.Threads, req.ThreadsBatch
	spec = spec.WithDefaultThreads()
	f, _ := req.Manifest.File(registry.RoleModel)
	key := planKey(req, f.Revision, string(p.Mode), preset.Name, p.NCtx)
	return Placement{Key: key, Spec: spec, Plan: p}, nil
}
