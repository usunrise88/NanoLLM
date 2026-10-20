package server

import (
	"cmp"
	"context"
	"fmt"
	"runtime"
	"strings"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/ngram"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

// PlannerPlacer places a model with the planner, from the catalog entry's
// measured inventory. The preset is models.overrides.<id>.preset; without
// one it is derived from what the chosen device has free, so a default
// installation uses the machine it is on without being told its size.
// CPUMTPOffReason is the plan's reason for the default of MTP on the CPU.
const CPUMTPOffReason = "off by default on the CPU: without the multi-row Q4_0 kernels the measured speed-up is at best 1.04x on a 16-core EPYC 7351P (threshold 1.2x, docs/STAGE2.md 2.7); models.<id>.speculative.type: mtp turns it on"

// The MTP setting for a CPU with the multi-row kernels (docs/AUTOTUNE.md, "MTP на CPU
// с ядрами OPT-16"): 1.36-1.41x on 16 threads and 1.21-1.25x on 4 cores for draft_max 2 and
// p_min 0.5-0.75; 0.75 is the best of the worst.
const (
	CPUMTPDraftMax  = 2
	CPUMTPDraftPMin = 0.75
)

// DefaultNGramDraftMax is how many tokens a prompt-lookup draft carries when
// the configuration does not say: provisional until the sweep of docs/STAGE2.md
// 2.7 sets it.
const DefaultNGramDraftMax = 4

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
		NoRepack: ov.Repack == "off",
	}
	// MTP on the CPU pays where checking a draft is cheap. The stage 2 sweep
	// (docs/STAGE2.md 2.7, docs/AUTOTUNE.md) found at best 1.04x with the
	// kernels that check 2-3 rows one by one, and 1.41x (16 threads) / 1.25x
	// (4 cores) with the multi-row Q4_0 kernels (threshold 1.2x). So it is on by
	// default where the device lists them, with the setting that passes on both
	// profiles, and off elsewhere with the reason in the plan.
	// speculative.type: mtp turns it on anywhere; none turns it off.
	if ov.Speculative.Type == "" && req.Device.Type == backend.CPU {
		if req.Device.Has(backend.FeatureMultiRowQ4_0) {
			if r.DraftMax == 0 {
				r.DraftMax = CPUMTPDraftMax
			}
			if r.DraftPMin == 0 {
				r.DraftPMin = CPUMTPDraftPMin
			}
		} else {
			r.NoMTP, r.NoMTPReason = true, CPUMTPOffReason
		}
	}
	if ov.Speculative.Type == config.SpecNGram {
		r.NoMTP, r.NoMTPReason = true, "prompt lookup (speculative.type: ngram) drafts instead"
		r.NGram = cmp.Or(ov.Speculative.Match, ngram.DefaultN)
		r.DraftMax = cmp.Or(ov.Speculative.NMax, DefaultNGramDraftMax)
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
	if spec.Stream != nil {
		spec.Stream.Ahead, spec.Stream.Keep, spec.Stream.Evict = ov.Stream.Ahead, ov.Stream.Keep, ov.Stream.Evict
	}
	switch ov.Repack {
	case "off":
		spec.Repack = false
	case "on":
		spec.Repack = true // an expert setting: the budget of the modes that keep weights in the file is not kept
	}
	spec.Threads, spec.ThreadsBatch = req.Threads, req.ThreadsBatch
	spec = spec.WithDefaultThreads()
	f, _ := req.Manifest.File(registry.RoleModel)
	key := planKey(req, f.Revision, string(p.Mode), preset.Name, p.NCtx)
	return Placement{Key: key, Spec: spec, Plan: p}, nil
}
