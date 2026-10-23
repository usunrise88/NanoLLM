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
		VerifyRows: ov.Speculative.VerifyRows, NoRepack: ov.Repack == "off",
	}
	// MTP on the CPU: on by default where the device lists the multi-row Q4_0
	// kernels, off elsewhere with the reason in the plan (plan.DefaultCPUMTP).
	// speculative.type: mtp turns it on anywhere; none turns it off.
	if ov.Speculative.Type == "" && req.Device.Type == backend.CPU {
		r.DefaultCPUMTP(req.Device.Has(backend.FeatureMultiRowQ4_0))
	}
	// The bound on the rows spent on drafts follows the kernels, not how MTP was
	// asked for: speculative.type: mtp on a CPU with them drafts while one
	// sequence decodes, too (speculative.verify_rows overrides).
	if ov.Speculative.Type == config.SpecMTP && r.VerifyRows == 0 && req.Device.Type == backend.CPU && req.Device.Has(backend.FeatureMultiRowQ4_0) {
		r.VerifyRows = 1 + cmp.Or(r.DraftMax, plan.DefaultDraftMax)
	}
	if ov.Speculative.Type == config.SpecNGram {
		r.NoMTP, r.NoMTPReason = true, "prompt lookup (speculative.type: ngram) drafts instead"
		r.NGram = cmp.Or(ov.Speculative.Match, ngram.DefaultN)
		r.DraftMax = cmp.Or(ov.Speculative.NMax, DefaultNGramDraftMax)
	}
	if req.Device.Type != backend.CPU {
		r.FlashAttnSmallBatch = req.Device.Has(backend.FeatureFlashAttnSmallBatch)
		r.BackendSampling = req.Device.Has(backend.FeatureBackendSampling)
		r.Device = &plan.Device{
			Selector: req.Device.Selector(), Backend: plan.Kind(req.Device.ID),
			Memory: plan.DeviceMemory(req.Device), Unified: req.Device.Unified,
		}
	}
	switch ov.FlashAttention {
	case "on":
		r.FlashAttnSmallBatch = false
	case "off":
		r.NoFlashAttention = true
	case "small":
		r.FlashAttnSmallBatch = true
	}
	if len(ov.CudaNano) > 0 && req.Device.Has(backend.FeatureCudaNano) {
		r.CudaNano = make(map[string]int, len(ov.CudaNano))
		for name, v := range ov.CudaNano {
			r.CudaNano[strings.ToUpper(name)] = v
		}
	}
	switch ov.Sampling {
	case "host":
		r.BackendSampling = false
	case "device":
		r.BackendSampling = true
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
		spec.Stream.Method, spec.Stream.Depth = ov.Stream.Method, ov.Stream.Depth
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
