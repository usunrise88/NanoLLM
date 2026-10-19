// Package server assembles a running NanoLLM out of its parts: the catalog
// and the model store, the opened backends and their devices, the placement
// of a model on a device, the pool and the engine. It implements the engine's
// Resolver and core.Admin, which are where those parts meet; cmd/nanollm only
// wires configuration into it and mounts the HTTP surface.
package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/engine"
	"github.com/usunrise88/nanollm/internal/memlock"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/pool"
	"github.com/usunrise88/nanollm/internal/registry"
)

// Placement is how one model loads on one device: the pool key, the resolved
// load, and the plan behind it for reporting.
type Placement struct {
	Key  pool.Key
	Spec backend.LoadSpec
	Plan any
}

// PlaceRequest is what a Placer decides from.
type PlaceRequest struct {
	Manifest registry.Manifest
	Backend  string
	Device   backend.Device
	Override config.ModelOverride
	// ModelPath and DrafterPath are the files on disk; empty for a backend
	// that needs none (fake).
	ModelPath    string
	DrafterPath  string
	Threads      int
	ThreadsBatch int
}

// Placer turns a model and a device into a placement: the planner's job.
type Placer func(ctx context.Context, req PlaceRequest) (Placement, error)

// Runtime is the running server minus HTTP.
type Runtime struct {
	cfg      config.Config
	catalog  []registry.Manifest
	store    *registry.Store
	backends []backend.Backend
	place    Placer
	log      *slog.Logger

	Pool   *pool.Pool
	Engine *engine.Service

	mu      sync.Mutex
	devices []backend.Device // probed once; Devices refreshes
	specs   map[pool.Key]backend.LoadSpec
	plans   map[pool.Key]any
	byName  map[string]backend.Backend
	pinned  map[string]bool
}

// Options are the runtime's parts.
type Options struct {
	Config   config.Config
	Catalog  []registry.Manifest
	Store    *registry.Store
	Backends []backend.Backend
	Place    Placer
	Log      *slog.Logger
}

// New assembles a runtime. It probes devices once; a backend that cannot list
// them is logged and serves nothing.
func New(ctx context.Context, o Options) *Runtime {
	if o.Log == nil {
		o.Log = slog.Default()
	}
	if o.Place == nil {
		o.Place = PlannerPlacer
	}
	r := &Runtime{
		cfg: o.Config, catalog: o.Catalog, store: o.Store, backends: o.Backends, place: o.Place, log: o.Log,
		specs: map[pool.Key]backend.LoadSpec{}, plans: map[pool.Key]any{}, byName: map[string]backend.Backend{}, pinned: map[string]bool{},
	}
	for _, b := range o.Backends {
		r.byName[b.Name()] = b
	}
	r.devices = r.probe(ctx)
	r.Pool = pool.New(r.load, o.Config.Pool.MaxResidentModels)
	r.Pool.MaxPending = o.Config.Pool.MaxPending
	// A pin is the model's, whichever load makes the instance: the admin's,
	// a request's, or a reload's.
	r.Pool.Pinned = func(k pool.Key) bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		return r.pinned[k.Model]
	}
	r.Engine = engine.New(engine.Config{
		DefaultModel:    o.Config.Models.Default,
		EnableThinking:  o.Config.Generation.EnableThinking,
		MaxTokensCap:    o.Config.Generation.MaxTokensCap,
		ContextOverflow: o.Config.Generation.ContextOverflow,
		PromptCache:     o.Config.Cache.Prompt,
	}, r.Pool, r.Resolve)
	return r
}

func (r *Runtime) probe(ctx context.Context) []backend.Device {
	var out []backend.Device
	for _, b := range r.backends {
		devs, err := b.Devices(ctx)
		if err != nil {
			r.log.Warn("cannot list devices", "backend", b.Name(), "err", err)
			continue
		}
		out = append(out, devs...)
	}
	return out
}

// Close unloads every model.
func (r *Runtime) Close() { r.Pool.Close() }

func (r *Runtime) load(ctx context.Context, k pool.Key) (backend.Model, any, error) {
	r.mu.Lock()
	spec, ok := r.specs[k]
	b := r.byName[k.Backend]
	r.mu.Unlock()
	if !ok || b == nil {
		return nil, nil, fmt.Errorf("no placement recorded for %s", k)
	}
	r.mu.Lock()
	info := r.plans[k]
	r.mu.Unlock()
	r.log.Info("loading model", "key", k.String(), "gpu_layers", spec.GPULayers, "n_ctx", spec.NCtx, "slots", spec.Slots)
	// Range-lock (M7, M8) pins the planned ranges before the engine maps
	// the file, so the first inference already reads locked pages.
	var pins []*memlock.Pin
	if p, ok := info.(*plan.Plan); ok {
		for role, path := range map[string]string{registry.RoleModel: spec.Path, registry.RoleDrafter: spec.DrafterPath} {
			var ranges []memlock.Range
			for _, lr := range p.LockRanges {
				if lr.Role == role {
					ranges = append(ranges, memlock.Range{Off: lr.Offset, Len: lr.Length})
				}
			}
			if len(ranges) == 0 || path == "" {
				continue
			}
			pin, err := memlock.PinFile(path, ranges)
			if err != nil {
				for _, p := range pins {
					_ = p.Close()
				}
				return nil, nil, fmt.Errorf("range-lock %s: %w", role, err)
			}
			r.log.Info("range-lock", "role", role, "locked_mib", pin.Locked()>>20)
			pins = append(pins, pin)
		}
	}
	m, err := b.Load(ctx, spec)
	if err != nil {
		for _, p := range pins {
			_ = p.Close()
		}
		return nil, nil, err
	}
	if info == nil {
		info = spec
	}
	if len(pins) > 0 {
		m = pinnedModel{Model: m, pins: pins}
	}
	return m, info, nil
}

// pinnedModel releases range-lock when the model is closed: the pin lives
// exactly as long as the model (docs/STAGE1.md 1.6).
type pinnedModel struct {
	backend.Model
	pins []*memlock.Pin
}

func (m pinnedModel) Close() error {
	err := m.Model.Close()
	for _, p := range m.pins {
		_ = p.Close()
	}
	return err
}

func (r *Runtime) manifest(id string) (registry.Manifest, error) {
	m, ok := registry.Find(r.catalog, id)
	if !ok {
		return registry.Manifest{}, core.Errorf(core.CodeModelNotFound, "no model %q in the catalog", id).WithParam("model")
	}
	return m, nil
}

// pickDevice resolves a selector: auto (the first GPU, else the CPU), a class
// (cpu, gpu, npu), or backend:id. Unless fallback is off, a selector that
// matches nothing falls back to auto.
func (r *Runtime) pickDevice(selector string) (backend.Device, error) {
	if selector == "" {
		selector = r.cfg.Devices.Default
	}
	r.mu.Lock()
	devs := append([]backend.Device(nil), r.devices...)
	r.mu.Unlock()
	find := func(sel string) (backend.Device, bool) {
		switch sel {
		case "", config.DeviceAuto:
			for _, d := range devs {
				if d.Type == backend.GPU {
					return d, true
				}
			}
			for _, d := range devs {
				if d.Type == backend.CPU {
					return d, true
				}
			}
		case config.DeviceCPU, config.DeviceGPU, config.DeviceNPU:
			for _, d := range devs {
				if string(d.Type) == sel {
					return d, true
				}
			}
		default:
			for _, d := range devs {
				if d.Selector() == sel {
					return d, true
				}
			}
		}
		return backend.Device{}, false
	}
	if d, ok := find(selector); ok {
		return d, nil
	}
	if r.cfg.Devices.Fallback {
		if d, ok := find(config.DeviceAuto); ok {
			return d, nil
		}
	}
	if len(devs) == 0 {
		return backend.Device{}, core.Errorf(core.CodeModelUnavailable, "no backend could be opened; see `nanollm devices`")
	}
	return backend.Device{}, core.Errorf(core.CodeDeviceUnavailable, "no device matches %q", selector).WithParam("device")
}

// Resolve is the engine's Resolver.
func (r *Runtime) Resolve(ctx context.Context, model, device string) (engine.Target, error) {
	m, err := r.manifest(model)
	if err != nil {
		return engine.Target{}, err
	}
	pl, err := r.placement(ctx, m, device)
	if err != nil {
		return engine.Target{}, err
	}
	return engine.Target{Key: pl.Key, Manifest: m}, nil
}

func (r *Runtime) placement(ctx context.Context, m registry.Manifest, device string) (Placement, error) {
	dev, err := r.pickDevice(device)
	if err != nil {
		return Placement{}, err
	}
	req := PlaceRequest{Manifest: m, Backend: dev.Backend, Device: dev, Override: r.cfg.Models.Overrides[m.ID], Threads: r.cfg.Devices.CPUThreads, ThreadsBatch: r.cfg.Devices.CPUThreadsBatch}
	if dev.Backend != config.BackendFake {
		if r.store == nil {
			return Placement{}, core.Errorf(core.CodeModelUnavailable, "no model store is configured")
		}
		req.ModelPath, req.DrafterPath, err = r.store.Ready(m)
		if err != nil {
			return Placement{}, core.Errorf(core.CodeModelUnavailable, "%v", err).WithCause(err)
		}
	}
	pl, err := r.place(ctx, req)
	if err != nil {
		var ce *core.Error
		if errors.As(err, &ce) {
			return Placement{}, err
		}
		return Placement{}, core.Errorf(core.CodeModelUnavailable, "model %s does not fit %s: %v", m.ID, dev.Selector(), err).WithCause(err)
	}
	r.mu.Lock()
	r.specs[pl.Key] = pl.Spec
	r.plans[pl.Key] = pl.Plan
	r.mu.Unlock()
	return pl, nil
}

// DefaultPlacer is the placement without the planner, for the fake backend
// and for models without a measured inventory: the configured context and
// slots (4096 and one by default), everything on the chosen device, weights
// mapped, KV f16, the drafter when the entry has one.
func DefaultPlacer(_ context.Context, req PlaceRequest) (Placement, error) {
	ov := req.Override
	nctx, slots := 4096, 1
	if !ov.NCtx.IsAuto() {
		nctx = ov.NCtx.N
	}
	if !ov.Slots.IsAuto() {
		slots = ov.Slots.N
	}
	kv := ov.KVType
	if kv == "" {
		kv = config.KVF16
	}
	spec := backend.LoadSpec{
		Path: req.ModelPath, Device: req.Device.Selector(), NCtx: nctx, Slots: slots,
		NBatch: 512, NUBatch: 512, Threads: req.Threads, ThreadsBatch: req.ThreadsBatch, KVTypeK: kv, KVTypeV: kv,
		FlashAttention: true, LoadMode: backend.LoadMMap, LazyMode: backend.LazyOff, SWAFull: true,
	}
	if req.Device.Type == backend.GPU {
		spec.GPULayers = -1
	}
	spec = spec.WithDefaultThreads()
	if req.DrafterPath != "" || (req.Backend == config.BackendFake && req.Manifest.Drafter != nil) {
		if ov.Speculative.Type != config.SpecNone {
			spec.DrafterPath = req.DrafterPath
			if spec.DrafterPath == "" {
				spec.DrafterPath = "fake-drafter"
			}
			spec.DraftMax, spec.DraftPMin = ov.Speculative.NMax, ov.Speculative.PMin
			if spec.DraftMax == 0 {
				spec.DraftMax = 3
			}
		}
	}
	f, _ := req.Manifest.File(registry.RoleModel)
	return Placement{Key: planKey(req, f.Revision, ov.Mode, ov.Preset, nctx), Spec: spec, Plan: spec}, nil
}

func planKey(req PlaceRequest, revision, mode, preset string, nctx int) pool.Key {
	return pool.Key{
		Model: req.Manifest.ID, Revision: revision, Backend: req.Backend, Device: req.Device.ID,
		Mode: mode, Preset: preset, NCtx: nctx,
	}
}

// --- core.Admin ---

var _ core.Admin = (*Runtime)(nil)

func (r *Runtime) Models(ctx context.Context) ([]core.ModelInfo, error) {
	resident := r.Pool.Status()
	var out []core.ModelInfo
	for _, m := range r.catalog {
		info := core.ModelInfo{ID: m.ID, Family: m.Family, Params: m.Params, Quant: m.Quant, Default: m.ID == r.cfg.Models.Default, Disk: registry.StateAbsent}
		for _, f := range m.Files {
			info.SizeBytes += f.Size
		}
		if r.store != nil {
			if st, err := r.store.Status(m); err == nil {
				info.Disk = st.State
			}
		}
		for _, s := range resident {
			if s.Key.Model == m.ID {
				info.Loaded = append(info.Loaded, s)
				info.Pinned = info.Pinned || s.Pinned
			}
		}
		out = append(out, info)
	}
	return out, nil
}

func (r *Runtime) Download(ctx context.Context, id string, progress func(core.DownloadProgress)) error {
	m, err := r.manifest(id)
	if err != nil {
		return err
	}
	if r.store == nil {
		return core.Errorf(core.CodeModelUnavailable, "no model store is configured")
	}
	err = r.store.Pull(ctx, m, func(p registry.Progress) {
		if progress != nil {
			progress(core.DownloadProgress(p))
		}
	})
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return err
		}
		return core.Errorf(core.CodeModelUnavailable, "download %s: %v", id, err).WithCause(err)
	}
	return nil
}

func (r *Runtime) info(ctx context.Context, id string) (*core.ModelInfo, error) {
	all, err := r.Models(ctx)
	if err != nil {
		return nil, err
	}
	for _, m := range all {
		if m.ID == id {
			return &m, nil
		}
	}
	return nil, core.Errorf(core.CodeModelNotFound, "no model %q", id)
}

// Load makes the model resident on the default device. It stays loaded after
// the call: an idle instance is only evicted for another.
func (r *Runtime) Load(ctx context.Context, id string) (*core.ModelInfo, error) {
	t, err := r.Resolve(ctx, id, "")
	if err != nil {
		return nil, err
	}
	_, release, err := r.Pool.Acquire(ctx, t.Key)
	if err != nil {
		return nil, core.Errorf(core.CodeModelUnavailable, "load %s: %v", id, err).WithCause(err)
	}
	release()
	r.mu.Lock()
	pin := r.pinned[id]
	r.mu.Unlock()
	if pin {
		r.Pool.Pin(t.Key, true)
	}
	return r.info(ctx, id)
}

func (r *Runtime) keys(id string) []pool.Key {
	var out []pool.Key
	for _, s := range r.Pool.Status() {
		if s.Key.Model == id {
			out = append(out, s.Key)
		}
	}
	return out
}

func (r *Runtime) Unload(ctx context.Context, id string) error {
	if _, err := r.manifest(id); err != nil {
		return err
	}
	keys := r.keys(id)
	if len(keys) == 0 {
		return core.Errorf(core.CodeInvalidRequest, "model %s is not loaded", id)
	}
	for _, k := range keys {
		_ = r.Pool.Unload(k)
	}
	return nil
}

// Pin keeps the model's instances resident, including ones loaded later.
func (r *Runtime) Pin(ctx context.Context, id string, pin bool) error {
	if _, err := r.manifest(id); err != nil {
		return err
	}
	r.mu.Lock()
	r.pinned[id] = pin
	r.mu.Unlock()
	for _, k := range r.keys(id) {
		r.Pool.Pin(k, pin)
	}
	return nil
}

// Reload loads the model afresh under its current placement and retires the
// old instances: the hot swap after a catalog update or a placement change.
func (r *Runtime) Reload(ctx context.Context, id string) (*core.ModelInfo, error) {
	t, err := r.Resolve(ctx, id, "")
	if err != nil {
		return nil, err
	}
	// A key still resident would just be reused; unload it first if idle so
	// the reload really reads the files again.
	for _, k := range r.keys(id) {
		if k == t.Key {
			_ = r.Pool.Unload(k)
		}
	}
	_, release, err := r.Pool.Acquire(ctx, t.Key)
	if err != nil {
		return nil, core.Errorf(core.CodeModelUnavailable, "reload %s: %v", id, err).WithCause(err)
	}
	release()
	r.Pool.Retire(id, t.Key)
	r.mu.Lock()
	pin := r.pinned[id]
	r.mu.Unlock()
	r.Pool.Pin(t.Key, pin)
	return r.info(ctx, id)
}

func (r *Runtime) Devices(ctx context.Context) ([]core.DeviceInfo, error) {
	devs := r.probe(ctx)
	r.mu.Lock()
	r.devices = devs
	r.mu.Unlock()
	out := make([]core.DeviceInfo, 0, len(devs))
	for _, d := range devs {
		info := core.DeviceInfo{
			Selector: d.Selector(), Backend: d.Backend, ID: d.ID, Type: string(d.Type), Name: d.Name,
			MemoryTotal: d.MemoryTotal, MemoryFree: d.MemoryFree, Unified: d.Unified,
		}
		for _, m := range r.catalog {
			for _, v := range m.Verified {
				if strings.HasPrefix(v, d.Backend+"/") || strings.HasPrefix(v, d.Selector()) {
					info.Verified = append(info.Verified, m.ID)
					break
				}
			}
		}
		out = append(out, info)
	}
	return out, nil
}

// Ready reports whether the default model is resident, for /readyz.
func (r *Runtime) Ready() (bool, string) {
	for _, s := range r.Pool.Status() {
		if s.Key.Model == r.cfg.Models.Default && !s.Retired {
			return true, ""
		}
	}
	return false, "the default model is not loaded"
}

// Preload loads models.preload in the background and pins nothing; a failure
// is logged, and the server keeps serving whatever else it can.
func (r *Runtime) Preload(ctx context.Context) {
	for _, id := range r.cfg.Models.Preload {
		if _, err := r.Load(ctx, id); err != nil {
			r.log.Warn("preload failed", "model", id, "err", err)
			continue
		}
		r.log.Info("model loaded", "model", id)
	}
}
