package plan

import (
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/gguf"
)

func mustPlan(t *testing.T, req Request) *Plan {
	t.Helper()
	p, err := Make(req)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func preset(t *testing.T, name string) Preset {
	t.Helper()
	p, err := ParsePreset(name)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func sumGroups(p *Plan) Bytes {
	var b Bytes
	for _, g := range p.Groups {
		b.add(g)
	}
	return b
}

func tensorBytes(m Model, name string) int64 {
	t, _ := m.Target.Find(name)
	return t.Bytes
}

// TestUMAArithmetic is the unified-memory accounting, checked by arithmetic
// because Metal arrives last (docs/ROADMAP.md, stage 4): one pool, no second
// copy of the tied embedding, and in M3 the PLE table outside the budget.
func TestUMAArithmetic(t *testing.T) {
	model := e4b(t)
	inv := model.Target
	for _, c := range []struct {
		preset string
		mode   Mode
	}{
		{"uma-4g", VRAMPLEDisk},
		{"uma-6g", VRAMFull},
		{"uma-8g", VRAMFull},
	} {
		p := mustPlan(t, Request{Model: model, Preset: preset(t, c.preset)})
		if p.Mode != c.mode {
			t.Errorf("%s: %s, want %s", c.preset, p.Mode, c.mode)
			continue
		}
		if p.Memory.Unified == nil || p.Memory.VRAM != nil || p.Memory.RAM != nil {
			t.Fatalf("%s: unified memory is not one pool: %+v", c.preset, p.Memory)
		}
		sum := sumGroups(p)
		if want := sum.VRAM + sum.RAMLocked + HostOverhead; p.Memory.Unified.Used != want {
			t.Errorf("%s: unified used %d, groups sum to %d", c.preset, p.Memory.Unified.Used, want)
		}
		if p.Memory.Unified.Used > p.Memory.Unified.Budget {
			t.Errorf("%s: over budget", c.preset)
		}
		if g0 := p.Groups[gguf.G0]; g0.VRAM != inv.Groups[gguf.G0] {
			t.Errorf("%s: G0 %d in the pool, the file has %d: tied output counted twice", c.preset, g0.VRAM, inv.Groups[gguf.G0])
		}
		g1 := p.Groups[gguf.G1]
		switch c.mode {
		case VRAMPLEDisk:
			if g1.FileBacked != inv.Groups[gguf.G1] || g1.VRAM != 0 || p.LazyMode != backend.LazyOn {
				t.Errorf("%s: M3 PLE %+v, lazy %s", c.preset, g1, p.LazyMode)
			}
		case VRAMFull:
			if g1.VRAM != inv.Groups[gguf.G1] || len(p.Overrides) != 0 || p.LoadMode != backend.LoadMMap {
				t.Errorf("%s: M1 PLE %+v, overrides %v, load %s", c.preset, g1, p.Overrides, p.LoadMode)
			}
		}
		// Hand arithmetic for the pool: weights + KV + compute + overhead.
		weights := inv.Total() + model.Drafter.Total()
		if c.mode == VRAMPLEDisk {
			weights -= inv.Groups[gguf.G1]
		}
		weights -= tensorBytes(Model{Target: *model.Drafter}, "token_embd.weight") // mapped, never read
		want := weights + p.KV.Bytes + p.Groups[gguf.G7].VRAM + HostOverhead
		if p.Memory.Unified.Used != want {
			t.Errorf("%s: pool %d, hand sum %d", c.preset, p.Memory.Unified.Used, want)
		}
	}
}

// TestTiedOutputOnGPU: the LM head is token_embd loaded a second time into the
// output layer's buffer, so a discrete GPU holds one more copy of G0.
func TestTiedOutputOnGPU(t *testing.T) {
	model := e4b(t)
	g0 := model.Target.Groups[gguf.G0]
	m2 := mustPlan(t, Request{Model: model, Preset: preset(t, "vram-4g"), Mode: VRAMPLEHost})
	if got := m2.Groups[gguf.G0]; got.VRAM != g0 || got.RAMLocked != g0 {
		t.Errorf("M2 G0 %+v: want the input copy in RAM and the head in VRAM", got)
	}
	m1 := mustPlan(t, Request{Model: model, Preset: preset(t, "vram-8g")})
	if m1.Mode != VRAMFull || m1.Groups[gguf.G0].VRAM != 2*g0 {
		t.Errorf("M1 G0 %+v: want both copies in VRAM", m1.Groups[gguf.G0])
	}
	if len(m1.Overrides) != 2 || m1.Overrides[0].Device != "llamacpp:cuda0" {
		t.Errorf("M1 overrides %v", m1.Overrides)
	}
	// On the CPU a mapped head aliases the input copy; a repacked one does not.
	m9 := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-2g"), Mode: MMap})
	if m9.Mode != MMap || m9.Groups[gguf.G0].Total() != g0 {
		t.Errorf("M9 G0 %+v", m9.Groups[gguf.G0])
	}
	m6 := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-8g")})
	if m6.Mode != RAMFull || m6.Groups[gguf.G0].RAMLocked != 2*g0 || !m6.Repack {
		t.Errorf("M6 G0 %+v repack %v", m6.Groups[gguf.G0], m6.Repack)
	}
}

func TestPartialOffload(t *testing.T) {
	model := e4b(t)
	h := model.Target.HParams
	p := mustPlan(t, Request{Model: model, Preset: preset(t, "vram-2g")})
	if p.Mode != VRAMPartial {
		t.Fatalf("vram-2g: %s", p.Mode)
	}
	n := p.LayersOnGPU
	if n <= 0 || n >= h.BlockCount || p.GPULayers != n+1 {
		t.Fatalf("N %d, gpu_layers %d", n, p.GPULayers)
	}
	// N+1 must not fit, or the planner did not take the largest N.
	pr := preset(t, "vram-2g")
	pl := &planner{req: Request{Model: model, Preset: pr}, h: h, kvType: gguf.TypeF16, fa: true,
		dev: SyntheticDevice(pr), proven: DefaultProven, nBatch: DefaultNBatch, nUBatch: DefaultNUBatch,
		draftMax: DefaultDraftMax, mtp: true}
	if tr := pl.evaluate(VRAMPartial, n+1, shape{nCtx: FloorNCtx, slots: 1}); tr.short.zero() {
		t.Errorf("N=%d fits too", n+1)
	}
	// KV lives with its layer: only KV layers at or past n_layer−N are in VRAM.
	var want int64
	for il := h.BlockCount - n; il < h.BlockCount; il++ {
		k, v := KVLayerBytes(h, KVShape{NCtx: p.NCtx, Slots: p.Slots, NUBatch: p.NUBatch, SWAFull: p.KV.SWAFull,
			FlashAttention: true, TypeK: gguf.TypeF16, TypeV: gguf.TypeF16}, il)
		want += k + v
	}
	if got := p.Groups[gguf.G5].VRAM; got != want {
		t.Errorf("KV in VRAM %d, want %d", got, want)
	}
	// The drafter goes with the output layer, whatever N: its layers and its
	// tied head on the device, its unused input copy mapped on the host.
	dEmbd := tensorBytes(Model{Target: *model.Drafter}, "token_embd.weight")
	if g6 := p.Groups[gguf.G6]; g6.VRAM != model.Drafter.Total() || g6.FileBacked != dEmbd {
		t.Errorf("G6 %+v, drafter %d", g6, model.Drafter.Total())
	}
	// A VRAM-only preset leaves PLE on disk (SPEC §15) and does not bound RAM.
	if p.Groups[gguf.G1].FileBacked != model.Target.Groups[gguf.G1] || p.Memory.RAM.Budgeted {
		t.Errorf("G1 %+v, RAM %+v", p.Groups[gguf.G1], p.Memory.RAM)
	}
	if !hasWarning(p, "does not budget host RAM") {
		t.Errorf("no warning about unbudgeted RAM: %v", p.Warnings)
	}
}

func hasWarning(p *Plan, sub string) bool {
	for _, w := range p.Warnings {
		if strings.Contains(w, sub) {
			return true
		}
	}
	return false
}

func TestRangeLockModes(t *testing.T) {
	model := e4b(t)
	all := ProvenModes{RAMPLEDisk: true, RAMHybrid: true}
	m7 := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-4g"), Proven: all, Mode: RAMPLEDisk, NoRepack: true})
	if !m7.OK() {
		t.Fatalf("M7 on ram-4g refused: %s", m7.Refused)
	}
	var locked int64
	for _, r := range m7.LockRanges {
		locked += r.Length
	}
	core := model.Target.Total() - model.Target.Groups[gguf.G1] + model.Drafter.Total()
	// Ranges include the padding between merged tensors.
	if locked < core || locked > core+int64(len(model.Target.Tensors)+len(model.Drafter.Tensors))*32 {
		t.Errorf("M7 locks %d bytes, core is %d", locked, core)
	}
	ple, _ := model.Target.Find("per_layer_token_embd.weight")
	for _, r := range m7.LockRanges {
		if r.Role == "model" && r.Offset < ple.Offset+ple.Bytes && ple.Offset < r.Offset+r.Length {
			t.Errorf("M7 locks the PLE table: %+v", r)
		}
	}
	if m7.Groups[gguf.G1].FileBacked != model.Target.Groups[gguf.G1] || m7.LoadMode != backend.LoadMMap || m7.LazyMode != backend.LazyOn {
		t.Errorf("M7: G1 %+v load %s lazy %s", m7.Groups[gguf.G1], m7.LoadMode, m7.LazyMode)
	}

	m8 := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-2g"), Proven: all})
	if m8.Mode != RAMHybrid {
		t.Fatalf("ram-2g with range-lock: %s", m8.Mode)
	}
	g3 := m8.Groups[gguf.G3]
	if g3.RAMLocked == 0 || g3.FileBacked == 0 || m8.Memory.RAM.Used > m8.Memory.RAM.Budget {
		t.Errorf("M8 G3 %+v, RAM %+v", g3, m8.Memory.RAM)
	}

	// Where range-lock is not proven (macOS), its modes are skipped with
	// the reason.
	p := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-3g"), Proven: ProvenFor("darwin")})
	for _, a := range p.Tried {
		if (a.Mode == RAMPLEDisk || a.Mode == RAMHybrid) && !strings.Contains(a.Skipped, "range-lock") {
			t.Errorf("%s: %+v", a.ID, a)
		}
	}
}

func TestRefusal(t *testing.T) {
	model := e4b(t)
	p := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-0.25g")})
	if p.OK() || !strings.Contains(p.Refused, "short by") {
		t.Fatalf("a 256 MiB budget planned: %+v", p)
	}
	var shortfalls int
	for _, a := range p.Tried {
		if a.Shortfall != nil && a.Shortfall.RAM > 0 {
			shortfalls++
		}
	}
	if shortfalls < 2 {
		t.Errorf("refusal without per-mode shortfall: %+v", p.Tried)
	}
	if _, err := p.LoadSpec("m.gguf", ""); err == nil {
		t.Error("a refused plan converted to a load")
	}
}

func TestRequestErrors(t *testing.T) {
	model := e4b(t)
	for name, req := range map[string]Request{
		"q8_0 V without FA": {Model: model, Preset: preset(t, "ram-8g"), KVType: "q8_0", NoFlashAttention: true},
		"research mode":     {Model: model, Preset: preset(t, "ram-8g"), Mode: Stream},
		"no inventory":      {Model: Model{ID: "x"}, Preset: preset(t, "ram-8g")},
		"q4_0 KV":           {Model: model, Preset: preset(t, "ram-8g"), KVType: "q4_0"},
	} {
		if _, err := Make(req); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

// TestFlashAttnSmallBatch: the request carries through to the plan and the
// load spec, the compute buffer of a fixed shape grows by the KQ matrix, and a
// quantized cache or no flash attention drops it.
func TestFlashAttnSmallBatch(t *testing.T) {
	model := e4b(t)
	base := Request{Model: model, Preset: preset(t, "uma-4g"), NCtx: 8192, Slots: 1}
	small := base
	small.FlashAttnSmallBatch = true
	pf, ps := mustPlan(t, base), mustPlan(t, small)
	if pf.FlashAttnSmallBatch || !ps.FlashAttnSmallBatch || !ps.FlashAttention {
		t.Fatalf("flash attention %v/%v, small batch %v/%v", pf.FlashAttention, ps.FlashAttention, pf.FlashAttnSmallBatch, ps.FlashAttnSmallBatch)
	}
	if ps.Groups[gguf.G7].Total() <= pf.Groups[gguf.G7].Total() {
		t.Errorf("compute %d with small batches, %d without", ps.Groups[gguf.G7].Total(), pf.Groups[gguf.G7].Total())
	}
	if ps.KV.Bytes != pf.KV.Bytes {
		t.Errorf("KV %d with small batches, %d without: the layout is flash attention's either way", ps.KV.Bytes, pf.KV.Bytes)
	}
	spec, err := ps.LoadSpec("m.gguf", "d.gguf")
	if err != nil || !spec.FlashAttnSmallBatch || !spec.FlashAttention {
		t.Errorf("load spec %+v, %v", spec, err)
	}
	for name, r := range map[string]Request{"q8_0": {KVType: "q8_0"}, "no FA": {NoFlashAttention: true}} {
		req := small
		req.KVType, req.NoFlashAttention = r.KVType, r.NoFlashAttention
		if p := mustPlan(t, req); p.FlashAttnSmallBatch {
			t.Errorf("%s: small batch kept", name)
		}
	}
}

// TestCudaNanoCarries: the switches of ggml-cuda in a request reach the load spec of a plan with a GPU, and a CPU plan
// does not keep them.
func TestCudaNanoCarries(t *testing.T) {
	model := e4b(t)
	req := Request{Model: model, Preset: preset(t, "uma-4g"), NCtx: 8192, Slots: 1, CudaNano: map[string]int{"MMVQ_NWARPS": 12}}
	p := mustPlan(t, req)
	spec, err := p.LoadSpec("m.gguf", "d.gguf")
	if err != nil || spec.CudaNano["MMVQ_NWARPS"] != 12 {
		t.Fatalf("load spec %+v, %v", spec.CudaNano, err)
	}
	req.CudaNano["MMVQ_NWARPS"] = 0
	if p.CudaNano["MMVQ_NWARPS"] != 12 {
		t.Errorf("the plan shares the request's map: %v", p.CudaNano)
	}
	cpu := Request{Model: model, Preset: preset(t, "ram-6g"), NCtx: 4096, Slots: 1, CudaNano: map[string]int{"NORM_LEAN": 0}}
	if pc := mustPlan(t, cpu); len(pc.CudaNano) != 0 {
		t.Errorf("a CPU plan keeps the switches: %v", pc.CudaNano)
	}
}

func TestSWAFullDecision(t *testing.T) {
	model := e4b(t)
	// ram-8g at 32k: the full SWA cache is 20 layers × 32768 cells × 2 KiB.
	p := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-8g")})
	if !p.KV.SWAFull || p.KV.CellsSWA != p.KV.CellsFull {
		t.Errorf("ram-8g: swa_full %v, cells %d/%d", p.KV.SWAFull, p.KV.CellsSWA, p.KV.CellsFull)
	}
	// vram-8g at 32k × 4 cannot afford it and says why.
	p = mustPlan(t, Request{Model: model, Preset: preset(t, "vram-8g")})
	if p.KV.SWAFull || p.KV.CellsSWA != 1024 || !strings.Contains(p.KV.SWAFullReason, "last 512 tokens") {
		t.Errorf("vram-8g: %+v", p.KV)
	}
}

func TestMTPAndOutputs(t *testing.T) {
	model := e4b(t)
	p := mustPlan(t, Request{Model: model, Preset: preset(t, "vram-8g")})
	if !p.MTP.Enabled || p.NOutputsMax != p.Slots*(1+DefaultDraftMax) || p.Groups[gguf.G6].VRAM == 0 {
		t.Errorf("MTP %+v, outputs %d, G6 %+v", p.MTP, p.NOutputsMax, p.Groups[gguf.G6])
	}
	off := mustPlan(t, Request{Model: model, Preset: preset(t, "vram-8g"), NoMTP: true})
	if off.MTP.Enabled || off.NOutputsMax != off.Slots || off.Groups[gguf.G6].Total() != 0 {
		t.Errorf("no MTP: %+v, outputs %d, G6 %+v", off.MTP, off.NOutputsMax, off.Groups[gguf.G6])
	}
	// The drafter adds no KV.
	if p.KV.Bytes != off.KV.Bytes && p.NCtx == off.NCtx && p.Slots == off.Slots {
		t.Errorf("KV with drafter %d, without %d", p.KV.Bytes, off.KV.Bytes)
	}
}

func TestFixedContext(t *testing.T) {
	model := e4b(t)
	p := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-8g"), NCtx: 8192, Slots: 2})
	if p.NCtx != 8192 || p.Slots != 2 {
		t.Errorf("fixed context became %d×%d", p.NCtx, p.Slots)
	}
	p = mustPlan(t, Request{Model: model, Preset: preset(t, "ram-1g"), NCtx: 131072, Slots: 8})
	if p.OK() {
		t.Errorf("131072×8 fit in 1 GiB: %+v", p.Memory)
	}
}

func TestDeviceSmallerThanPreset(t *testing.T) {
	model := e4b(t)
	dev := &Device{Selector: "llamacpp:vulkan0", Backend: BackendVulkan, Memory: 3 << 30}
	p := mustPlan(t, Request{Model: model, Preset: preset(t, "vram-8g"), Device: dev})
	if p.Memory.VRAM.Budget != 3<<30 || !hasWarning(p, "less than the preset") {
		t.Errorf("budget %d, warnings %v", p.Memory.VRAM.Budget, p.Warnings)
	}
	if p.ComputeCalibrated || !hasWarning(p, "not calibrated") {
		t.Error("a Vulkan plan claims calibrated compute buffers")
	}
}

func TestRAMPresetIgnoresDevice(t *testing.T) {
	model := e4b(t)
	dev := &Device{Selector: "llamacpp:vulkan0", Backend: BackendVulkan, Memory: 8 << 30}
	p := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-8g"), Device: dev})
	if p.Mode != RAMFull || p.Device != nil {
		t.Errorf("%s on %v", p.Mode, p.Device)
	}
	for _, a := range p.Tried[:4] {
		if a.Skipped == "" {
			t.Errorf("%s tried on a RAM preset: %+v", a.ID, a)
		}
	}
}

func TestLoadSpec(t *testing.T) {
	model := e4b(t)
	p := mustPlan(t, Request{Model: model, Preset: preset(t, "vram-4g")})
	s, err := p.LoadSpec("/m/model.gguf", "/m/drafter.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if s.Device != "llamacpp:cuda0" || s.GPULayers != -1 || s.NCtx != p.NCtx || s.Slots != p.Slots ||
		s.NOutputsMax != p.NOutputsMax || s.DrafterPath == "" || s.DraftMax != DefaultDraftMax ||
		s.KVTypeV != "f16" || !s.FlashAttention || s.LoadMode != p.LoadMode {
		t.Errorf("%+v", s)
	}
	if _, err := p.LoadSpec("/m/model.gguf", ""); err == nil {
		t.Error("an MTP plan converted without the drafter")
	}
	cpu := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-2g"), NoMTP: true})
	s, err = cpu.LoadSpec("/m/model.gguf", "")
	if err != nil || s.Device != "cpu" || s.GPULayers != 0 || s.DrafterPath != "" {
		t.Errorf("%+v, %v", s, err)
	}
}

// TestPlanAgainstTinyModel runs the planner on the synthetic fixture, which
// has no drafter and much smaller tensors: every preset should land in the
// first mode its kind allows.
func TestPlanAgainstTinyModel(t *testing.T) {
	b, err := gguf.TinyGemma4().Bytes()
	if err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Parse(strings.NewReader(string(b)))
	if err != nil {
		t.Fatal(err)
	}
	f.FileSize = int64(len(b))
	inv, err := gguf.BuildInventory(f)
	if err != nil {
		t.Fatal(err)
	}
	inv.Role = "model"
	model, err := ModelFrom("tiny", []gguf.Inventory{inv})
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]Mode{"vram-2g": VRAMFull, "ram-1g": RAMFull, "uma-4g": VRAMFull} {
		p := mustPlan(t, Request{Model: model, Preset: preset(t, name)})
		if p.Mode != want || p.MTP.Enabled {
			t.Errorf("%s: %s, MTP %v", name, p.Mode, p.MTP.Enabled)
		}
	}
}

// OPT-1: M7 repacks its core, which copies it into anonymous memory, so it
// range-locks nothing, and the planned RAM is the same as with the lock.
func TestM7Repacks(t *testing.T) {
	model := e4b(t)
	all := ProvenModes{RAMPLEDisk: true, RAMHybrid: true}
	rp := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-4g"), Proven: all, Mode: RAMPLEDisk})
	lk := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-4g"), Proven: all, Mode: RAMPLEDisk, NoRepack: true})
	if !rp.OK() || !lk.OK() {
		t.Fatalf("refused: %q %q", rp.Refused, lk.Refused)
	}
	if !rp.Repack || rp.LoadMode != backend.LoadMMap || rp.LazyMode != backend.LazyOn {
		t.Errorf("repacked M7: repack %v, load %s lazy %s", rp.Repack, rp.LoadMode, rp.LazyMode)
	}
	// Only the small tensors every token reads (G4) are range-locked: the
	// matrices are anonymous repacked copies, the PLE table stays in the file.
	var locked int64
	for _, r := range rp.LockRanges {
		locked += r.Length
	}
	if len(rp.LockRanges) == 0 || locked > 4*model.Target.Groups[gguf.G4] {
		t.Errorf("repacked M7 locks %d bytes in %d ranges, want about the %d bytes of G4", locked, len(rp.LockRanges), model.Target.Groups[gguf.G4])
	}
	if lk.Repack || len(lk.LockRanges) == 0 {
		t.Errorf("M7 without repack: repack %v, %d ranges", lk.Repack, len(lk.LockRanges))
	}
	if rp.Groups[gguf.G1].FileBacked != model.Target.Groups[gguf.G1] {
		t.Errorf("the PLE table is not file-backed: %+v", rp.Groups[gguf.G1])
	}
	for _, g := range []gguf.Group{gguf.G2, gguf.G3} {
		if rp.Groups[g].RAMLocked != model.Target.Groups[g] || rp.Groups[g].FileBacked != 0 {
			t.Errorf("%s: %+v, want %d of anonymous memory", g, rp.Groups[g], model.Target.Groups[g])
		}
	}
	// Both plans fit the same budget, and the repacked one needs the tied
	// head's copy as well, which the locked one aliased: no more than 1% apart.
	if d := rp.Memory.RAM.Used - lk.Memory.RAM.Used; d < -(1<<26) || d > 1<<26 {
		t.Errorf("planned RAM differs by %d bytes between repack and lock", d)
	}
}

// OPT-8: a plan that leaves layer weights in the file names those ranges
// per layer, and a plan that holds the layers in memory names none.
func TestStreamRanges(t *testing.T) {
	model := e4b(t)
	all := ProvenModes{RAMFull: true, RAMPLEDisk: true, RAMHybrid: true, MMap: true}
	full := mustPlan(t, Request{Model: model, Preset: preset(t, "ram-8g"), Proven: all})
	if len(full.StreamRanges) != 0 {
		t.Errorf("%s keeps the layers in memory but names %d stream ranges", full.ModeID, len(full.StreamRanges))
	}
	for _, name := range []string{"ram-2g", "ram-1g"} {
		p := mustPlan(t, Request{Model: model, Preset: preset(t, name), Proven: all, NoMTP: true})
		if !p.OK() {
			t.Fatalf("%s refused: %s", name, p.Refused)
		}
		layers := map[int]int64{}
		for _, r := range p.StreamRanges {
			if r.Layer < 0 || r.Layer >= model.Target.HParams.BlockCount || r.Length <= 0 {
				t.Errorf("%s: stream range %+v", name, r)
			}
			layers[r.Layer] += r.Length
		}
		if len(layers) == 0 {
			t.Errorf("%s (%s) leaves the layers in the file but names no stream range", name, p.ModeID)
		}
		// The ranges are the file-backed weights of the layers: no more
		// than the plan says is file-backed.
		var total int64
		for _, n := range layers {
			total += n
		}
		if fb := p.Memory.FileBacked; total > fb {
			t.Errorf("%s: %d stream bytes but only %d file-backed", name, total, fb)
		}
		spec, err := p.LoadSpec("/m.gguf", "")
		if err != nil {
			t.Fatal(err)
		}
		if spec.Stream == nil || len(spec.Stream.Ranges) != len(p.StreamRanges) || spec.Stream.Ahead != 0 || spec.Stream.Evict {
			t.Errorf("%s: load spec stream %+v, want the ranges and everything off", name, spec.Stream)
		}
	}
}
