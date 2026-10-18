package plan

import (
	"github.com/usunrise88/nanollm/internal/gguf"
)

// place puts every tensor, the KV cache and the compute buffers where llama.cpp
// b11600 would put them in this mode, and checks the budgets.
//
// The rules it follows, all from the pinned loader:
//   - Input-layer tensors (token_embd, per_layer_token_embd) stay in host
//     memory whatever gpu_layers says; only an override moves them (M1).
//   - Layers are offloaded from the end: N layers on the device are the last
//     N, and the output layer goes with the first offloaded one.
//   - A tied LM head is token_embd loaded a second time into the output
//     layer's buffer. On a discrete GPU that is a second copy in VRAM; on the
//     CPU it aliases the mapping, unless the weights are read into anonymous
//     memory or repacked, which copy it.
//   - The KV cache of a layer lives on that layer's device; the last
//     shared_kv_layers layers have none.
//   - The drafter has no KV of its own and runs where the target's output
//     layer is.
func (pl *planner) place(m Mode, n int, s shape, spec modeSpec) *trial {
	t := &trial{mode: m, n: n, shape: s, spec: spec, groups: map[gguf.Group]Bytes{}}
	if !spec.gpu {
		t.n = 0
	}
	nLayer := pl.h.BlockCount
	onDevice := func(role string, tn gguf.Tensor) bool {
		switch {
		case !spec.gpu:
			return false
		case tn.Input:
			return (spec.inputOnGPU && role == "model") || spec.unified
		case role == "drafter":
			// All of the drafter sits with the target's output layer.
			return true
		case tn.Output, tn.Layer < 0:
			// Layer-less tensors other than inputs (rope_freqs) are a few
			// bytes; they are counted with the output layer.
			return true
		default:
			return tn.Layer >= nLayer-t.n
		}
	}
	add := func(g gguf.Group, b Bytes) {
		cur := t.groups[g]
		cur.add(b)
		t.groups[g] = cur
	}
	weight := func(role string, tn gguf.Tensor, g gguf.Group, align int64, lock bool) {
		var b Bytes
		dev := onDevice(role, tn)
		switch {
		case dev && spec.unified && tn.Group == gguf.G1 && spec.pleFile:
			b.FileBacked = tn.Bytes
		case dev && spec.unified && tn.Input && role == "drafter":
			// The drafter reads the target's embedding; its own input copy is
			// mapped and never touched.
			b.FileBacked = tn.Bytes
		case dev:
			b.VRAM = tn.Bytes
		case lock:
			b.RAMLocked = tn.Bytes
			t.locks = append(t.locks, lockedTensor{role: role, t: tn, pad: align})
		case tn.Group == gguf.G1 && spec.pleFile:
			b.FileBacked = tn.Bytes
		case spec.hostAnon:
			b.RAMLocked = tn.Bytes
		default:
			b.FileBacked = tn.Bytes
		}
		add(g, b)

		// The tied LM head: a second load of token_embd into the output
		// layer's buffer.
		if tn.Name != "token_embd.weight" {
			return
		}
		tied := (role == "model" && pl.h.TiedOutput) || role == "drafter"
		if !tied {
			return
		}
		var dup Bytes
		switch {
		case spec.gpu && !spec.unified:
			dup.VRAM = tn.Bytes
		case !spec.gpu && (spec.hostAnon || spec.repack):
			dup.RAMLocked = tn.Bytes
		}
		add(g, dup)
	}

	target := pl.req.Model.Target
	lockHotOrAll := spec.lockAll || spec.lockHot
	for _, tn := range target.Tensors {
		lock := lockHotOrAll && tn.Group != gguf.G1 && (spec.lockAll || tn.Group != gguf.G3)
		weight("model", tn, tn.Group, target.Alignment, lock)
	}
	if pl.mtp {
		d := pl.req.Model.Drafter
		for _, tn := range d.Tensors {
			weight("drafter", tn, gguf.G6, d.Alignment, lockHotOrAll)
		}
	}

	// G5.
	kv := KVShape{
		NCtx: s.nCtx, Slots: s.slots, NUBatch: pl.nUBatch, SWAFull: s.swaFull,
		FlashAttention: pl.fa, TypeK: pl.kvType, TypeV: pl.kvType,
	}
	t.cells = Cells(pl.h, kv)
	for il := range nLayer {
		k, v := KVLayerBytes(pl.h, kv, il)
		if k+v == 0 {
			continue
		}
		t.kvBytes += k + v
		if spec.gpu && (spec.unified || il >= nLayer-t.n) {
			add(gguf.G5, Bytes{VRAM: k + v})
		} else {
			add(gguf.G5, Bytes{RAMLocked: k + v})
		}
	}

	// G7: the target's and the drafter's compute buffers, and the host-side
	// buffers around them.
	t.nOutputsMax = s.slots
	if pl.mtp {
		t.nOutputsMax = s.slots * (1 + pl.draftMax)
	}
	t.nOutputsMax = min(t.nOutputsMax, pl.nBatch)
	backendKind := BackendCPU
	if spec.gpu && pl.dev != nil {
		backendKind = pl.dev.Backend
	}
	co := coeffsFor(backendKind)
	cs := ComputeShape{NUBatch: pl.nUBatch, NOutputsMax: t.nOutputsMax, NKV: t.cells.Full, FlashAttention: pl.fa}
	compute := co.ComputeBytes(pl.h, cs)
	vocab := int64(pl.h.VocabSize)
	// The host output buffer holds the logits of every output.
	hostOut := vocab * 4 * int64(t.nOutputsMax)
	if pl.mtp {
		dh := pl.req.Model.Drafter.HParams
		dcs := cs
		dcs.NOutputsMax = s.slots // the draft context outputs one row per slot
		compute += co.ComputeBytes(dh, dcs)
		hostOut += vocab * 4 * int64(s.slots)
	}
	switch {
	case !spec.gpu:
		add(gguf.G7, Bytes{RAMLocked: compute + hostOut})
	case spec.unified:
		add(gguf.G7, Bytes{VRAM: compute + hostOut})
	default:
		// The input layer runs on the CPU: its get_rows outputs for the
		// token and per-layer embeddings are host compute.
		ub := int64(pl.nUBatch)
		hostIn := ub * 4 * int64(pl.h.EmbeddingLength+pl.h.EmbeddingLengthPerLayerInput*nLayer)
		host := hostOut + hostIn
		if t.n < nLayer {
			// Layers left on the CPU need a compute buffer of their own; it
			// is sized like the device's, without the logits.
			host += cpuCoeffs.ComputeBytes(pl.h, ComputeShape{NUBatch: pl.nUBatch, NOutputsMax: 1, NKV: t.cells.Full,
				FlashAttention: pl.fa})
		}
		add(gguf.G7, Bytes{VRAM: compute, RAMLocked: host})
	}

	if spec.lockHot {
		pl.lockG3(t)
	}
	pl.budget(t)
	return t
}

// lockG3 is M8's "part of G3": with the hot groups pinned, it pins FFN
// tensors in file order while the RAM budget has room, and leaves the rest
// file-backed. Decode then reads only the unpinned share from disk.
func (pl *planner) lockG3(t *trial) {
	pl.budget(t)
	u := t.mem.RAM
	if t.mem.Unified != nil {
		u = t.mem.Unified
	}
	if u == nil || !t.short.zero() {
		return
	}
	free := u.Budget - u.Used
	target := pl.req.Model.Target
	g3 := t.groups[gguf.G3]
	for _, tn := range target.Tensors {
		if tn.Group != gguf.G3 || tn.Bytes > free {
			continue
		}
		free -= tn.Bytes
		g3.FileBacked -= tn.Bytes
		g3.RAMLocked += tn.Bytes
		t.locks = append(t.locks, lockedTensor{role: "model", t: tn, pad: target.Alignment})
	}
	t.groups[gguf.G3] = g3
}

// budget sums the trial against the preset. Host RAM under a VRAM-only
// preset is measured, not bounded; on unified memory everything resident is
// one pool, and file-backed pages are outside every budget.
func (pl *planner) budget(t *trial) {
	var sum Bytes
	for _, b := range t.groups {
		sum.add(b)
	}
	b := pl.req.Preset.Budget
	devBudget := func(preset int64) int64 {
		if pl.dev != nil && pl.dev.Memory > 0 && pl.dev.Memory < preset {
			return pl.dev.Memory
		}
		return preset
	}
	t.mem = Memory{FileBacked: sum.FileBacked, OverheadRAM: HostOverhead}
	t.short = Shortfall{}
	switch {
	case b.Unified > 0:
		// GPU or CPU backend alike: one pool.
		u := &Usage{Budget: devBudget(b.Unified), Used: sum.VRAM + sum.RAMLocked + HostOverhead, Budgeted: true}
		t.mem.Unified = u
		t.short.Unified = u.Short()
	case t.spec.gpu:
		ov := gpuOverhead(pl.dev)
		t.mem.OverheadVRAM = ov
		v := &Usage{Budget: devBudget(b.VRAM), Used: sum.VRAM + ov, Budgeted: true}
		r := &Usage{Budget: b.RAM, Used: sum.RAMLocked + HostOverhead, Budgeted: b.RAM > 0}
		t.mem.VRAM, t.mem.RAM = v, r
		t.short.VRAM, t.short.RAM = v.Short(), r.Short()
	default:
		r := &Usage{Budget: b.RAM, Used: sum.RAMLocked + HostOverhead, Budgeted: true}
		t.mem.RAM = r
		t.short.RAM = r.Short()
	}
}
