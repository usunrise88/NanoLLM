package plan

import (
	"fmt"
	"sort"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/gguf"
)

// finish turns the chosen trial into a plan: load settings, overrides, lock
// ranges, the MTP decision and the warnings a reader of the plan needs.
func (pl *planner) finish(t *trial, target Target) *Plan {
	info, _ := Info(t.mode)
	p := &Plan{
		Model: pl.req.Model.ID, Preset: pl.req.Preset.Name, Device: pl.dev,
		Mode: t.mode, ModeID: info.ID,
		NCtx: t.shape.nCtx, Slots: t.shape.slots,
		NBatch: pl.nBatch, NUBatch: pl.nUBatch, NOutputsMax: t.nOutputsMax,
		FlashAttention: pl.fa,
		LoadMode:       t.spec.load, LazyMode: t.spec.lazy, Repack: t.spec.repack,
		Groups: t.groups, Memory: t.mem, Target: target,
	}
	if !t.spec.gpu {
		p.Device = nil
	}
	switch {
	case !t.spec.gpu:
		p.GPULayers, p.LayersOnGPU = 0, 0
	case t.n >= pl.h.BlockCount:
		p.GPULayers, p.LayersOnGPU = -1, pl.h.BlockCount
	default:
		p.GPULayers, p.LayersOnGPU = t.n+1, t.n
	}
	backendKind := BackendCPU
	if t.spec.gpu && pl.dev != nil {
		backendKind = pl.dev.Backend
	}
	p.ComputeCalibrated = coeffsFor(backendKind).Calibrated

	p.KV = KV{
		TypeK: pl.kvType.String(), TypeV: pl.kvType.String(), SWAFull: t.shape.swaFull,
		CellsFull: t.cells.Full, CellsSWA: t.cells.SWA, Streams: t.cells.Streams, Bytes: t.kvBytes,
	}
	if t.shape.swaFull {
		p.KV.SWAFullReason = "the full sliding-window cache fits the budget: the prompt cache can reuse a prefix of any length"
	} else {
		full := KVShape{NCtx: t.shape.nCtx, Slots: t.shape.slots, NUBatch: pl.nUBatch, SWAFull: true,
			FlashAttention: pl.fa, TypeK: pl.kvType, TypeV: pl.kvType}
		extra := KVBytes(pl.h, full) - t.kvBytes
		p.KV.SWAFullReason = fmt.Sprintf("the full sliding-window cache needs %s more than the budget leaves: "+
			"the prompt cache reuses a prefix only within the last %d tokens, and a slot without one prefills again",
			mibStr(extra), pl.h.SlidingWindow)
	}

	if t.spec.inputOnGPU && pl.dev != nil {
		// M1 on a discrete GPU: upstream keeps the input layer on the CPU, so
		// the embedding and PLE tables are moved by name.
		p.Overrides = []backend.BufferOverride{
			{Pattern: `^token_embd\.weight$`, Device: pl.dev.Selector},
			{Pattern: `^per_layer_token_embd\.weight$`, Device: pl.dev.Selector},
		}
	}
	p.LockRanges = lockRanges(t.locks)

	switch {
	case pl.req.Model.Drafter == nil:
		p.MTP = MTP{Reason: "the model has no drafter"}
	case pl.req.NoMTP:
		p.MTP = MTP{Reason: "disabled by the request"}
	default:
		where := "the CPU"
		if t.spec.gpu && pl.dev != nil {
			where = pl.dev.Selector
		}
		p.MTP = MTP{
			Enabled: true, DraftMax: pl.draftMax, DraftPMin: pl.req.DraftPMin,
			Reason: fmt.Sprintf("the drafter (%s) runs on %s with the target's output layer and reads its KV cache",
				mibStr(pl.req.Model.Drafter.Total()), where),
		}
	}
	p.Warnings = pl.warnings(t, p)
	return p
}

func (pl *planner) warnings(t *trial, p *Plan) []string {
	var w []string
	b := pl.req.Preset.Budget
	if pl.dev != nil && pl.dev.Memory > 0 && t.spec.gpu {
		want := b.VRAM
		if pl.dev.Unified {
			want = b.Unified
		}
		if pl.dev.Memory < want {
			w = append(w, fmt.Sprintf("%s has %s, less than the preset's %s; the plan uses what the device has",
				pl.dev.Selector, mibStr(pl.dev.Memory), mibStr(want)))
		}
	}
	if t.mem.RAM != nil && !t.mem.RAM.Budgeted {
		w = append(w, fmt.Sprintf("a VRAM-only preset does not budget host RAM: this plan takes %s of it, measured, not bounded",
			mibStr(t.mem.RAM.Used)))
	}
	if !p.ComputeCalibrated {
		w = append(w, fmt.Sprintf("compute buffers (G7) for %s are not calibrated yet: the CPU fit stands in", pl.dev.Backend))
	}
	if t.mem.FileBacked > 0 {
		w = append(w, fmt.Sprintf("%s stays file-backed: outside every budget, and cold pages fault in from disk",
			mibStr(t.mem.FileBacked)))
	}
	if t.spec.pleOnDisk {
		w = append(w, "the preset leaves no host RAM for the PLE table: it is mapped from the file and read on demand")
	}
	if t.mode == RAMHybrid {
		g3 := t.groups[gguf.G3]
		w = append(w, fmt.Sprintf("%s of %s of FFN weights are pinned; decode reads the rest from disk",
			mibStr(g3.RAMLocked), mibStr(g3.Total())))
	}
	if p.NCtx < p.Target.NCtx || p.Slots < p.Target.Slots {
		w = append(w, fmt.Sprintf("context is %d×%d, short of the preset's target %d×%d",
			p.NCtx, p.Slots, p.Target.NCtx, p.Target.Slots))
	}
	return w
}

// lockRanges merges the locked tensors of each file into ranges: tensors
// that follow each other in the file, padding included, become one range.
func lockRanges(locks []lockedTensor) []Range {
	sort.SliceStable(locks, func(i, j int) bool {
		if locks[i].role != locks[j].role {
			return locks[i].role > locks[j].role // model before drafter
		}
		return locks[i].t.Offset < locks[j].t.Offset
	})
	var out []Range
	for _, l := range locks {
		end := l.t.Offset + l.t.Bytes
		if n := len(out); n > 0 && out[n-1].Role == l.role {
			last := &out[n-1]
			if pad(last.Offset+last.Length, l.pad) == l.t.Offset {
				last.Length = end - last.Offset
				continue
			}
		}
		out = append(out, Range{Role: l.role, Offset: l.t.Offset, Length: l.t.Bytes})
	}
	return out
}

func pad(n, align int64) int64 {
	if align <= 1 {
		return n
	}
	return (n + align - 1) &^ (align - 1)
}

func mibStr(b int64) string {
	return fmt.Sprintf("%.1f MiB", float64(b)/(1<<20))
}
