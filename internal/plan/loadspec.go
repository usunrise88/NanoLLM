package plan

import (
	"errors"
	"fmt"

	"github.com/usunrise88/nanollm/internal/backend"
)

// LoadSpec turns the plan into what a backend loads: every budget decision is
// already made, so the backend never sees a preset (backend.LoadSpec). NCtx
// stays per slot, as Capabilities reports it; the backend multiplies by Slots
// for llama.cpp's n_ctx. Lock ranges are not part of it: range-lock pins them
// itself, beside the engine's own mapping.
func (p *Plan) LoadSpec(modelPath, drafterPath string) (backend.LoadSpec, error) {
	if !p.OK() {
		return backend.LoadSpec{}, fmt.Errorf("plan refused: %s", p.Refused)
	}
	if modelPath == "" {
		return backend.LoadSpec{}, errors.New("no model path")
	}
	if p.MTP.Enabled && drafterPath == "" {
		return backend.LoadSpec{}, errors.New("the plan runs MTP but no drafter path was given")
	}
	s := backend.LoadSpec{
		Path:           modelPath,
		Device:         "cpu",
		GPULayers:      p.GPULayers,
		Overrides:      append([]backend.BufferOverride(nil), p.Overrides...),
		LoadMode:       p.LoadMode,
		LazyMode:       p.LazyMode,
		Repack:         p.Repack,
		NCtx:           p.NCtx,
		Slots:          p.Slots,
		NBatch:         p.NBatch,
		NUBatch:        p.NUBatch,
		NOutputsMax:    p.NOutputsMax,
		KVTypeK:        p.KV.TypeK,
		KVTypeV:        p.KV.TypeV,
		FlashAttention: p.FlashAttention,
		SWAFull:        p.KV.SWAFull,
		KVUnified:      p.KV.Unified,
	}
	if p.Device != nil {
		s.Device = p.Device.Selector
	}
	if p.MTP.Enabled {
		s.DrafterPath = drafterPath
		s.DraftMax = p.MTP.DraftMax
		s.DraftPMin = p.MTP.DraftPMin
	}
	if len(p.StreamRanges) > 0 {
		st := &backend.StreamSpec{}
		for _, r := range p.StreamRanges {
			st.Ranges = append(st.Ranges, backend.StreamRange{Layer: r.Layer, Offset: r.Offset, Length: r.Length})
		}
		s.Stream = st
	}
	// lazy loading reads through a mapping; direct I/O has none.
	if s.LoadMode == backend.LoadDirectIO && s.LazyMode != backend.LazyOff {
		return s, fmt.Errorf("plan pairs lazy %s with direct I/O", s.LazyMode)
	}
	return s, nil
}
