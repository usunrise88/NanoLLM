package plan

import "github.com/usunrise88/nanollm/internal/gguf"

// KVShape is the context a KV cache is sized for. NCtx is per slot, the way
// the rest of NanoLLM counts context; llama.cpp's n_ctx is NCtx·Slots.
type KVShape struct {
	NCtx    int
	Slots   int
	NUBatch int
	// Unified shares one cache across slots (kv_unified); otherwise each slot
	// is a stream of its own.
	Unified bool
	// SWAFull gives the sliding-window layers the full context too.
	SWAFull bool
	// FlashAttention decides the V layout: without it V is stored
	// transposed and every layer is padded to the widest V.
	FlashAttention bool
	TypeK, TypeV   gguf.Type
}

// KVCells is how llama.cpp b11600 sizes the two caches of an iSWA model
// (llama-context.cpp: n_ctx and n_ctx_seq are padded to 256;
// llama-kv-cache-iswa.cpp: the SWA cache is padded to 256 too).
type KVCells struct {
	NCtx    int // llama n_ctx after padding, all slots
	NCtxSeq int // cells per stream of the full-attention cache
	Full    int // cells per stream, full-attention layers
	SWA     int // cells per stream, sliding-window layers
	Streams int
}

func pad256(n int) int { return (n + 255) &^ 255 }

// Cells applies the b11600 formula:
//
//	n_ctx      = pad256(NCtx·Slots)
//	n_ctx_seq  = unified ? n_ctx : pad256(n_ctx/Slots)
//	full cells = n_ctx_seq
//	SWA cells  = swa_full ? n_ctx_seq
//	           : pad256(min(n_ctx_seq, n_swa·(unified ? Slots : 1) + n_ubatch))
func Cells(h gguf.HParams, s KVShape) KVCells {
	slots := max(s.Slots, 1)
	c := KVCells{NCtx: pad256(s.NCtx * slots), Streams: slots}
	if s.Unified {
		c.NCtxSeq, c.Streams = c.NCtx, 1
	} else {
		c.NCtxSeq = pad256(c.NCtx / slots)
		c.NCtx = c.NCtxSeq * slots
	}
	c.Full = c.NCtxSeq
	if s.SWAFull {
		c.SWA = c.NCtxSeq
	} else {
		perStream := 1
		if s.Unified {
			perStream = slots
		}
		c.SWA = pad256(min(c.NCtxSeq, h.SlidingWindow*perStream+s.NUBatch))
	}
	return c
}

// KVLayerBytes is the K and V bytes of layer il; zero for a layer that reuses
// another's cache. The drafter never calls this: it reads the target's cache.
func KVLayerBytes(h gguf.HParams, s KVShape, il int) (k, v int64) {
	if il >= h.KVLayers() {
		return 0, 0
	}
	c := Cells(h, s)
	cells := c.Full
	if h.IsSWA(il) {
		cells = c.SWA
	}
	vWidth := h.EmbdVGQA(il)
	if !s.FlashAttention {
		vWidth = h.EmbdVGQAMax()
	}
	n := int64(cells) * int64(c.Streams)
	return s.TypeK.RowSize(int64(h.EmbdKGQA(il))) * n, s.TypeV.RowSize(int64(vWidth)) * n
}

// KVBytes is the whole cache (G5).
func KVBytes(h gguf.HParams, s KVShape) int64 {
	var total int64
	for il := range h.BlockCount {
		k, v := KVLayerBytes(h, s, il)
		total += k + v
	}
	return total
}
