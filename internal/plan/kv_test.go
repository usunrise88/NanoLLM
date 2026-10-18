package plan

import (
	"testing"

	"github.com/usunrise88/nanollm/internal/gguf"
)

const mib = 1 << 20

// kvSplit sums the full-attention and sliding-window caches separately, the
// two lines libllama logs.
func kvSplit(h gguf.HParams, s KVShape) (full, swa int64) {
	for il := range h.BlockCount {
		k, v := KVLayerBytes(h, s, il)
		if h.IsSWA(il) {
			swa += k + v
		} else {
			full += k + v
		}
	}
	return full, swa
}

// TestKVMatchesLlamaLog checks the formula against the `llama_kv_cache: size`
// lines of b11600 on E4B, recorded on the stage-1 VM (llama-completion and
// llama-server, CPU). n_ctx in the logs is all slots; here it is per slot.
func TestKVMatchesLlamaLog(t *testing.T) {
	h := e4b(t).Target.HParams
	f16, q8 := gguf.TypeF16, gguf.TypeQ8_0
	for _, c := range []struct {
		name               string
		shape              KVShape
		cellsFull, cellsSW int
		full, swa          float64 // MiB, as logged
	}{
		{"c4096", KVShape{NCtx: 4096, Slots: 1, NUBatch: 512, FlashAttention: true, TypeK: f16, TypeV: f16}, 4096, 1024, 64, 40},
		{"ub256", KVShape{NCtx: 4096, Slots: 1, NUBatch: 256, FlashAttention: true, TypeK: f16, TypeV: f16}, 4096, 768, 64, 30},
		{"c32768", KVShape{NCtx: 32768, Slots: 1, NUBatch: 512, FlashAttention: true, TypeK: f16, TypeV: f16}, 32768, 1024, 512, 40},
		{"no FA", KVShape{NCtx: 4096, Slots: 1, NUBatch: 512, TypeK: f16, TypeV: f16}, 4096, 1024, 64, 60},
		{"no FA c32768", KVShape{NCtx: 32768, Slots: 1, NUBatch: 512, TypeK: f16, TypeV: f16}, 32768, 1024, 512, 60},
		{"q8_0", KVShape{NCtx: 4096, Slots: 1, NUBatch: 512, FlashAttention: true, TypeK: q8, TypeV: q8}, 4096, 1024, 34, 21.25},
		{"swa_full", KVShape{NCtx: 4096, Slots: 1, NUBatch: 512, SWAFull: true, FlashAttention: true, TypeK: f16, TypeV: f16}, 4096, 4096, 64, 160},
		// -c 8192 -np 4: four streams of 2048.
		{"np4 c8192", KVShape{NCtx: 2048, Slots: 4, NUBatch: 512, FlashAttention: true, TypeK: f16, TypeV: f16}, 2048, 1024, 128, 160},
		// llama-server -c 16384 -np 4 with MTP.
		{"server np4", KVShape{NCtx: 4096, Slots: 4, NUBatch: 512, FlashAttention: true, TypeK: f16, TypeV: f16}, 4096, 1024, 256, 160},
	} {
		cells := Cells(h, c.shape)
		if cells.Full != c.cellsFull || cells.SWA != c.cellsSW {
			t.Errorf("%s: cells %d/%d, logged %d/%d", c.name, cells.Full, cells.SWA, c.cellsFull, c.cellsSW)
		}
		full, swa := kvSplit(h, c.shape)
		if full != int64(c.full*mib) || swa != int64(c.swa*mib) {
			t.Errorf("%s: %.2f + %.2f MiB, logged %.2f + %.2f", c.name,
				float64(full)/mib, float64(swa)/mib, c.full, c.swa)
		}
	}
}

func TestCellsPadding(t *testing.T) {
	h := gguf.HParams{BlockCount: 1, SlidingWindow: 512}
	// n_ctx rounds up to 256; per-stream context rounds again.
	c := Cells(h, KVShape{NCtx: 1000, Slots: 3, NUBatch: 512})
	if c.NCtx != 3*1024 || c.NCtxSeq != 1024 || c.Streams != 3 {
		t.Errorf("%+v", c)
	}
	// Unified: one stream, and the window counts every slot.
	c = Cells(h, KVShape{NCtx: 4096, Slots: 4, NUBatch: 512, Unified: true})
	if c.NCtxSeq != 16384 || c.Streams != 1 || c.SWA != pad256(512*4+512) {
		t.Errorf("unified: %+v", c)
	}
}
