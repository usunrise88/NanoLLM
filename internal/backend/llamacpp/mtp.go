package llamacpp

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"runtime"
	"unsafe"

	"github.com/usunrise88/nanollm/internal/backend"
)

// MTP drafting with a Gemma 4 assistant (STAGE1 1.5.M), the path llama.cpp's
// common/speculative.cpp calls is_mem_shared.
//
// The drafter is a small model with no KV cache of its own: its context is
// created with ctx_type MTP and ctx_other pointing at the target's context,
// and its attention layers read the target's cache. It takes a token and a
// state row — the target's last hidden state (h_nextn) — and predicts the
// token after. Each draft round feeds the drafted token back with the
// drafter's own h_nextn, at the same position: nothing is written to the
// cache, so the position never advances (transformers' Gemma4Assistant does
// the same).
//
// A step that drafts:
//
//  1. drafts up to MaxDraft tokens per slot from (next token, pendingH),
//     greedily, one drafter decode per round for all slots together,
//     stopping a slot when its top token's probability among the top 10 is
//     below p_min;
//  2. puts the next token and the drafts into the target batch, each with
//     an output row;
//  3. samples position by position through the slot's chain, so grammar and
//     penalties accept only what is returned, and stops at the first token
//     that differs from the draft, which becomes the step's last token;
//  4. removes the rejected tail from the cache and keeps the h_nextn row of
//     the last accepted input as the next step's pendingH.
//
// This matches common_speculative_impl_draft_mtp with greedy drafting
// (top-k 10, accept by sample-and-match) and n_min 0.

// drafter is a loaded MTP drafter and its context.
type drafter struct {
	mdl   LlamaModel
	ctx   LlamaContext
	batch LlamaBatchExt
	// nEmbd is the width of a state row, the target's n_embd_out.
	nEmbd    int
	nVocab   int
	maxDraft int
	pMin     float64
}

// defaultDraftMax is llama.cpp's --spec-draft-n-max default.
const defaultDraftMax = 3

// draftTopK is the candidate set whose softmax the p_min test reads, as in
// llama.cpp's MTP sampler.
const draftTopK = 10

// loadDrafter loads the drafter with the target's placement and creates its
// context with the target's shape: llama.cpp builds the drafter's view of
// the target cache from these parameters, so n_ctx, n_seq_max, the KV types,
// flash attention, swa_full and kv_unified must all agree.
func (m *model) loadDrafter(spec backend.LoadSpec, mp LlamaModelParams, cp LlamaContextParams) (*drafter, error) {
	if spec.DraftMax < 0 || spec.DraftPMin < 0 || spec.DraftPMin > 1 || math.IsNaN(spec.DraftPMin) {
		return nil, errors.New("llamacpp: draft max or p_min out of range")
	}
	if m.recurrent {
		return nil, errors.New("llamacpp: MTP drafting needs a target whose cache can be trimmed")
	}
	if m.b.lib.cxx == nil || !m.b.lib.cxx["llama_set_embeddings_nextn"] || !m.b.lib.cxx["llama_get_embeddings_nextn_ith"] || !m.b.lib.cxx["llama_get_ctx_other"] {
		return nil, errors.New("llamacpp: this libllama does not export the MTP staging functions")
	}
	// The device list and overrides were consumed by the target's load;
	// the drafter gets the same device and no overrides.
	mp.TensorBuftOverrides = nil
	dmdl := llamaModelLoadFromFile(spec.DrafterPath, mp)
	if dmdl == 0 {
		return nil, fmt.Errorf("llamacpp: load drafter %s failed%s", spec.DrafterPath, lastErrors(m.loadLog))
	}
	d := &drafter{mdl: dmdl, maxDraft: spec.DraftMax, pMin: spec.DraftPMin}
	if d.maxDraft == 0 {
		d.maxDraft = defaultDraftMax
	}
	fail := func(err error) (*drafter, error) {
		d.close()
		return nil, err
	}
	d.nEmbd = int(llamaModelNEmbdOut(dmdl))
	if want := int(llamaModelNEmbdOut(m.mdl)); d.nEmbd != want {
		return fail(fmt.Errorf("llamacpp: drafter %s takes %d-wide states, the target produces %d", spec.DrafterPath, d.nEmbd, want))
	}
	if llamaModelNLayerNextn(dmdl) <= 0 {
		return fail(fmt.Errorf("llamacpp: %s is not an MTP drafter (no nextn layers)", spec.DrafterPath))
	}
	d.nVocab = int(llamaVocabNTokens(llamaModelGetVocab(dmdl)))

	dp := cp
	dp.CtxType = LlamaContextTypeMtp
	dp.CtxOther = m.ctx
	dp.NCtx = llamaNCtx(m.ctx)
	// One draft token per slot and round.
	dp.NOutputsMax = cp.NSeqMax
	dp.NOutputsMaxPerSeq = 1
	d.ctx = llamaInitFromModel(dmdl, dp)
	if d.ctx == 0 {
		return fail(fmt.Errorf("llamacpp: create the drafter context failed%s", lastErrors(m.loadLog)))
	}
	if llamaGetCtxOther(d.ctx) != m.ctx {
		return fail(errors.New("llamacpp: the drafter context does not share the target's cache"))
	}
	d.batch = llamaBatchExtInit(d.ctx)
	// Masked on both sides: a state row only where a logits row is asked
	// for — the sampled positions of the target, the draft positions of the
	// drafter. llama.cpp's driver leaves the target unmasked because its
	// other MTP kinds keep a cache of their own and replay the prompt into
	// it; the Gemma 4 assistant reads the target's cache and needs only the
	// row before the next token.
	llamaSetEmbeddingsNextn(m.ctx, true, true)
	llamaSetEmbeddingsNextn(d.ctx, true, true)
	return d, nil
}

func (d *drafter) close() {
	if d == nil {
		return
	}
	if d.batch != 0 {
		llamaBatchExtFree(d.batch)
	}
	if d.ctx != 0 {
		llamaFree(d.ctx)
	}
	if d.mdl != 0 {
		llamaModelFree(d.mdl)
	}
	d.batch, d.ctx, d.mdl = 0, 0, 0
}

// drafting is one slot's progress through the draft rounds.
type drafting struct {
	step  int // index into the steps
	seq   int
	pos   int // the position of the next token; every round uses it
	tok   backend.Token
	state []float32
	limit int
	out   []backend.Token
	idx   int32
}

// draft runs the draft rounds for the steps that may draft and returns the
// drafts per step. limits[i] is how many step i may draft (0 for none); the
// caller has already fitted them to the batch and output budgets. A failed
// drafter decode ends drafting for the step with what it has: drafts are an
// optimization, never a reason to fail the step.
func (m *model) draft(steps []backend.SeqStep, limits []int) [][]backend.Token {
	out := make([][]backend.Token, len(steps))
	d := m.dft
	var active []*drafting
	for i, st := range steps {
		if limits[i] <= 0 {
			continue
		}
		s := &m.slots[st.Seq]
		active = append(active, &drafting{
			step: i, seq: st.Seq, pos: st.Pos, tok: st.Tokens[0],
			state: append([]float32(nil), s.pendingH...), limit: limits[i],
		})
	}
	for len(active) > 0 {
		llamaBatchExtClear(d.batch)
		var keep []any
		for _, a := range active {
			a.idx = llamaBatchExtAddToken(d.batch, int32(a.seq), int32(a.tok))
			pos := int32(a.pos)
			if a.idx < 0 || !llamaBatchExtSetPos(d.batch, a.idx, &pos) ||
				!llamaBatchExtSetEmbdState(d.batch, a.idx, LlamaEmbd{Data: &a.state[0], NRows: 1, NEmbd: uintptr(d.nEmbd)}) {
				slog.Warn("llamacpp: drafter batch rejected an entry; drafting stops this step", "slot", a.seq)
				return out
			}
			llamaBatchExtSetOutputLogits(d.batch, a.idx, true)
			keep = append(keep, a.state)
		}
		rc := llamaProcess(d.ctx, LlamaProcessTypeDecode, d.batch)
		runtime.KeepAlive(keep)
		if rc != 0 {
			slog.Warn("llamacpp: drafter decode failed; drafting stops this step", "rc", rc)
			return out
		}
		var next []*drafting
		for _, a := range active {
			p := llamaGetLogitsIth(d.ctx, a.idx)
			h := llamaGetEmbeddingsNextnIth(d.ctx, a.idx)
			if p == nil || h == nil {
				continue
			}
			tok, prob := topOfTopK(unsafe.Slice(p, d.nVocab), draftTopK)
			if prob < d.pMin || int(tok) >= m.nVocab {
				continue
			}
			a.out = append(a.out, tok)
			out[a.step] = a.out
			if len(a.out) >= a.limit {
				continue
			}
			a.tok = tok
			copy(a.state, unsafe.Slice(h, d.nEmbd))
			next = append(next, a)
		}
		active = next
	}
	return out
}

// topOfTopK is the most likely token and its probability under a softmax
// over the k largest logits — what llama.cpp's MTP draft sampler (top-k 10,
// then the distribution) reports as the candidate's p.
func topOfTopK(row []float32, k int) (backend.Token, float64) {
	top := make([]float32, 0, k)
	best, bestTok := float32(math.Inf(-1)), 0
	mi := 0 // index of the smallest kept logit, once top is full
	for i, v := range row {
		if v > best {
			best, bestTok = v, i
		}
		if len(top) < k {
			top = append(top, v)
			if len(top) == k {
				mi = minIndex(top)
			}
			continue
		}
		if v > top[mi] {
			top[mi] = v
			mi = minIndex(top)
		}
	}
	var sum float64
	for _, v := range top {
		sum += math.Exp(float64(v - best))
	}
	return backend.Token(bestTok), 1 / sum
}

func minIndex(v []float32) int {
	mi := 0
	for j := 1; j < len(v); j++ {
		if v[j] < v[mi] {
			mi = j
		}
	}
	return mi
}
