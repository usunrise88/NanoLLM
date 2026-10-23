package llamacpp

import (
	"fmt"
	"sync/atomic"

	"github.com/usunrise88/nanollm/internal/backend"
)

// The output shortlist (OPT-4, docs/AUTOTUNE.md; patches/llama.cpp/0009). The
// head of the model reads a matrix of 262144 rows on every token, and the
// tokens a text uses lie in a part of it: LoadSpec.HeadShortlist names the
// rows to keep, the library makes a second, smaller (and repacked) copy of
// them, and a context with the shortlist on computes only those logits and
// writes -inf for the rest. A request that could use any token gets the whole
// matrix: the library is told per batch which one the batch needs, and a
// change costs the scheduler a new reservation, so the choice follows the
// request and does not flicker.

// HeadStats counts the batches that sampled with each head.
type HeadStats struct {
	Short, Full, Switches int64
	// Rows is the size of the shortlist.
	Rows int
}

// headState is the model's shortlist, nil when there is none.
type headState struct {
	rows   int
	inList []bool // by vocabulary id

	on                    bool // the context currently computes the shortlist only
	short, full, switches atomic.Int64
}

// attachHead gives the loaded model its shortlist. It runs after the model is
// loaded and before the context is created; a library without the functions,
// or a model whose head cannot carry one, loads as it was, with the reason in
// the load log.
func (m *model) attachHead(ids []int32) {
	note := func(format string, a ...any) {
		m.loadLog = append(m.loadLog, "nanollm: output shortlist "+fmt.Sprintf(format, a...))
	}
	if !m.b.lib.cxx["llama_model_set_output_shortlist"] || !m.b.lib.cxx["llama_set_output_shortlist"] {
		note("is off: the library has no shortlist (patches/llama.cpp/0009)")
		return
	}
	nVocab := int(llamaVocabNTokens(llamaModelGetVocab(m.mdl)))
	in := make([]bool, nVocab)
	for _, id := range ids {
		if id < 0 || int(id) >= nVocab {
			note("is off: id %d is outside the vocabulary of %d", id, nVocab)
			return
		}
		in[id] = true
	}
	if !llamaModelSetOutputShortlist(m.mdl, &ids[0], int32(len(ids))) {
		note("is off: the library refused the %d ids (the head must be a tied Q4_0 matrix in host memory)", len(ids))
		return
	}
	m.head = &headState{rows: len(ids), inList: in}
	note("is on: %d of %d rows", len(ids), nVocab)
}

// chooseHead picks the head for a batch and tells the context. A batch that
// samples nothing keeps the head it has. The whole vocabulary is needed when
//   - a slot reports logprobs, or reads raw logits (the alternatives of a
//     position can be any token);
//   - a slot's sampler can move the choice outside the list: a grammar, or a
//     logit bias that raises a token the list leaves out;
//   - a slot's context holds a token outside the list: such a prompt is in a
//     language or a script the list did not cover, and its answer probably is
//     too.
//
// The last is tracked per slot and cleared when the slot's cache is.
func (m *model) chooseHead(steps []backend.SeqStep) {
	h := m.head
	if h == nil {
		return
	}
	sampling, full := false, false
	for _, s := range steps {
		sl := &m.slots[s.Seq]
		if !sl.outside {
			for _, t := range s.Tokens {
				if t < 0 || int(t) >= len(h.inList) || !h.inList[t] {
					sl.outside = true
					break
				}
			}
		}
		if !s.Sample {
			continue
		}
		sampling = true
		if s.Logprobs >= 0 || sl.outside || sl.wholeVocab {
			full = true
		}
	}
	if !sampling {
		return
	}
	if want := !full; want != h.on {
		llamaSetOutputShortlist(m.ctx, want)
		h.on = want
		h.switches.Add(1)
	}
	if h.on {
		h.short.Add(1)
	} else {
		h.full.Add(1)
	}
}

// noteHistory marks a slot whose history holds a token outside the list.
func (m *model) noteHistory(sl *slot, toks []backend.Token) {
	h := m.head
	if h == nil || sl.outside {
		return
	}
	for _, t := range toks {
		if t < 0 || int(t) >= len(h.inList) || !h.inList[t] {
			sl.outside = true
			return
		}
	}
}

// wholeVocabulary says a sampler spec needs logits for any token: the caller
// reads raw logits, there is a grammar (it may force any token), or a logit
// bias raises a token the list leaves out. A bias that only lowers tokens (the
// benchmark bans the end-of-generation tokens) keeps them at -inf or lower
// either way.
func (m *model) wholeVocabulary(spec backend.SamplerSpec) bool {
	if spec.RawLogits || spec.Grammar != "" {
		return true
	}
	if m.head == nil {
		return false
	}
	for t, v := range spec.LogitBias {
		if v > 0 && (t < 0 || int(t) >= len(m.head.inList) || !m.head.inList[t]) {
			return true
		}
	}
	return false
}

// HeadStats is what the head has done so far; the zero value without a shortlist.
func (m *model) HeadStats() HeadStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.head == nil {
		return HeadStats{}
	}
	return HeadStats{Short: m.head.short.Load(), Full: m.head.full.Load(), Switches: m.head.switches.Load(), Rows: m.head.rows}
}
