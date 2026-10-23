package llamacpp

import (
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/ngram"
)

// Prompt-lookup drafting (OPT-5, internal/ngram): a decode step whose slot
// has a token history proposes the continuation of an earlier occurrence of
// its last tokens. The drafts ride in the batch exactly as MTP's do and are
// verified by the same code, so the answer is the one the target would give
// without them.

// resetHistory forgets a slot's history: it is empty and trusted again.
func (m *model) resetHistory(seq int) {
	sl := &m.slots[seq]
	sl.hist, sl.histBad = sl.hist[:0], false
	sl.ng = ngram.Backoff{}
	sl.outside = false
}

// noteTokens records the inputs a step put in the slot's cache: the step's
// tokens, then the drafts that were accepted. A step that does not continue
// the history (a gap, or a position inside it) is the caller moving the
// cache some other way; inside the history it is a rewrite, beyond it the
// history is lost.
func (m *model) noteTokens(s backend.SeqStep, drafts []backend.Token, accepted int) {
	if m.ngN == 0 {
		return
	}
	sl := &m.slots[s.Seq]
	if sl.histBad {
		return
	}
	switch {
	case s.Pos == len(sl.hist):
	case s.Pos < len(sl.hist):
		sl.hist = sl.hist[:s.Pos]
	default:
		sl.hist, sl.histBad = nil, true
		return
	}
	sl.hist = append(sl.hist, s.Tokens...)
	if accepted > 0 {
		sl.hist = append(sl.hist, drafts[:accepted]...)
	}
}

// lookup proposes drafts for the steps that may draft (limits from
// draftLimits). nil when no step got one.
func (m *model) lookup(steps []backend.SeqStep, limits []int) [][]backend.Token {
	var drafts [][]backend.Token
	for i, s := range steps {
		if limits[i] <= 0 {
			continue
		}
		sl := &m.slots[s.Seq]
		if sl.histBad || len(sl.hist) != s.Pos || !sl.ng.Allow() {
			continue
		}
		d := ngram.Propose(sl.hist, s.Tokens[0], m.ngN, limits[i])
		if len(d) == 0 {
			continue
		}
		if drafts == nil {
			drafts = make([][]backend.Token, len(steps))
		}
		drafts[i] = d
	}
	return drafts
}
