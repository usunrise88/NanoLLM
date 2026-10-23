package llamacpp

import (
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
)

// A model with a shortlist of the tokens 0..9 out of 16.
func shortModel() *model {
	in := make([]bool, 16)
	for i := range 10 {
		in[i] = true
	}
	return &model{head: &headState{rows: 10, inList: in}, slots: make([]slot, 2)}
}

func TestWholeVocabulary(t *testing.T) {
	m := shortModel()
	for name, c := range map[string]struct {
		spec backend.SamplerSpec
		want bool
	}{
		"greedy":                      {backend.SamplerSpec{}, false},
		"temperature":                 {backend.SamplerSpec{Temperature: 0.7, TopK: 40, TopP: 0.9}, false},
		"raw logits":                  {backend.SamplerSpec{RawLogits: true}, true},
		"a grammar":                   {backend.SamplerSpec{Grammar: `root ::= "a"`}, true},
		"a ban on a listed token":     {backend.SamplerSpec{LogitBias: map[backend.Token]float64{3: -100}}, false},
		"a ban on an unlisted token":  {backend.SamplerSpec{LogitBias: map[backend.Token]float64{12: -100}}, false},
		"a boost of a listed token":   {backend.SamplerSpec{LogitBias: map[backend.Token]float64{3: 5}}, false},
		"a boost of an unlisted one":  {backend.SamplerSpec{LogitBias: map[backend.Token]float64{12: 5}}, true},
		"a boost of a token past the": {backend.SamplerSpec{LogitBias: map[backend.Token]float64{99: 5}}, true},
	} {
		if got := m.wholeVocabulary(c.spec); got != c.want {
			t.Errorf("%s: %v, want %v", name, got, c.want)
		}
	}
	if (&model{}).wholeVocabulary(backend.SamplerSpec{LogitBias: map[backend.Token]float64{12: 5}}) {
		t.Error("a model without a shortlist asks for nothing but raw logits and grammars")
	}
}

func TestNoteHistory(t *testing.T) {
	m := shortModel()
	sl := &m.slots[0]
	m.noteHistory(sl, []backend.Token{1, 2, 9})
	if sl.outside {
		t.Error("listed tokens only")
	}
	m.noteHistory(sl, []backend.Token{3, 10})
	if !sl.outside {
		t.Error("token 10 is not listed")
	}
	// A cleared slot is trusted again, and a model without a shortlist never marks anything.
	m.resetHistory(0)
	if sl.outside {
		t.Error("the slot was cleared")
	}
	(&model{slots: make([]slot, 1)}).noteHistory(&slot{}, []backend.Token{1000000})
	m.noteHistory(&m.slots[1], []backend.Token{-1})
	if !m.slots[1].outside {
		t.Error("a negative token id is not listed")
	}
}
