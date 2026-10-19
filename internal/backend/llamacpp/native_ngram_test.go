//go:build native

package llamacpp

import (
	"context"
	"slices"
	"testing"
)

// OPT-5: prompt lookup drafts from the sequence and the target still
// samples every position, so greedy output is token for token the one
// without drafting.
func TestNativeNGramMatchesPlainDecoding(t *testing.T) {
	b := testBackend(t)
	path := testModelPath(t)
	load := func(ngram int) *model {
		spec := baseSpec(path)
		spec.Slots = 2
		if ngram > 0 {
			spec.NGram, spec.DraftMax = ngram, 4
			spec.NOutputsMax = spec.Slots * 5
		}
		mm, err := b.Load(context.Background(), spec)
		if err != nil {
			t.Fatal(err)
		}
		return mm.(*model)
	}
	plain, withNG := load(0), load(3)
	defer plain.Close()
	defer withNG.Close()
	if withNG.caps.MaxDraft != 4 || plain.caps.MaxDraft != 0 {
		t.Fatalf("MaxDraft with lookup %d, without %d", withNG.caps.MaxDraft, plain.caps.MaxDraft)
	}

	// A task whose answer repeats its input.
	text := "Repeat the following list exactly, once: alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima. Then repeat it once more."
	prompt := tokenize(t, plain, chat(plain, text))

	const n = 80
	want := decode(t, plain, 0, prompt, n, greedy)
	withNG.ClearSeq(0)
	if err := withNG.SetSampler(0, greedy, prompt); err != nil {
		t.Fatal(err)
	}
	got := generate(t, withNG, 0, prompt, 0, n, -1)
	if !slices.Equal(got.toks, want) {
		t.Fatalf("lookup changed the output:\n got %v\nwant %v", got.toks, want)
	}
	t.Logf("drafted %d, accepted %d in %d steps for %d tokens", got.drafted, got.accepted, got.steps, len(got.toks))
	if got.drafted == 0 || got.accepted == 0 {
		t.Errorf("a repeating answer drafted %d and accepted %d", got.drafted, got.accepted)
	}
	if got.steps >= n-1 {
		t.Errorf("%d steps for %d tokens: nothing was gained", got.steps, n)
	}

	// The history follows a trim: back to the 19th generated token, which is
	// fed again at its position, the continuation is still exact.
	cut := len(prompt) + 19
	if !withNG.TrimSeq(0, cut) || !plain.TrimSeq(0, cut) {
		t.Fatal("trim refused")
	}
	again := generate(t, withNG, 0, want[19:20], cut, 30, -1)
	ref := generate(t, plain, 0, want[19:20], cut, 30, -1)
	if !slices.Equal(again.toks, ref.toks) {
		t.Errorf("after a trim:\n got %v\nwant %v", again.toks, ref.toks)
	}

	// A restored state has no history: no drafts, still exact.
	st, err := withNG.SaveSeq(0)
	if err != nil {
		t.Fatal(err)
	}
	withNG.ClearSeq(1)
	if err := withNG.SetSampler(1, greedy, nil); err != nil {
		t.Fatal(err)
	}
	if err := withNG.RestoreSeq(1, st); err != nil {
		t.Fatal(err)
	}
	if !withNG.slots[1].histBad {
		t.Error("a restored slot trusts a history it never had")
	}
}
