//go:build native

package llamacpp

import (
	"context"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
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

// margin is how far the best token's log-probability is above the runner-up's.
func margin(lp backend.TokenLogprobs) float64 {
	top := slices.Clone(lp.Top)
	slices.SortFunc(top, func(a, b backend.TokenLogprob) int {
		switch {
		case a.Logprob > b.Logprob:
			return -1
		case a.Logprob < b.Logprob:
			return 1
		}
		return 0
	})
	if len(top) < 2 {
		return math.Inf(1)
	}
	return top[0].Logprob - top[1].Logprob
}

// Verifying drafts runs the target on several rows at once, and ggml's kernels
// are not bit-identical across batch shapes (a row of a gemm against the same
// row alone, the split-KV attention of a single row against the tiled one of
// several). So greedy output with drafts can differ from the output without,
// but only where the two best tokens are a tie within that noise. This test
// measures both: the largest difference in a token's log-probability between
// the two runs, and, at the first token that differs, how close the call was.
// (Measured: no difference up to a context of ~256 tokens, up to ~0.07 nats
// beyond; docs/AUTOTUNE.md OPT-5.)
// A difference at a clear margin is a bug in the drafting. Long: it generates
// a few hundred tokens per prompt twice.
func TestNativeNGramBatchNoise(t *testing.T) {
	if os.Getenv("NANOLLM_TEST_LONG") == "" {
		t.Skip("set NANOLLM_TEST_LONG=1 (a few hundred tokens per prompt, twice)")
	}
	b := testBackend(t)
	path := testModelPath(t)
	load := func(ngram int) *model {
		spec := baseSpec(path)
		spec.Slots = 1
		if ngram > 0 {
			spec.NGram, spec.DraftMax = ngram, 4
			spec.NOutputsMax = 5
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

	think := func(user string) string {
		return "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n" + user + "<turn|>\n<|turn>model\n"
	}
	cases := []struct {
		name, text string
		n          int
	}{
		// The item of the eval that differed with drafts (ru-instruct, thinking).
		{"ru-no-latin, thinking", think("Объясни в двух предложениях, что такое фотосинтез. Не используй латинские буквы."), 450},
		{"repeat the list, thinking", think("Repeat this list three times, once per line: alpha bravo charlie delta echo foxtrot golf hotel india juliet."), 450},
		{"ru story", chat(plain, "Напиши рассказ из шести предложений о маяке и о человеке, который его охраняет."), 300},
		{"en explanation", chat(plain, "Explain how a hash map handles collisions, with a small example in Go."), 300},
		{"json", chat(plain, `Return a JSON array of five objects with the keys "id", "name" and "tags"; tags is a list of two short strings.`), 300},
	}
	var worst float64
	for _, c := range cases {
		prompt := tokenize(t, plain, c.text)
		plain.ClearSeq(0)
		if err := plain.SetSampler(0, greedy, prompt); err != nil {
			t.Fatal(err)
		}
		want := generate(t, plain, 0, prompt, 0, c.n, 2)
		withNG.ClearSeq(0)
		if err := withNG.SetSampler(0, greedy, prompt); err != nil {
			t.Fatal(err)
		}
		got := generate(t, withNG, 0, prompt, 0, c.n, 2)

		n := min(len(want.toks), len(got.toks))
		first := -1
		for i := range n {
			if want.toks[i] != got.toks[i] {
				first = i
				break
			}
		}
		same := n
		if first >= 0 {
			same = first
		}
		var maxd float64
		for i := 0; i < same && i < len(want.lps) && i < len(got.lps); i++ {
			maxd = math.Max(maxd, math.Abs(want.lps[i].Logprob-got.lps[i].Logprob))
		}
		worst = math.Max(worst, maxd)
		t.Logf("%s: %d tokens, drafted %d accepted %d in %d steps; log-probability differs by at most %.2g over %d equal tokens",
			c.name, len(want.toks), got.drafted, got.accepted, got.steps, maxd, same)
		if first < 0 {
			if len(want.toks) != len(got.toks) {
				t.Errorf("%s: lengths %d and %d", c.name, len(want.toks), len(got.toks))
			}
			continue
		}
		m := margin(want.lps[first])
		t.Logf("%s: first difference at token %d: without drafts %d (margin to the runner-up %.3g), with drafts %d",
			c.name, first, want.toks[first], m, got.toks[first])
		// A flip is a tie within the noise. Measured on hyper, the noise is
		// 0 up to a context of about 256 tokens and up to ~0.07 nats beyond
		// (the single-row decode takes ggml's split-KV attention from a KV of
		// 512 cells, the verify batch the tiled one), so a margin above three
		// times the noise seen in this run is not a tie.
		if m > math.Max(0.01, 3*maxd) {
			t.Errorf("%s: differs at token %d with a clear margin of %.3g (noise %.2g): the drafting changed the output", c.name, first, m, maxd)
		}
	}
	t.Logf("largest log-probability difference over all prompts: %.3g", worst)
}
