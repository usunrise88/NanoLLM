//go:build native

package llamacpp

import (
	"context"
	"math"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
)

// TestNativeBatchInvariance evaluates the greedy continuation of a few prompts one token at a time and in batches of
// 2, 3 and 4 tokens (the shapes of a draft's verification) and compares the top log-probabilities after each batch with
// those after the same token evaluated alone. Speculative decoding promises the tokens plain decoding would give; kernels
// that add up a row in another order for another batch size make the two differ by rounding, which the model's quantized
// activations amplify (an int8 value that flips). The largest difference is logged; the test fails when the top
// alternatives change or a log-probability moves by more than 0.1, which two correct kernels do not do.
func TestNativeBatchInvariance(t *testing.T) {
	m := sharedModel(t)
	const n = 48
	for _, ask := range []string{
		"List five prime numbers and say why they are prime.",
		"Write a short paragraph about the history of Paris.",
	} {
		prompt := tokenize(t, m, chat(m, ask))
		m.ClearSeq(0)
		if err := m.SetSampler(0, greedy, prompt); err != nil {
			t.Fatal(err)
		}
		cont := generate(t, m, 0, prompt, 0, n+2, -1).toks[:n]

		step := func(toks []backend.Token, pos int) []backend.TokenLogprob {
			t.Helper()
			out, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: toks, Pos: pos, Sample: true, Logprobs: 3}})
			if err != nil {
				t.Fatal(err)
			}
			return out[0].Logprobs[0].Top
		}
		prefill := func() {
			t.Helper()
			m.ClearSeq(0)
			if err := m.SetSampler(0, greedy, prompt); err != nil {
				t.Fatal(err)
			}
			if _, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: prompt, Pos: 0}}); err != nil {
				t.Fatal(err)
			}
		}
		prefill()
		alone := make([][]backend.TokenLogprob, n)
		for i, tok := range cont {
			alone[i] = step([]backend.Token{tok}, len(prompt)+i)
		}
		for _, b := range []int{2, 3, 4} {
			prefill()
			worst := 0.0
			for start := 0; start+b <= n; start += b {
				got := step(cont[start:start+b], len(prompt)+start)
				want := alone[start+b-1]
				for k := range want {
					if want[k].Token != got[k].Token {
						t.Errorf("%q, batches of %d: after token %d alternative %d is %d alone, %d in the batch", ask, b, start+b, k, want[k].Token, got[k].Token)
						continue
					}
					d := math.Abs(want[k].Logprob - got[k].Logprob)
					worst = math.Max(worst, d)
					if d > 0.1 {
						t.Errorf("%q, batches of %d: after token %d, token %d has log-probability %.4f alone, %.4f in the batch", ask, b, start+b, want[k].Token, want[k].Logprob, got[k].Logprob)
					}
				}
			}
			t.Logf("%q, batches of %d: largest difference %g", ask, b, worst)
		}
	}
}

// TestNativeCudaNano: the host program reaches the switches of NanoLLM's ggml-cuda through the registry, by name (case
// does not matter), and an unknown name is refused. Skipped with upstream's ggml-cuda or without a GPU.
func TestNativeCudaNano(t *testing.T) {
	b := testBackend(t)
	devs, err := b.Devices(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ours := false
	for _, d := range devs {
		ours = ours || d.Has(backend.FeatureCudaNano)
	}
	if !ours {
		t.Skip("no ggml-cuda of ours (device feature cuda-nano)")
	}
	if !setCudaNano("MMVQ_NWARPS", 0) || !setCudaNano("mmvq_nwarps", -1) {
		t.Error("a known switch was refused")
	}
	if setCudaNano("NO_SUCH_SWITCH", 1) {
		t.Error("an unknown switch was accepted")
	}
	applyCudaNano(map[string]int{"GRAPH_SHAPES": 1, "NO_SUCH_SWITCH": 1}) // logs the unknown one, does not fail
}
