//go:build native

// Tests of backend.Tuner and backend.BatchTimer on a real model: what the
// autotuner (docs/OPTIMIZE.md, §3.3) changes on a loaded model must give the
// results of a load made with that setting. They need a model and its drafter
// (see native_mtp_test.go).
package llamacpp

import (
	"context"
	"math"
	"os"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
)

// tunerPrompts is how many of the equivalence prompts a run generates for: the
// macOS runner decodes the tiny model at 1-3 tok/s and the native suite has the
// 30 minutes of its timeout, so the short form is the default and
// NANOLLM_TEST_LONG=1 asks for the full one.
func tunerPrompts() int {
	if os.Getenv("NANOLLM_TEST_LONG") != "" {
		return 3
	}
	return 2
}

// tunerRun generates n tokens for each of the first prompts and returns the
// tokens and the decode rate.
func tunerRun(t *testing.T, m *model, n int) ([][]backend.Token, float64) {
	t.Helper()
	var all [][]backend.Token
	var toks int
	var d time.Duration
	for _, p := range equivalencePrompts[:tunerPrompts()] {
		prompt := tokenize(t, m, chat(m, p))
		m.ClearSeq(0)
		if err := m.SetSampler(0, greedy, prompt); err != nil {
			t.Fatal(err)
		}
		g := generate(t, m, 0, prompt, 0, n, -1)
		all = append(all, g.toks)
		toks += len(g.toks)
		d += g.decode
	}
	return all, float64(toks) / d.Seconds()
}

// TestNativeTunerEquivalence: SetDraft on a model loaded with a larger
// draft_max gives the tokens (and, with NANOLLM_TEST_LONG, the speed) of a
// model loaded with that draft_max; SetThreads changes nothing but the speed;
// SetDraft above the loaded limit is refused.
func TestNativeTunerEquivalence(t *testing.T) {
	long := os.Getenv("NANOLLM_TEST_LONG") != ""
	n := 24
	if long {
		n = 64
	}
	big := sharedMTPModel(t)
	if big.DraftCeiling() != 3 {
		t.Fatalf("draft ceiling %d, loaded with 3", big.DraftCeiling())
	}
	if err := big.SetDraft(4, 0); err == nil {
		t.Error("SetDraft above the loaded draft_max was accepted")
	}
	if err := big.SetDraft(2, 1.5); err == nil {
		t.Error("SetDraft with p_min 1.5 was accepted")
	}
	tunerRun(t, big, 16) // warm up
	if err := big.SetDraft(2, 0); err != nil {
		t.Fatal(err)
	}
	if got := big.Capabilities().MaxDraft; got != 2 {
		t.Fatalf("Capabilities().MaxDraft %d after SetDraft(2)", got)
	}
	lowered, lowRate := tunerRun(t, big, n)

	// Threads: the tokens do not move; the rate returns when the team does.
	threads := big.spec.Threads
	base, baseRate := lowered, lowRate
	if err := big.SetThreads(max(threads/2, 1), max(threads/2, 1)); err != nil {
		t.Fatal(err)
	}
	half, halfRate := tunerRun(t, big, n)
	if err := big.SetThreads(threads, threads); err != nil {
		t.Fatal(err)
	}
	back, backRate := tunerRun(t, big, n)
	t.Logf("draft 2 of 3: %.1f tok/s; %d threads %.1f; back to %d threads %.1f", lowRate, max(threads/2, 1), halfRate, threads, backRate)
	for i := range base {
		if !slices.Equal(base[i], half[i]) || !slices.Equal(base[i], back[i]) {
			t.Errorf("prompt %d: SetThreads changed the tokens", i)
		}
	}
	if long && math.Abs(backRate-baseRate)/baseRate > 0.05 {
		t.Errorf("rate after SetThreads back: %.1f against %.1f before (more than 5%%)", backRate, baseRate)
	}

	// Drafting off through SetDraft(0) gives the plain model's tokens. On a GPU
	// they may part where the model hesitates between two words (the verified
	// batch runs other kernels than a batch of one), never anywhere else: the
	// autotuner's own check allows a gap of NearTieNats (internal/tune).
	if err := big.SetDraft(0, 0); err != nil {
		t.Fatal(err)
	}
	off, _ := tunerRun(t, big, n)
	for i := range base {
		if slices.Equal(base[i], off[i]) {
			continue
		}
		if testOnCPU() {
			t.Errorf("prompt %d: drafting off changed the tokens", i)
			continue
		}
		gap := divergenceGap(t, big, equivalencePrompts[i], base[i], n)
		t.Logf("prompt %d: drafting and plain decoding part at a gap of %.4f nats", i, gap)
		if gap < 0 || gap > 0.1 {
			t.Errorf("prompt %d: drafting off changed the tokens at more than a near tie (gap %.4f)", i, gap)
		}
	}
	if err := big.SetDraft(3, 0); err != nil {
		t.Fatal(err)
	}
	releaseSharedMTP()

	// The same limit set at load.
	b := testBackend(t)
	spec := mtpSpec(testModelPath(t), testDrafterPath(t))
	spec.DraftMax = 2
	spec.NOutputsMax = spec.Slots * (1 + spec.DraftMax)
	mm, err := b.Load(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	loaded := mm.(*model)
	defer loaded.Close()
	tunerRun(t, loaded, 16)
	fresh, freshRate := tunerRun(t, loaded, n)
	t.Logf("draft 2 loaded: %.1f tok/s, draft 2 set on a load of 3: %.1f", freshRate, lowRate)
	for i := range base {
		if !slices.Equal(base[i], fresh[i]) {
			t.Errorf("prompt %d: a model loaded with draft_max 2 gives other tokens than SetDraft(2) on one loaded with 3", i)
		}
	}
	if long && math.Abs(lowRate-freshRate)/freshRate > 0.03 {
		t.Errorf("SetDraft(2) rate %.1f against %.1f loaded with 2 (more than 3%%)", lowRate, freshRate)
	}
	_ = back
}

// divergenceGap generates n plain tokens and, where they part from with, the
// distance in nats between the plain token and with's at that position; -1 when
// they do not part or with's token is not among the alternatives.
func divergenceGap(t *testing.T, m *model, text string, with []backend.Token, n int) float64 {
	t.Helper()
	prompt := tokenize(t, m, chat(m, text))
	m.ClearSeq(0)
	if err := m.SetSampler(0, greedy, prompt); err != nil {
		t.Fatal(err)
	}
	g := generate(t, m, 0, prompt, 0, n, 5)
	for i := range min(len(g.toks), len(with)) {
		if g.toks[i] == with[i] {
			continue
		}
		for _, alt := range g.lps[i].Top {
			if alt.Token == with[i] {
				return g.lps[i].Logprob - alt.Logprob
			}
		}
		return -1
	}
	return -1
}

// TestNativeTimeBatch: the cost of a batch grows with its size, and the cache
// is the one it was before.
func TestNativeTimeBatch(t *testing.T) {
	m := sharedMTPModel(t)
	prompt := tokenize(t, m, chat(m, equivalencePrompts[0]))
	m.ClearSeq(0)
	if err := m.SetSampler(0, greedy, prompt); err != nil {
		t.Fatal(err)
	}
	before := generate(t, m, 0, prompt, 0, 8, -1)
	pos := len(prompt) + len(before.toks) - 1
	if got := int(llamaMemorySeqPosMax(m.mem, 0)) + 1; got != pos {
		t.Fatalf("cache holds %d positions, expected %d", got, pos)
	}
	filler := make([]backend.Token, 4)
	for i := range filler {
		filler[i] = prompt[len(prompt)-1-i]
	}
	cost := map[int]time.Duration{}
	for _, k := range []int{1, 2, 3, 4} {
		var ds []time.Duration
		for range 7 {
			d, err := m.TimeBatch(context.Background(), 0, filler[:k], k)
			if err != nil {
				t.Fatalf("batch of %d: %v", k, err)
			}
			ds = append(ds, d)
		}
		sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
		cost[k] = ds[len(ds)/2]
		if got := int(llamaMemorySeqPosMax(m.mem, 0)) + 1; got != pos {
			t.Fatalf("after a batch of %d the cache holds %d positions, expected %d", k, got, pos)
		}
	}
	t.Logf("median batch cost: 1 token %v, 2 %v (c=%.2f), 3 %v (c=%.2f), 4 %v (c=%.2f)", cost[1],
		cost[2], float64(cost[2])/float64(cost[1]), cost[3], float64(cost[3])/float64(cost[1]), cost[4], float64(cost[4])/float64(cost[1]))
	if cost[1] <= 0 || cost[4] < cost[1]*9/10 {
		t.Errorf("a batch of 4 (%v) is cheaper than a batch of 1 (%v)", cost[4], cost[1])
	}

	// Generation continues from the same state: after the prompt and one
	// token, a timed batch leaves the cache as it was, and the next tokens
	// equal those of a generation that was never timed.
	m.ClearSeq(0)
	if err := m.SetSampler(0, greedy, prompt); err != nil {
		t.Fatal(err)
	}
	ref := generate(t, m, 0, prompt, 0, 25, -1)
	m.ClearSeq(0)
	if err := m.SetSampler(0, greedy, prompt); err != nil {
		t.Fatal(err)
	}
	out, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: prompt, Pos: 0, Sample: true}})
	if err != nil {
		t.Fatal(err)
	}
	if first := out[0].Tokens[0]; first != ref.toks[0] {
		t.Fatalf("first token %d, reference %d", first, ref.toks[0])
	}
	if _, err := m.TimeBatch(context.Background(), 0, filler[:3], 3); err != nil {
		t.Fatal(err)
	}
	cont := generate(t, m, 0, ref.toks[:1], len(prompt), 24, -1)
	if !slices.Equal(cont.toks, ref.toks[1:25]) {
		t.Errorf("generation after a timed batch differs:\n got %v\nwant %v", cont.toks, ref.toks[1:25])
	}
	if _, err := m.TimeBatch(context.Background(), 0, nil, 0); err == nil {
		t.Error("an empty batch was accepted")
	}
}
