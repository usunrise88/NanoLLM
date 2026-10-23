//go:build native

package llamacpp

import (
	"context"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/registry"
)

// OPT-4: with the output shortlist on, greedy decoding gives the tokens the
// whole head gives (the argmax is in the list in 99.93 % of positions), a batch
// that could use any token falls back to the whole head, and the head's
// statistics say which was used. Needs the libraries of the series with patch
// 0009 (the test skips on older ones).
func TestNativeOutputShortlist(t *testing.T) {
	if testOnCPU() == false {
		t.Skip("the shortlist is a CPU head")
	}
	b := testBackend(t)
	if !b.lib.cxx["llama_model_set_output_shortlist"] || !b.lib.cxx["llama_set_output_shortlist"] {
		t.Skip("the libraries have no output shortlist (patches/llama.cpp/0009)")
	}
	path := testModelPath(t)
	sl, err := registry.LoadShortlist("gemma-4-e4b-qat", registry.HeadShortlist, "")
	if err != nil {
		t.Fatal(err)
	}
	load := func(ids []int32) *model {
		s := baseSpec(path)
		s.Slots = 1
		s.HeadShortlist = ids
		mm, err := b.Load(context.Background(), s)
		if err != nil {
			t.Fatal(err)
		}
		return mm.(*model)
	}
	full := load(nil)
	defer full.Close()
	short := load(sl.IDs)
	defer short.Close()
	if st := short.HeadStats(); st.Rows != len(sl.IDs) {
		t.Fatalf("the shortlist did not attach: %+v; log %v", st, short.loadLog[max(0, len(short.loadLog)-3):])
	}

	prompts := map[string]string{
		"en":   "The three primary colours are",
		"ru":   "Назови три основных цвета. Ответ:",
		"code": "func add(a, b int) int {\n\treturn",
		"json": `{"name": "Alice", "age": 30, "city":`,
	}
	for name, text := range prompts {
		want := decode(t, full, 0, tokenize(t, full, text), 24, greedy)
		got := decode(t, short, 0, tokenize(t, short, text), 24, greedy)
		if !slices.Equal(got, want) {
			t.Errorf("%s: the shortlist changed the output\n got %v\nwant %v", name, got, want)
		}
	}
	st := short.HeadStats()
	if st.Short == 0 || st.Full != 0 {
		t.Errorf("ordinary prompts should run on the shortlist alone: %+v", st)
	}

	// A prompt with tokens outside the list: the answer may need any token, so the whole head runs, and it
	// gives exactly what the whole head gives.
	rare := "日本語のテキストです。中文，한국어, العربية, हिन्दी — translate these words to English."
	outside := 0
	for _, tok := range tokenize(t, short, rare) {
		if !short.head.inList[tok] {
			outside++
		}
	}
	if outside == 0 {
		t.Fatal("the rare-script prompt holds no token outside the list; the test proves nothing")
	}
	before := short.HeadStats()
	want := decode(t, full, 0, tokenize(t, full, rare), 16, greedy)
	got := decode(t, short, 0, tokenize(t, short, rare), 16, greedy)
	after := short.HeadStats()
	if !slices.Equal(got, want) {
		t.Errorf("rare script: the fallback changed the output\n got %v\nwant %v", got, want)
	}
	if after.Full == before.Full {
		t.Errorf("a prompt with tokens outside the list should use the whole head: %+v -> %+v", before, after)
	}
	// ...and a slot that has been cleared is trusted again.
	short.ClearSeq(0)
	mid := short.HeadStats()
	decode(t, short, 0, tokenize(t, short, prompts["en"]), 4, greedy)
	if end := short.HeadStats(); end.Full != mid.Full || end.Short == mid.Short {
		t.Errorf("after a clear the slot should be back on the shortlist: %+v -> %+v", mid, end)
	}

	// Logprobs ask for alternatives anywhere in the vocabulary: the whole head, the same numbers.
	step := func(m *model) backend.SeqOut {
		t.Helper()
		m.ClearSeq(0)
		toks := tokenize(t, m, prompts["en"])
		if err := m.SetSampler(0, greedy, toks); err != nil {
			t.Fatal(err)
		}
		outs, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: toks, Pos: 0, Sample: true, Logprobs: 5}})
		if err != nil {
			t.Fatal(err)
		}
		return outs[0]
	}
	before = short.HeadStats()
	a, c := step(full), step(short)
	after = short.HeadStats()
	if after.Full != before.Full+1 {
		t.Errorf("a step with logprobs should use the whole head: %+v -> %+v", before, after)
	}
	if len(a.Logprobs) != 1 || len(c.Logprobs) != 1 || a.Logprobs[0].Token != c.Logprobs[0].Token || a.Logprobs[0].Logprob != c.Logprobs[0].Logprob {
		t.Errorf("logprobs differ: %+v against %+v", c.Logprobs, a.Logprobs)
	}
}

// A grammar can force a token the list does not hold (any character of a JSON
// string), so a slot with one runs on the whole head.
func TestNativeOutputShortlistGrammar(t *testing.T) {
	if testOnCPU() == false {
		t.Skip("the shortlist is a CPU head")
	}
	b := testBackend(t)
	if !b.lib.cxx["llama_model_set_output_shortlist"] || !b.lib.cxx["llama_set_output_shortlist"] {
		t.Skip("the libraries have no output shortlist (patches/llama.cpp/0009)")
	}
	path := testModelPath(t)
	sl, err := registry.LoadShortlist("gemma-4-e4b-qat", registry.HeadShortlist, "")
	if err != nil {
		t.Fatal(err)
	}
	s := baseSpec(path)
	s.Slots = 1
	s.HeadShortlist = sl.IDs
	mm, err := b.Load(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	m := mm.(*model)
	defer m.Close()
	sp := greedy
	sp.Grammar = `root ::= "{" [a-z]+ "}"`
	decode(t, m, 0, tokenize(t, m, "Write a word in braces: "), 6, sp)
	if st := m.HeadStats(); st.Full == 0 || st.Short != 0 {
		t.Errorf("a grammar should use the whole head: %+v", st)
	}
}

// logitsOf decodes toks in slot 0 with every position an output and returns
// the rows, on the shortlist or on the whole head.
func logitsOf(t *testing.T, m *model, toks []backend.Token, shortlist bool) [][]float32 {
	t.Helper()
	m.ClearSeq(0)
	llamaSetOutputShortlist(m.ctx, shortlist)
	defer llamaSetOutputShortlist(m.ctx, false)
	llamaBatchExtClear(m.batch)
	idxs := make([]int32, len(toks))
	for i, tok := range toks {
		idx := llamaBatchExtAddToken(m.batch, 0, int32(tok))
		p := int32(i)
		if !llamaBatchExtSetPos(m.batch, idx, &p) {
			t.Fatalf("position %d rejected", i)
		}
		llamaBatchExtSetOutputLogits(m.batch, idx, true)
		idxs[i] = idx
	}
	if rc := llamaProcess(m.ctx, LlamaProcessTypeDecode, m.batch); rc != 0 {
		t.Fatalf("decode failed: %d", rc)
	}
	rows := make([][]float32, len(toks))
	for i, idx := range idxs {
		row, err := m.logitsRow(idx)
		if err != nil {
			t.Fatal(err)
		}
		rows[i] = slices.Clone(row)
	}
	return rows
}

// The logits of the listed tokens are the whole head's, to the last bit (the
// copy of the rows goes through the same repacked kernels), every other token
// is -inf, and the argmax agrees wherever the whole head's is in the list.
func TestNativeOutputShortlistLogits(t *testing.T) {
	if !testOnCPU() {
		t.Skip("the shortlist is a CPU head")
	}
	b := testBackend(t)
	if !b.lib.cxx["llama_model_set_output_shortlist"] || !b.lib.cxx["llama_set_output_shortlist"] {
		t.Skip("the libraries have no output shortlist (patches/llama.cpp/0009)")
	}
	sl, err := registry.LoadShortlist("gemma-4-e4b-qat", registry.HeadShortlist, "")
	if err != nil {
		t.Fatal(err)
	}
	s := baseSpec(testModelPath(t))
	s.Slots = 1
	s.HeadShortlist = sl.IDs
	mm, err := b.Load(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	m := mm.(*model)
	defer m.Close()
	if m.head == nil {
		t.Fatalf("the shortlist did not attach: %v", m.loadLog[max(0, len(m.loadLog)-2):])
	}
	text := "The history of the printing press begins long before Gutenberg. История книгопечатания началась задолго до Гутенберга: " +
		"в Китае печатали с деревянных досок, в Корее переставляли металлические литеры. func main() { fmt.Println(\"hello\") } " +
		"Привет, мир! Das ist ein Satz auf Deutsch, et voilà une phrase en français."
	if raw, err := os.ReadFile("../../../.cache/kld/corpus.txt"); err == nil && len(raw) > 4000 {
		text = string(raw[1000:4000]) // a few hundred tokens of the corpus of the KLD gate, when it is built
	}
	toks := tokenize(t, m, text)
	if len(toks) > 400 {
		toks = toks[:400]
	}
	whole := logitsOf(t, m, toks, false)
	short := logitsOf(t, m, toks, true)
	agree, inList := 0, 0
	maxDiff := 0.0
	for i := range toks {
		sa := 0
		for id := range whole[i] {
			if !m.head.inList[id] {
				if !math.IsInf(float64(short[i][id]), -1) {
					t.Fatalf("position %d: token %d is not listed but has the logit %v", i, id, short[i][id])
				}
				continue
			}
			if d := math.Abs(float64(whole[i][id] - short[i][id])); d > maxDiff {
				maxDiff = d
			}
			if short[i][id] > short[i][sa] {
				sa = id
			}
		}
		// the whole head's argmax over all the tokens
		best := 0
		for id := range whole[i] {
			if whole[i][id] > whole[i][best] {
				best = id
			}
		}
		if m.head.inList[best] {
			inList++
			if sa != best {
				t.Errorf("position %d: the argmax %d is listed but the shortlist picks %d", i, best, sa)
			}
		}
		if sa == best {
			agree++
		}
	}
	t.Logf("%d positions: argmax agrees at %d, the whole head's argmax is listed at %d; largest difference of a listed logit %g", len(toks), agree, inList, maxDiff)
	if maxDiff > 1e-4 {
		t.Errorf("listed logits differ from the whole head's by %g", maxDiff)
	}
	if agree < len(toks)*99/100 {
		t.Errorf("only %d of %d positions agree", agree, len(toks))
	}
}
