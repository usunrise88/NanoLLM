//go:build native

// MTP tests (STAGE1 1.5.M). Besides the target model they need its drafter:
// NANOLLM_TEST_DRAFTER names the GGUF file, or NANOLLM_TEST_MODELS holds one
// whose name says mtp, draft or assistant.
package llamacpp

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/sched"
)

// trained reports whether the model is a real one rather than a random-weight
// fixture: acceptance and output quality are checked only on trained
// weights (STAGE1 facts table), mechanics and equivalence on both.
func trained(m *model) bool {
	n, _ := strconv.ParseUint(m.meta["llama.n_params"], 10, 64)
	return n > 1e9
}

func testDrafterPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("NANOLLM_TEST_DRAFTER"); p != "" {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("NANOLLM_TEST_DRAFTER: %v", err)
		}
		return p
	}
	dir := os.Getenv("NANOLLM_TEST_MODELS")
	files, _ := filepath.Glob(filepath.Join(dir, "*.gguf"))
	for _, f := range files {
		low := strings.ToLower(filepath.Base(f))
		if dir != "" && (strings.Contains(low, "mtp") || strings.Contains(low, "draft") || strings.Contains(low, "assistant")) {
			return f
		}
	}
	t.Skip("no MTP drafter: set NANOLLM_TEST_DRAFTER, or put one named *mtp*.gguf in NANOLLM_TEST_MODELS")
	return ""
}

// mtpSpec is baseSpec with the drafter, and n_outputs_max sized the way the
// planner sizes it: slots × (1 + DraftMax).
func mtpSpec(path, drafter string) backend.LoadSpec {
	s := baseSpec(path)
	s.DrafterPath, s.DraftMax = drafter, 3
	s.NOutputsMax = s.Slots * (1 + s.DraftMax)
	return s
}

var sharedMTP struct {
	mu sync.Mutex
	m  *model
}

// sharedMTPModel is the target with its drafter, loaded on first use and
// kept until releaseSharedMTP.
func sharedMTPModel(t *testing.T) *model {
	t.Helper()
	b := testBackend(t)
	path, drafter := testModelPath(t), testDrafterPath(t)
	sharedMTP.mu.Lock()
	defer sharedMTP.mu.Unlock()
	if sharedMTP.m == nil {
		m, err := b.Load(context.Background(), mtpSpec(path, drafter))
		if err != nil {
			t.Fatal(err)
		}
		sharedMTP.m = m.(*model)
	}
	for s := 0; s < sharedMTP.m.caps.Slots; s++ {
		sharedMTP.m.ClearSeq(s)
	}
	return sharedMTP.m
}

// releaseSharedMTP frees the MTP model before a test loads models of its
// own: every E4B load with repacked weights holds 2.5 GiB of anonymous
// memory, and three at once do not fit a 15 GB VM next to the page cache.
func releaseSharedMTP() {
	sharedMTP.mu.Lock()
	defer sharedMTP.mu.Unlock()
	if sharedMTP.m != nil {
		_ = sharedMTP.m.Close()
		sharedMTP.m = nil
	}
}

func TestNativeMTPLoads(t *testing.T) {
	m := sharedMTPModel(t)
	c := m.Capabilities()
	t.Logf("caps %+v, n_outputs_max %d, memory %+v", c, m.nOutputsMax, m.MemoryReport().Devices)
	if c.MaxDraft != 3 || m.dft == nil || m.dft.nEmbd != int(llamaModelNEmbdOut(m.mdl)) {
		t.Fatalf("drafter not set up: caps %+v", c)
	}
	if llamaGetCtxOther(m.dft.ctx) != m.ctx {
		t.Error("the drafter context does not point at the target")
	}
	// A drafter that is not one, or a missing file, is refused before C.
	spec := mtpSpec(testModelPath(t), filepath.Join(t.TempDir(), "missing.gguf"))
	if _, err := testBackend(t).Load(context.Background(), spec); err == nil {
		t.Error("a missing drafter file was accepted")
	}
}

// TestNativeMTPMatchesServer compares the state machine with llama-server's
// own MTP (tools/golden -drafter) under greedy decoding: the same tokens and
// the same numbers of drafted and accepted tokens.
func TestNativeMTPMatchesServer(t *testing.T) {
	files := goldenFiles(true)
	if len(files) == 0 {
		t.Skip("no MTP golden files for " + bindingTag)
	}
	b := testBackend(t)
	path, drafter := testModelPath(t), testDrafterPath(t)
	releaseSharedMTP()
	fi, _ := os.Stat(path)
	di, _ := os.Stat(drafter)
	ran := false
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var g golden
		if err := json.Unmarshal(data, &g); err != nil {
			t.Fatal(err)
		}
		if g.ModelSize != fi.Size() || g.DrafterSize != di.Size() || g.CPUVariant != b.CPUVariant() ||
			g.ModelSHA256 != "" && g.ModelSHA256 != fileSHA256(t, path) || g.DrafterSHA != "" && g.DrafterSHA != fileSHA256(t, drafter) {
			t.Logf("%s: recorded for other files or another CPU variant", f)
			continue
		}
		ran = true
		spec := mtpSpec(path, drafter)
		spec.NCtx, spec.Slots, spec.NBatch, spec.NUBatch, spec.Threads = g.NCtx, 1, g.NBatch, g.NUBatch, g.Threads
		spec.KVTypeK, spec.KVTypeV = g.kv(), g.kv()
		spec.DraftMax, spec.DraftPMin, spec.NOutputsMax = g.DraftMax, g.DraftPMin, 1+g.DraftMax
		mm, err := b.Load(context.Background(), spec)
		if err != nil {
			t.Fatal(err)
		}
		m := mm.(*model)
		prompt := tokenize(t, m, g.Prompt)
		if err := m.SetSampler(0, greedy, prompt); err != nil {
			t.Fatal(err)
		}
		got := generate(t, m, 0, prompt, 0, g.NPredict, -1)
		text, _ := m.Detokenize(got.toks)
		t.Logf("ours: %d tokens, drafted %d, accepted %d in %d steps; llama-server: %d tokens, drafted %d, accepted %d",
			len(got.toks), got.drafted, got.accepted, got.steps, len(g.Tokens), g.Drafted, g.Accepted)
		if fmt.Sprint(got.toks) != fmt.Sprint(g.Tokens) || text != g.Text {
			t.Errorf("%s: tokens differ\n got %v\nwant %v", f, got.toks, g.Tokens)
		}
		if got.drafted != g.Drafted || got.accepted != g.Accepted {
			t.Errorf("%s: drafted/accepted %d/%d, llama-server %d/%d", f, got.drafted, got.accepted, g.Drafted, g.Accepted)
		}
		mm.Close()
	}
	if !ran {
		t.Skip("no MTP golden matches these files and this CPU variant")
	}
}

var equivalencePrompts = []string{
	"Write a short paragraph about the history of Paris.",
	"Explain in three sentences why the sky is blue.",
	"List five prime numbers and say why they are prime.",
	"Напиши короткое стихотворение о зиме.",
	"Write a Go function that reverses a string, with a comment.",
}

// TestNativeMTPEquivalence decodes five prompts greedily with and without
// the drafter. The target is an exact match; a divergence is reported with
// the top-1/top-2 logprob margin of the plain run at the first differing
// position, which tells kernel noise (a near tie: batched verification
// rounds differently from one-token decoding) from a bug (a wide margin).
// It also checks that drafts get accepted and reports both speeds.
func TestNativeMTPEquivalence(t *testing.T) {
	mtpEquivalence(t, sharedModel(t), sharedMTPModel(t))
}

// TestNativeMTPEquivalenceQ8 is the same with a q8_0 cache: the drafter
// inherits the target's cache types, and its drafts are verified against a
// quantized cache, which rounds differently from f16. The models are loaded
// here and freed after, one pair at a time.
func TestNativeMTPEquivalenceQ8(t *testing.T) {
	b := testBackend(t)
	path, drafter := testModelPath(t), testDrafterPath(t)
	releaseSharedMTP()
	load := func(s backend.LoadSpec) *model {
		s.KVTypeK, s.KVTypeV = "q8_0", "q8_0"
		m, err := b.Load(context.Background(), s)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = m.Close() })
		return m.(*model)
	}
	plain := load(baseSpec(path))
	mtp := load(mtpSpec(path, drafter))
	mtpEquivalence(t, plain, mtp)
}

func mtpEquivalence(t *testing.T, plain, mtp *model) {
	t.Helper()
	const n = 48
	var drafted, accepted, genPlain, genMTP int
	var tPlain, tMTP time.Duration
	for _, p := range equivalencePrompts {
		prompt := tokenize(t, plain, chat(plain, p))
		if err := plain.SetSampler(0, greedy, prompt); err != nil {
			t.Fatal(err)
		}
		plain.ClearSeq(0)
		want := generate(t, plain, 0, prompt, 0, n, 2)
		mtp.ClearSeq(0)
		if err := mtp.SetSampler(0, greedy, prompt); err != nil {
			t.Fatal(err)
		}
		got := generate(t, mtp, 0, prompt, 0, n, -1)
		drafted += got.drafted
		accepted += got.accepted
		genPlain += len(want.toks)
		genMTP += len(got.toks)
		tPlain += want.decode
		tMTP += got.decode
		t.Logf("%q: %d tokens, drafted %d accepted %d in %d steps", p, len(got.toks), got.drafted, got.accepted, got.steps)
		for i := range max(len(want.toks), len(got.toks)) {
			if i < len(want.toks) && i < len(got.toks) && want.toks[i] == got.toks[i] {
				continue
			}
			msg := fmt.Sprintf("%q: first difference at token %d", p, i)
			if i < len(want.lps) && len(want.lps[i].Top) >= 2 {
				top := want.lps[i].Top
				msg += fmt.Sprintf(": plain picked %d, top-1/top-2 margin %.4f (%d vs %d)", want.toks[i], top[0].Logprob-top[1].Logprob, top[0].Token, top[1].Token)
				if i < len(got.toks) {
					msg += fmt.Sprintf(", MTP picked %d", got.toks[i])
				}
			}
			t.Error(msg)
			break
		}
	}
	rate := func(n int, d time.Duration) float64 { return float64(n) / d.Seconds() }
	t.Logf("acceptance %d/%d (%.0f%%); decode %.2f tok/s plain, %.2f tok/s MTP",
		accepted, drafted, 100*float64(accepted)/math.Max(1, float64(drafted)), rate(genPlain, tPlain), rate(genMTP, tMTP))
	if accepted == 0 && trained(mtp) {
		t.Error("no draft token was accepted")
	}
}

// TestNativeMTPCarryReset checks that a step drafts only from a state that
// belongs to its position: after a trim or a restore the first step runs
// without drafts and computes a fresh state; the next one drafts again.
func TestNativeMTPCarryReset(t *testing.T) {
	m := sharedMTPModel(t)
	prompt := tokenize(t, m, chat(m, "Count from one to twenty."))
	if err := m.SetSampler(0, greedy, prompt); err != nil {
		t.Fatal(err)
	}
	first := generate(t, m, 0, prompt, 0, 1, -1).toks[0]
	step := func(tok backend.Token, pos int) backend.SeqOut {
		t.Helper()
		out, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: []backend.Token{tok}, Pos: pos, Sample: true, MaxDraft: 3, Logprobs: -1}})
		if err != nil {
			t.Fatal(err)
		}
		return out[0]
	}
	pos := len(prompt)
	o := step(first, pos)
	if o.Drafted == 0 {
		t.Fatal("a decode step right after the prompt did not draft")
	}
	pos += len(o.Tokens)
	saved, err := m.SaveSeq(0)
	if err != nil {
		t.Fatal(err)
	}
	last := o.Tokens[len(o.Tokens)-1]

	// Trim back one token and feed it again: the carried state is gone.
	if !m.TrimSeq(0, pos-1) {
		t.Fatal("trim refused")
	}
	redo := first
	if len(o.Tokens) >= 2 {
		redo = o.Tokens[len(o.Tokens)-2]
	}
	o2 := step(redo, pos-1)
	if o2.Drafted != 0 {
		t.Errorf("drafted %d right after a trim", o2.Drafted)
	}
	o3 := step(o2.Tokens[0], pos)
	if o3.Drafted == 0 {
		t.Error("no draft on the step after the fresh state")
	}

	if err := m.RestoreSeq(0, saved); err != nil {
		t.Fatal(err)
	}
	if o4 := step(last, pos); o4.Drafted != 0 {
		t.Errorf("drafted %d right after a restore", o4.Drafted)
	}
}

// TestNativeMTPGrammar: verification samples position by position through
// the chain, so a grammar holds with drafting on.
func TestNativeMTPGrammar(t *testing.T) {
	m := sharedMTPModel(t)
	prompt := tokenize(t, m, chat(m, "Give me a list of three fruits."))
	// Bounded, so that even random weights reach the end within the budget.
	sp := backend.SamplerSpec{Grammar: `root ::= "[" item ("," item)? ("," item)? "]"
item ::= "\"" [a-z] [a-z]? [a-z]? [a-z]? [a-z]? "\""`}
	if err := m.SetSampler(0, sp, prompt); err != nil {
		t.Fatal(err)
	}
	g := generate(t, m, 0, prompt, 0, 40, -1)
	toks := g.toks
	if len(toks) > 0 && m.IsEOG(toks[len(toks)-1]) {
		toks = toks[:len(toks)-1]
	}
	text, _ := m.Detokenize(toks)
	t.Logf("%q, drafted %d accepted %d", text, g.drafted, g.accepted)
	var list []string
	if err := json.Unmarshal([]byte(text), &list); err != nil || len(list) == 0 {
		t.Errorf("grammar output %q is not a JSON list of strings: %v", text, err)
	}
}

// TestNativeSchedulerOnMTP runs the scheduling loop on the real model: two
// concurrent requests with drafting, then a request that extends the first
// prompt and reuses its cached prefix. Every output must equal plain greedy
// decoding of the same prompt, so the backend keeps the Step contract the
// loop relies on — returned tokens, cache contents, positions — with drafts
// in the batch.
func TestNativeSchedulerOnMTP(t *testing.T) {
	plain := sharedModel(t)
	m := sharedMTPModel(t)
	l := sched.New(m)
	defer l.Close()
	const n = 32
	ask := func(p []backend.Token) (<-chan []backend.Token, <-chan sched.Event) {
		toks, done := make(chan []backend.Token, 1), make(chan sched.Event, 1)
		h, err := l.Submit(context.Background(), sched.Request{Prompt: p, Sampler: greedy, MaxTokens: n, Logprobs: -1})
		if err != nil {
			t.Fatal(err)
		}
		go func() {
			var all []backend.Token
			for {
				ev, err := h.Next(context.Background())
				if err != nil {
					done <- sched.Event{Err: err}
					return
				}
				all = append(all, ev.Tokens...)
				if ev.Done {
					toks <- all
					done <- ev
					return
				}
			}
		}()
		return toks, done
	}
	expect := func(p []backend.Token) []backend.Token {
		plain.ClearSeq(0)
		if err := plain.SetSampler(0, greedy, p); err != nil {
			t.Fatal(err)
		}
		g := generate(t, plain, 0, p, 0, n, -1)
		if len(g.toks) > 0 && plain.IsEOG(g.toks[len(g.toks)-1]) {
			return g.toks[:len(g.toks)-1]
		}
		return g.toks
	}
	p0 := tokenize(t, m, chat(m, "Write a haiku about the sea."))
	p1 := tokenize(t, m, chat(m, "Name the planets of the solar system."))
	w0, w1 := expect(p0), expect(p1)

	start := time.Now()
	t0, d0 := ask(p0)
	t1, d1 := ask(p1)
	got0, got1 := <-t0, <-t1
	e0, e1 := <-d0, <-d1
	t.Logf("two slots in %.1fs: %+v | %+v", time.Since(start).Seconds(), *e0.Stats, *e1.Stats)
	for i, c := range []struct {
		got, want []backend.Token
		ev        sched.Event
	}{{got0, w0, e0}, {got1, w1, e1}} {
		if c.ev.Err != nil {
			t.Fatalf("request %d: %v", i, c.ev.Err)
		}
		if fmt.Sprint(c.got) != fmt.Sprint(c.want) {
			t.Errorf("request %d through the loop:\n got %v\nwant %v", i, c.got, c.want)
		}
	}
	if e0.Stats.Drafted+e1.Stats.Drafted == 0 || trained(m) && e0.Stats.Accepted+e1.Stats.Accepted == 0 {
		t.Errorf("no drafting through the loop: %+v %+v", *e0.Stats, *e1.Stats)
	}

	// A follow-up that extends the first conversation reuses its prefix.
	follow := append(append(append([]backend.Token(nil), p0...), got0...), tokenize(t, m, "<turn|>\n"+chat(m, "Now one about mountains."))[1:]...)
	wf := expect(follow)
	tf, df := ask(follow)
	gotf, ef := <-tf, <-df
	if ef.Err != nil {
		t.Fatal(ef.Err)
	}
	t.Logf("follow-up: %+v", *ef.Stats)
	if ef.Stats.CachedTokens < len(p0) {
		t.Errorf("follow-up reused %d tokens, the first prompt alone is %d", ef.Stats.CachedTokens, len(p0))
	}
	if fmt.Sprint(gotf) != fmt.Sprint(wf) {
		t.Errorf("follow-up through the loop:\n got %v\nwant %v", gotf, wf)
	}
}

// TestNativeMTPSpeed compares decode speed without and with drafting for a
// few draft settings, in one process with both models warm (a process per
// run measures the page cache more than the model). Long; NANOLLM_TEST_LONG.
func TestNativeMTPSpeed(t *testing.T) {
	if os.Getenv("NANOLLM_TEST_LONG") == "" {
		t.Skip("set NANOLLM_TEST_LONG=1 to measure decode speed")
	}
	plain := sharedModel(t)
	mtp := sharedMTPModel(t)
	const n = 96
	prompts := equivalencePrompts[:3]
	run := func(m *model) (float64, int, int) {
		var toks, drafted, accepted int
		var d time.Duration
		for _, p := range prompts {
			prompt := tokenize(t, m, chat(m, p))
			m.ClearSeq(0)
			if err := m.SetSampler(0, greedy, prompt); err != nil {
				t.Fatal(err)
			}
			g := generate(t, m, 0, prompt, 0, n, -1)
			toks += len(g.toks)
			d += g.decode
			drafted += g.drafted
			accepted += g.accepted
		}
		return float64(toks) / d.Seconds(), drafted, accepted
	}
	run(plain) // warm up
	base, _, _ := run(plain)
	t.Logf("plain: %.2f tok/s", base)
	defer func(n int, p float64) { mtp.dft.maxDraft, mtp.caps.MaxDraft, mtp.dft.pMin = n, n, p }(mtp.dft.maxDraft, mtp.dft.pMin)
	for _, c := range []struct {
		max  int
		pMin float64
	}{{1, 0}, {2, 0}, {3, 0}, {4, 0}, {2, 0.5}, {3, 0.5}, {3, 0.75}} {
		mtp.dft.maxDraft, mtp.caps.MaxDraft, mtp.dft.pMin = c.max, c.max, c.pMin
		rate, dr, acc := run(mtp)
		t.Logf("MTP n_max %d p_min %.2f: %.2f tok/s (%.2fx), drafted %d accepted %d (%.0f%%)",
			c.max, c.pMin, rate, rate/base, dr, acc, 100*float64(acc)/math.Max(1, float64(dr)))
	}
}
