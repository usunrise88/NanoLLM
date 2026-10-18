//go:build native

// Tests against the real libllama. They need the pinned libraries
// (scripts/fetch-libs.sh puts them in dist/<os>_<arch>/lib, or set
// NANOLLM_TEST_LIB_DIR) and, for everything past device enumeration, a target
// model: NANOLLM_TEST_MODEL names a GGUF file, or NANOLLM_TEST_MODELS a
// directory whose smallest non-drafter *.gguf is used. Without them a test
// skips and says why.
//
//	go test -tags native ./internal/backend/llamacpp/...
//
// The tests generate a handful of tokens each, so they run on CPU with the
// primary model (E4B) in minutes; the CI matrix runs them with a tiny model.
package llamacpp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
)

func testLibDir(t *testing.T) string {
	t.Helper()
	dir := os.Getenv("NANOLLM_TEST_LIB_DIR")
	if dir == "" {
		dir = filepath.Join("..", "..", "..", "dist", runtime.GOOS+"_"+runtime.GOARCH, "lib")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(abs, backend.SharedLibrary("llama"))); err != nil {
		t.Skipf("no libllama in %s: run scripts/fetch-libs.sh or set NANOLLM_TEST_LIB_DIR", abs)
	}
	return abs
}

func testBackend(t *testing.T) *llamaBackend {
	t.Helper()
	b, err := backend.Open(Name, testLibDir(t))
	if err != nil {
		t.Fatal(err)
	}
	return b.(*llamaBackend)
}

// testModelPath finds the target model, or skips.
func testModelPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("NANOLLM_TEST_MODEL"); p != "" {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("NANOLLM_TEST_MODEL: %v", err)
		}
		return p
	}
	dir := os.Getenv("NANOLLM_TEST_MODELS")
	if dir == "" {
		t.Skip("no test model: set NANOLLM_TEST_MODEL to a GGUF file or NANOLLM_TEST_MODELS to a directory")
	}
	files, _ := filepath.Glob(filepath.Join(dir, "*.gguf"))
	best, bestSize := "", int64(math.MaxInt64)
	for _, f := range files {
		low := strings.ToLower(filepath.Base(f))
		if strings.Contains(low, "mtp") || strings.Contains(low, "draft") || strings.Contains(low, "assistant") {
			continue
		}
		if fi, err := os.Stat(f); err == nil && fi.Size() < bestSize {
			best, bestSize = f, fi.Size()
		}
	}
	if best == "" {
		t.Skipf("no target *.gguf in NANOLLM_TEST_MODELS=%s", dir)
	}
	return best
}

// baseSpec is the context the shared model is loaded with: two slots, so
// batches across slots are exercised, flash attention, repacked CPU weights
// (as llama.cpp's tools default to), swa_full off (as llama-server does).
func baseSpec(path string) backend.LoadSpec {
	return backend.LoadSpec{
		Path: path, Device: "cpu", Repack: true,
		NCtx: 1024, Slots: 2, NBatch: 512, NUBatch: 512, Threads: 4,
		KVTypeK: "f16", KVTypeV: "f16", FlashAttention: true,
	}
}

var shared struct {
	once sync.Once
	m    *model
	err  error
}

// sharedModel is loaded once per test binary: a load is seconds even from the
// page cache, and every test below only needs its own slots cleared.
func sharedModel(t *testing.T) *model {
	t.Helper()
	b := testBackend(t)
	path := testModelPath(t)
	shared.once.Do(func() {
		m, err := b.Load(context.Background(), baseSpec(path))
		if err != nil {
			shared.err = err
			return
		}
		shared.m = m.(*model)
	})
	if shared.err != nil {
		t.Fatal(shared.err)
	}
	for s := 0; s < shared.m.caps.Slots; s++ {
		shared.m.ClearSeq(s)
	}
	return shared.m
}

var greedy = backend.SamplerSpec{Temperature: 0}

// decode prefills prompt on seq and generates up to n tokens greedily (or
// with sp), returning the generated ids.
func decode(t *testing.T, m *model, seq int, prompt []backend.Token, n int, sp backend.SamplerSpec) []backend.Token {
	t.Helper()
	m.ClearSeq(seq)
	if err := m.SetSampler(seq, sp, prompt); err != nil {
		t.Fatal(err)
	}
	return continueFrom(t, m, seq, prompt, 0, n)
}

// continueFrom evaluates toks starting at pos and samples n tokens.
func continueFrom(t *testing.T, m *model, seq int, toks []backend.Token, pos, n int) []backend.Token {
	t.Helper()
	return generate(t, m, seq, toks, pos, n, -1).toks
}

// genOut is one generation through Step, with its MTP counts.
type genOut struct {
	toks              []backend.Token
	lps               []backend.TokenLogprobs
	drafted, accepted int
	steps             int
	decode            time.Duration
}

// generate evaluates toks at pos, then decodes up to n tokens, drafting as
// llama-server does: at most Capabilities().MaxDraft per step and never past
// the token budget (n_remaining - 1).
func generate(t *testing.T, m *model, seq int, toks []backend.Token, pos, n, logprobs int) genOut {
	t.Helper()
	var out []backend.SeqOut
	var err error
	for p := 0; p < len(toks); p += m.caps.NBatch {
		end := min(p+m.caps.NBatch, len(toks))
		out, err = m.Step(context.Background(), []backend.SeqStep{{Seq: seq, Tokens: toks[p:end], Pos: pos + p, Sample: end == len(toks), Logprobs: logprobs}})
		if err != nil {
			t.Fatal(err)
		}
	}
	pos += len(toks)
	var g genOut
	start := time.Now()
	for {
		o := out[0]
		g.drafted += o.Drafted
		g.accepted += o.Accepted
		for i, tok := range o.Tokens {
			g.toks = append(g.toks, tok)
			if i < len(o.Logprobs) {
				g.lps = append(g.lps, o.Logprobs[i])
			}
			if (o.EOG && i == len(o.Tokens)-1) || len(g.toks) >= n {
				g.decode = time.Since(start)
				return g
			}
		}
		pos += len(o.Tokens) - 1
		draft := max(0, min(m.caps.MaxDraft, n-len(g.toks)-1))
		out, err = m.Step(context.Background(), []backend.SeqStep{{Seq: seq, Tokens: o.Tokens[len(o.Tokens)-1:], Pos: pos, Sample: true, MaxDraft: draft, Logprobs: logprobs}})
		if err != nil {
			t.Fatal(err)
		}
		g.steps++
		pos++
	}
}

func chat(m *model, user string) string {
	if strings.HasPrefix(m.meta["general.architecture"], "gemma4") {
		return "<|turn>user\n" + user + "<turn|>\n<|turn>model\n"
	}
	return user
}

func tokenize(t *testing.T, m *model, text string) []backend.Token {
	t.Helper()
	toks, err := m.Tokenize(text, true, true)
	if err != nil {
		t.Fatal(err)
	}
	return toks
}

func TestNativeRefusesForeignCommit(t *testing.T) {
	dir := testLibDir(t)
	saved := PinnedCommit
	PinnedCommit = "0123456789abcdef0123456789abcdef01234567"
	defer func() { PinnedCommit = saved }()
	_, err := backend.Open(Name, dir)
	if err == nil {
		t.Fatal("a library from another commit was accepted")
	}
	if !strings.Contains(err.Error(), "refused") || !strings.Contains(err.Error(), PinnedCommit) {
		t.Errorf("unclear refusal: %v", err)
	}
	t.Logf("refusal: %v", err)
}

func TestNativeCPUDevice(t *testing.T) {
	b := testBackend(t)
	t.Logf("ggml_commit() = %q, version %s, cpu variant %q", b.lib.commit, b.Version(), b.CPUVariant())
	if checkCommit(b.lib.commit, bindingCommit) != nil {
		t.Errorf("commit %q", b.lib.commit)
	}
	devs, err := b.Devices(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(devs) == 0 || devs[0].Type != backend.CPU || devs[0].Selector() != "llamacpp:cpu" {
		t.Fatalf("no CPU device first: %+v", devs)
	}
	if devs[0].MemoryTotal <= 0 || devs[0].Name == "" {
		t.Errorf("CPU device without memory or name: %+v", devs[0])
	}
	for _, d := range devs {
		t.Logf("%s %s %q %d/%d unified=%v", d.Selector(), d.Type, d.Name, d.MemoryFree, d.MemoryTotal, d.Unified)
	}
	for _, w := range b.Warnings() {
		t.Logf("warning: %s", w)
	}
	// Archives with one generic CPU library (Windows arm64) have no variant
	// to report; where there are variants, ggml must have named one.
	variants, _ := filepath.Glob(filepath.Join(testLibDir(t), "*ggml-cpu-*"))
	if len(variants) > 0 && b.CPUVariant() == "" {
		t.Error("the CPU variant ggml picked was not found in its log")
	}
	if _, _, err := b.device("llamacpp:nosuchgpu7"); err == nil {
		t.Error("an unknown device selector was accepted")
	}
	t.Logf("staging symbols bound: %v", b.lib.cxx)
	if len(b.lib.cxx) != len(cxxSymbols) {
		t.Errorf("bound %d of %d staging-header functions", len(b.lib.cxx), len(cxxSymbols))
	}
}

func TestNativeTokenizeRoundTrip(t *testing.T) {
	m := sharedModel(t)
	for _, text := range []string{
		"Hello, world!",
		"  leading spaces and\ttabs\nnew lines\n\n",
		"Привет, мир! Как дела?",
		"日本語のテキスト 🙂 emoji",
		"code: func main() { fmt.Println(\"x\") }",
	} {
		toks, err := m.Tokenize(text, false, false)
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.Detokenize(toks)
		if err != nil {
			t.Fatal(err)
		}
		if got != text {
			t.Errorf("round trip of %q gave %q", text, got)
		}
		var pieces strings.Builder
		for _, tok := range toks {
			p, err := m.TokenPiece(tok)
			if err != nil {
				t.Fatal(err)
			}
			pieces.WriteString(p)
		}
		if pieces.String() != text {
			t.Errorf("pieces of %q join to %q", text, pieces.String())
		}
	}
	// BOS with addSpecial; special tokens parsed only when asked.
	withBOS := tokenize(t, m, "hi")
	plain, _ := m.Tokenize("hi", false, false)
	if len(withBOS) != len(plain)+1 {
		t.Errorf("addSpecial: %v vs %v", withBOS, plain)
	}
	if strings.HasPrefix(m.meta["general.architecture"], "gemma4") {
		sp, _ := m.Tokenize("<turn|>", false, true)
		lit, _ := m.Tokenize("<turn|>", false, false)
		if len(sp) != 1 || !m.IsEOG(sp[0]) || len(lit) <= 1 {
			t.Errorf("special parsing: parsed %v literal %v", sp, lit)
		}
	}
	if _, err := m.Detokenize([]backend.Token{backend.Token(m.nVocab)}); err == nil {
		t.Error("a token outside the vocabulary was detokenized")
	}
}

func TestNativeGreedyIsDeterministic(t *testing.T) {
	m := sharedModel(t)
	prompt := tokenize(t, m, chat(m, "Count from one to five."))
	a := decode(t, m, 0, prompt, 8, greedy)
	b := decode(t, m, 0, prompt, 8, greedy)
	c := decode(t, m, 1, prompt, 8, greedy)
	text, _ := m.Detokenize(a)
	t.Logf("greedy: %v %q", a, text)
	if fmt.Sprint(a) != fmt.Sprint(b) {
		t.Errorf("two greedy runs differ: %v vs %v", a, b)
	}
	if fmt.Sprint(a) != fmt.Sprint(c) {
		t.Errorf("slot 1 differs from slot 0: %v vs %v", c, a)
	}
}

func TestNativeTwoSlotsInOneStep(t *testing.T) {
	m := sharedModel(t)
	p0 := tokenize(t, m, chat(m, "Say hello."))
	p1 := tokenize(t, m, chat(m, "Say goodbye."))
	for s, p := range [][]backend.Token{p0, p1} {
		if err := m.SetSampler(s, greedy, p); err != nil {
			t.Fatal(err)
		}
	}
	outs, err := m.Step(context.Background(), []backend.SeqStep{
		{Seq: 0, Tokens: p0, Pos: 0, Sample: true, Logprobs: 3},
		{Seq: 1, Tokens: p1, Pos: 0, Sample: true, Logprobs: -1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 2 || outs[0].Seq != 0 || outs[1].Seq != 1 {
		t.Fatalf("outputs: %+v", outs)
	}
	// Batched prefill equals prefill alone for the first token.
	alone := decode(t, m, 0, p1, 1, greedy)
	if alone[0] != outs[1].Tokens[0] {
		t.Errorf("slot 1 sampled %d in the shared batch, %d alone", outs[1].Tokens[0], alone[0])
	}
	lp := outs[0].Logprobs
	if len(lp) != 1 || len(lp[0].Top) != 3 {
		t.Fatalf("logprobs: %+v", lp)
	}
	top := lp[0].Top
	if top[0].Token != outs[0].Tokens[0] || math.Abs(top[0].Logprob-lp[0].Logprob) > 1e-9 {
		t.Errorf("greedy pick %d is not the top logprob %+v", outs[0].Tokens[0], top[0])
	}
	sum := 0.0
	for i, x := range top {
		if x.Logprob > 0 || (i > 0 && x.Logprob > top[i-1].Logprob) {
			t.Errorf("logprobs not sorted or positive: %+v", top)
		}
		sum += math.Exp(x.Logprob)
	}
	if sum > 1+1e-6 {
		t.Errorf("top probabilities sum to %f", sum)
	}
}

func TestNativeStateRoundTrip(t *testing.T) {
	m := sharedModel(t)
	prompt := tokenize(t, m, chat(m, "Name three colors."))
	first := decode(t, m, 0, prompt, 1, greedy)
	state, err := m.SaveSeq(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("slot state after %d tokens: %d bytes", len(prompt), len(state))
	want := continueFrom(t, m, 0, first, len(prompt), 6)

	// Restore into the other slot and continue from the same point.
	if err := m.RestoreSeq(1, state); err != nil {
		t.Fatal(err)
	}
	if err := m.SetSampler(1, greedy, prompt); err != nil {
		t.Fatal(err)
	}
	got := continueFrom(t, m, 1, first, len(prompt), 6)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("after restore: %v, want %v", got, want)
	}

	bad := append([]byte(nil), state...)
	bad[30] ^= 0xff
	if err := m.RestoreSeq(1, bad); err == nil {
		t.Error("a state with a wrong fingerprint was restored")
	}
	if err := m.RestoreSeq(1, []byte("garbage")); err == nil {
		t.Error("garbage was restored")
	}
}

func TestNativeTrimAndRedo(t *testing.T) {
	m := sharedModel(t)
	prompt := tokenize(t, m, chat(m, "Write the alphabet."))
	want := decode(t, m, 0, prompt, 4, greedy)
	// The cache now holds the prompt and three generated tokens. Cut it back
	// to before the prompt's last token and sample the same continuation.
	if !m.TrimSeq(0, len(prompt)-1) {
		t.Fatal("trim within the window refused")
	}
	if err := m.SetSampler(0, greedy, prompt); err != nil {
		t.Fatal(err)
	}
	got := continueFrom(t, m, 0, prompt[len(prompt)-1:], len(prompt)-1, 4)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("after trim: %v, want %v", got, want)
	}
	if !m.TrimSeq(0, 1<<20) {
		t.Error("a trim past the end should be a no-op")
	}
}

// TestNativeTrimPastWindow fills the sliding-window cache past its size, so
// llama.cpp reuses the cells that fell out of the window, and checks the trim
// rule: within the window a trim works, further back it is refused with
// swa_full off. It prefills about a thousand tokens, so it runs only with
// NANOLLM_TEST_LONG=1.
func TestNativeTrimPastWindow(t *testing.T) {
	if os.Getenv("NANOLLM_TEST_LONG") == "" {
		t.Skip("set NANOLLM_TEST_LONG=1 to prefill ~1000 tokens")
	}
	b := testBackend(t)
	m0 := sharedModel(t)
	if m0.nSWA == 0 {
		t.Skip("the model has no sliding-window layers")
	}
	text := strings.Repeat("The quick brown fox jumps over the lazy dog while the band plays on. ", 80)
	for _, swaFull := range []bool{false, true} {
		spec := baseSpec(testModelPath(t))
		spec.NCtx, spec.Slots, spec.NUBatch, spec.SWAFull = 2048, 1, 64, swaFull
		mm, err := b.Load(context.Background(), spec)
		if err != nil {
			t.Fatal(err)
		}
		m := mm.(*model)
		toks := tokenize(t, m, text)
		if err := m.SetSampler(0, greedy, nil); err != nil {
			t.Fatal(err)
		}
		continueFrom(t, m, 0, toks, 0, 1)
		pmin := int(llamaMemorySeqPosMin(m.mem, 0))
		pmax := int(llamaMemorySeqPosMax(m.mem, 0))
		t.Logf("swa_full=%v: %d tokens, cache holds positions %d..%d", swaFull, len(toks), pmin, pmax)
		if !m.TrimSeq(0, len(toks)-10) {
			t.Errorf("swa_full=%v: a trim inside the window was refused", swaFull)
		}
		far := m.TrimSeq(0, 100)
		if swaFull != far {
			t.Errorf("swa_full=%v: trim to 100 with positions %d.. kept: %v", swaFull, pmin, far)
		}
		mm.Close()
	}
}

func TestNativeGrammarConstrains(t *testing.T) {
	m := sharedModel(t)
	prompt := tokenize(t, m, chat(m, "Is the sky green? Answer in a full sentence."))
	sp := backend.SamplerSpec{Temperature: 1.5, Seed: 7, Grammar: `root ::= "yes" | "no"`}
	got := decode(t, m, 0, prompt, 8, sp)
	if len(got) == 0 || !m.IsEOG(got[len(got)-1]) {
		t.Fatalf("grammar output %v does not end the generation", got)
	}
	text, _ := m.Detokenize(got[:len(got)-1])
	t.Logf("constrained: %q", text)
	if text != "yes" && text != "no" {
		t.Errorf("grammar output %q (%v)", text, got)
	}
	if err := m.SetSampler(0, backend.SamplerSpec{Grammar: `root ::= "unterminated`}, nil); err == nil {
		t.Error("a broken grammar was accepted")
	} else {
		t.Logf("broken grammar: %v", err)
	}
	// Lazy: unconstrained until the trigger appears in the output.
	lazy := backend.SamplerSpec{Grammar: `root ::= "<call>" [0-9]+ "</call>"`, GrammarTriggers: []string{"<call>"}}
	if err := m.SetSampler(0, lazy, prompt); err != nil {
		t.Fatal(err)
	}
}

func TestNativeRejectsBadSteps(t *testing.T) {
	m := sharedModel(t)
	prompt := tokenize(t, m, "hello")
	_ = m.SetSampler(0, greedy, nil)
	ctx := context.Background()
	for name, steps := range map[string][]backend.SeqStep{
		"empty":        nil,
		"no tokens":    {{Seq: 0}},
		"wrong pos":    {{Seq: 0, Tokens: prompt, Pos: 5}},
		"bad slot":     {{Seq: 9, Tokens: prompt}},
		"bad token":    {{Seq: 0, Tokens: []backend.Token{-1}}},
		"duplicate":    {{Seq: 0, Tokens: prompt}, {Seq: 0, Tokens: prompt}},
		"logprobs":     {{Seq: 0, Tokens: prompt, Sample: true, Logprobs: 21}},
		"over n_batch": {{Seq: 0, Tokens: make([]backend.Token, m.caps.NBatch+1)}},
	} {
		if _, err := m.Step(ctx, steps); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	m.freeSampler(1)
	if _, err := m.Step(ctx, []backend.SeqStep{{Seq: 1, Tokens: prompt, Sample: true}}); err == nil {
		t.Error("sampling without a sampler accepted")
	}
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := m.Step(cctx, []backend.SeqStep{{Seq: 0, Tokens: prompt}}); err == nil {
		t.Error("a cancelled step ran")
	}
}

func TestNativeMemoryReport(t *testing.T) {
	m := sharedModel(t)
	r := m.MemoryReport()
	t.Logf("memory: %+v", r.Devices)
	if len(r.Devices) == 0 || r.Devices[0].Device != "cpu" || r.Devices[0].Weights == 0 || r.Devices[0].KV == 0 || r.Devices[0].Compute == 0 {
		t.Errorf("report: %+v", r)
	}
	for _, l := range m.loadLog {
		if strings.Contains(l, "creating") && strings.Contains(l, "KV cache") || strings.Contains(l, "llama_kv_cache: size") ||
			strings.Contains(l, "buffer size") && !strings.Contains(l, "=     0.0") {
			t.Log(l)
		}
	}
}

// KV cache size (STAGE1 facts table, llama-context.cpp:377 and
// llama-kv-cache-iswa.cpp:73):
//
//	n_ctx      = pad256(n_ctx)
//	size_base  = unified ? n_ctx : pad256(n_ctx / n_seq_max)   cells per stream
//	size_swa   = swa_full ? size_base : pad256(min(size_base, n_swa·(unified ? n_seq_max : 1) + n_ubatch))
//	streams    = unified ? 1 : n_seq_max
//	bytes      = cells · streams · layers · (n_embd_k_gqa·sizeof(K) + n_embd_v_gqa·sizeof(V))
//
// with separate layer counts and head dims for the full and SWA caches. The
// test reads the cell counts and sizes llama.cpp logs for a grid of shapes.
type kvShape struct {
	nCtxPerSlot, slots, ubatch int
	unified, swaFull           bool
}

func pad256(n int) int { return (n + 255) / 256 * 256 }

func (s kvShape) cells(nSWA int) (base, swa int) {
	nctx := pad256(s.nCtxPerSlot * s.slots)
	base = nctx
	if !s.unified {
		base = pad256(nctx / s.slots)
	}
	if s.swaFull {
		return base, base
	}
	k := 1
	if s.unified {
		k = s.slots
	}
	return base, pad256(min(base, nSWA*k+s.ubatch))
}

var (
	reCells   = regexp.MustCompile(`creating\s+(non-SWA|SWA) KV cache, size = (\d+) cells`)
	reKVTotal = regexp.MustCompile(`llama_kv_cache: size =\s+([0-9.]+) MiB \(\s*(\d+) cells,\s+(\d+) layers,\s+(\d+)/(\d+) seqs\)`)
)

func TestNativeKVCellsMatchFormula(t *testing.T) {
	b := testBackend(t)
	path := testModelPath(t)
	m0 := sharedModel(t)
	arch := m0.meta["general.architecture"]
	nSWA := m0.nSWA
	if nSWA == 0 {
		t.Skip("the model has no sliding-window layers")
	}
	atoi := func(k string) int { n, _ := strconv.Atoi(m0.meta[arch+"."+k]); return n }
	headsKV := atoi("attention.head_count_kv")
	full := 2 * headsKV * (atoi("attention.key_length") + atoi("attention.value_length")) // f16 K+V bytes per cell per layer
	swa := 2 * headsKV * (atoi("attention.key_length_swa") + atoi("attention.value_length_swa"))
	grid := []kvShape{
		{1000, 1, 512, false, false}, // n_ctx padded to 1024
		{4000, 2, 512, false, false}, // per-stream base, SWA window + ubatch
		{1000, 2, 256, true, false},  // unified: one stream, window × slots
		{3000, 1, 512, false, false},
		{3000, 1, 512, false, true}, // swa_full: SWA as large as the base
	}
	releaseSharedMTP()
	for _, s := range grid {
		spec := baseSpec(path)
		// File-backed weights: the cache shape does not depend on them.
		spec.Repack = false
		spec.NCtx, spec.Slots, spec.NUBatch, spec.KVUnified, spec.SWAFull = s.nCtxPerSlot, s.slots, s.ubatch, s.unified, s.swaFull
		mm, err := b.Load(context.Background(), spec)
		if err != nil {
			t.Fatal(err)
		}
		lm := mm.(*model)
		wantBase, wantSWA := s.cells(nSWA)
		var gotBase, gotSWA int
		var sizes []string
		per := full // the non-SWA cache is created and logged first
		for _, l := range lm.loadLog {
			if mt := reCells.FindStringSubmatch(l); mt != nil {
				n, _ := strconv.Atoi(mt[2])
				if mt[1] == "SWA" {
					gotSWA, per = n, swa
				} else {
					gotBase, per = n, full
				}
			}
			if mt := reKVTotal.FindStringSubmatch(l); mt != nil {
				mib, _ := strconv.ParseFloat(mt[1], 64)
				cells, _ := strconv.Atoi(mt[2])
				layers, _ := strconv.Atoi(mt[3])
				streams := s.slots
				if s.unified {
					streams = 1
				}
				want := float64(cells*streams*layers*per) / (1 << 20)
				sizes = append(sizes, strings.TrimPrefix(l, "llama_kv_cache: "))
				if math.Abs(want-mib) > 0.01 {
					t.Errorf("%+v: %s, formula %.2f MiB", s, l, want)
				}
			}
		}
		t.Logf("%+v: cells base %d (formula %d), swa %d (formula %d); %s", s, gotBase, wantBase, gotSWA, wantSWA, strings.Join(sizes, " | "))
		if gotBase != wantBase || gotSWA != wantSWA {
			t.Errorf("%+v: cells base %d swa %d, formula %d %d", s, gotBase, gotSWA, wantBase, wantSWA)
		}
		mm.Close()
	}
}

// golden is tools/golden's format.
type golden struct {
	Tag          string  `json:"tag"`
	Commit       string  `json:"commit"`
	ModelSize    int64   `json:"model_size"`
	ModelSHA256  string  `json:"model_sha256"`
	CPUVariant   string  `json:"cpu_variant"`
	Threads      int     `json:"threads"`
	NCtx         int     `json:"n_ctx"`
	NBatch       int     `json:"n_batch"`
	NUBatch      int     `json:"n_ubatch"`
	Prompt       string  `json:"prompt"`
	PromptTokens []int32 `json:"prompt_tokens"`
	NPredict     int     `json:"n_predict"`
	Tokens       []int32 `json:"tokens"`
	Text         string  `json:"text"`
	DrafterSize  int64   `json:"drafter_size"`
	DrafterSHA   string  `json:"drafter_sha256"`
	DraftMax     int     `json:"draft_max"`
	DraftPMin    float64 `json:"draft_p_min"`
	Drafted      int     `json:"drafted"`
	Accepted     int     `json:"accepted"`
}

// TestNativeGolden compares greedy decoding with llama-server's output of the
// same tag (tools/golden). It applies only to the model file the golden was
// recorded from and only on the same ggml CPU variant: other kernels round
// differently, and greedy picks may flip.
func TestNativeGolden(t *testing.T) {
	files, _ := filepath.Glob(filepath.Join("..", "..", "..", "testdata", "golden", bindingTag, "*", "cpu.json"))
	if len(files) == 0 {
		t.Skip("no golden files for " + bindingTag)
	}
	b := testBackend(t)
	path := testModelPath(t)
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	releaseSharedMTP()
	ran := false
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		var g golden
		if err := json.Unmarshal(data, &g); err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		if g.ModelSize != fi.Size() || g.ModelSHA256 != "" && g.ModelSHA256 != fileSHA256(t, path) {
			t.Logf("%s: recorded for another model file", f)
			continue
		}
		if g.CPUVariant != b.CPUVariant() {
			t.Logf("%s: recorded on ggml-cpu-%s, this machine uses %s", f, g.CPUVariant, b.CPUVariant())
			continue
		}
		ran = true
		spec := baseSpec(path)
		spec.NCtx, spec.Slots, spec.NBatch, spec.NUBatch, spec.Threads = g.NCtx, 1, g.NBatch, g.NUBatch, g.Threads
		mm, err := b.Load(context.Background(), spec)
		if err != nil {
			t.Fatal(err)
		}
		m := mm.(*model)
		prompt := tokenize(t, m, g.Prompt)
		if fmt.Sprint(prompt) != fmt.Sprint(g.PromptTokens) {
			t.Errorf("%s: prompt tokens %v, llama-server %v", f, prompt, g.PromptTokens)
		}
		got := decode(t, m, 0, prompt, g.NPredict, greedy)
		text, _ := m.Detokenize(got)
		t.Logf("%s: %q", f, text)
		if fmt.Sprint(got) != fmt.Sprint(g.Tokens) {
			for i := range min(len(got), len(g.Tokens)) {
				if int32(got[i]) != g.Tokens[i] {
					t.Errorf("%s: first difference at token %d: %d, llama-server %d", f, i, got[i], g.Tokens[i])
					break
				}
			}
			t.Errorf("%s:\n got %v\nwant %v", f, got, g.Tokens)
		}
		if text != g.Text {
			t.Errorf("%s: text %q, llama-server %q", f, text, g.Text)
		}
		mm.Close()
	}
	if !ran {
		t.Skip("no golden file matches this model and CPU variant")
	}
}

var shaCache sync.Map

// fileSHA256 hashes a file once per test binary.
func fileSHA256(t *testing.T, path string) string {
	t.Helper()
	if v, ok := shaCache.Load(path); ok {
		return v.(string)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	sum := hex.EncodeToString(h.Sum(nil))
	shaCache.Store(path, sum)
	return sum
}

func TestNativeMetadata(t *testing.T) {
	m := sharedModel(t)
	md := m.Metadata()
	keys := make([]string, 0, len(md))
	for k := range md {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if md["general.architecture"] == "" || md["llama.n_params"] == "" {
		t.Errorf("metadata keys: %v", keys)
	}
	c := m.Capabilities()
	t.Logf("arch %s, %s, caps %+v", md["general.architecture"], md["llama.desc"], c)
	if c.NCtx != 1024 || c.Slots != 2 || c.NVocab == 0 || c.MaxDraft != 0 {
		t.Errorf("capabilities: %+v", c)
	}
}
