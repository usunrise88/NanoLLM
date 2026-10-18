package fake

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
)

func load(t *testing.T, opts Options, spec backend.LoadSpec) *Model {
	t.Helper()
	m, err := New(opts).Load(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	return m.(*Model)
}

// generate drives one slot the way the scheduler does: prefill, then feed
// back the last returned token until end of turn or n tokens.
func generate(t *testing.T, m *Model, seq int, prompt string, maxDraft, n int) (string, []backend.SeqOut) {
	t.Helper()
	toks, _ := m.Tokenize(prompt, true, true)
	pos := 0
	step := backend.SeqStep{Seq: seq, Tokens: toks, Pos: pos, Sample: true, MaxDraft: maxDraft, Logprobs: -1}
	var got []backend.Token
	var outs []backend.SeqOut
	for len(got) < n {
		res, err := m.Step(context.Background(), []backend.SeqStep{step})
		if err != nil {
			t.Fatal(err)
		}
		out := res[0]
		outs = append(outs, out)
		got = append(got, out.Tokens...)
		if out.EOG {
			break
		}
		pos = step.Pos + len(step.Tokens) + len(out.Tokens) - 1
		step = backend.SeqStep{Seq: seq, Tokens: out.Tokens[len(out.Tokens)-1:], Pos: pos, Sample: true, MaxDraft: maxDraft, Logprobs: -1}
	}
	text, err := m.Detokenize(got)
	if err != nil {
		t.Fatal(err)
	}
	return text, outs
}

func TestTokenizeRoundTrip(t *testing.T) {
	m := load(t, Options{}, backend.LoadSpec{})
	in := "<|turn>user\nПривет <|\"|>x<|\"|><turn|>\n"
	toks, _ := m.Tokenize(in, false, true)
	if toks[0] != special("<|turn>") {
		t.Fatalf("control token not parsed: %v", toks[:3])
	}
	out, _ := m.Detokenize(toks)
	if out != in {
		t.Fatalf("round trip: %q", out)
	}
	raw, _ := m.Tokenize("<|turn>", false, false)
	if len(raw) != len("<|turn>") {
		t.Fatalf("parseSpecial=false must keep bytes, got %d tokens", len(raw))
	}
}

func TestScenarios(t *testing.T) {
	m := load(t, Options{}, backend.LoadSpec{})
	for _, tc := range []struct{ prompt, want string }{
		{"<|turn>user\nhi<turn|>\n<|turn>model\n", DefaultReply + "<turn|>"},
		{"<|turn>user\nsay this [[fake:echo]]<turn|>\n<|turn>model\n", "say this<turn|>"},
		{"<|turn>user\n[[fake:json]]<turn|>\n<|turn>model\n", `{"ok":true,"items":[1,2,3],"name":"fake"}<turn|>`},
		{"[[fake:tool get_weather {city:<|\"|>Paris<|\"|>}]]", "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><turn|>"},
	} {
		got, _ := generate(t, m, 0, tc.prompt, 0, 1000)
		if got != tc.want {
			t.Errorf("%q: got %q want %q", tc.prompt, got, tc.want)
		}
		m.ClearSeq(0)
	}
}

func TestDraftingIsEquivalent(t *testing.T) {
	prompt := "<|turn>user\n[[fake:long 200]]<turn|>\n<|turn>model\n"
	plain, _ := generate(t, load(t, Options{}, backend.LoadSpec{}), 0, prompt, 0, 1000)
	m := load(t, Options{MaxDraft: 4, Acceptance: 0.5}, backend.LoadSpec{DrafterPath: "x"})
	drafted, outs := generate(t, m, 0, prompt, 4, 1000)
	if plain != drafted {
		t.Fatalf("drafting changed the output:\n%q\n%q", plain, drafted)
	}
	var d, a int
	for _, o := range outs {
		d += o.Drafted
		a += o.Accepted
	}
	if len(outs) >= len(drafted) || d == 0 || a == 0 {
		t.Fatalf("expected several tokens per step: %d steps for %d bytes, drafted %d accepted %d", len(outs), len(drafted), d, a)
	}
}

func TestBatchAcrossSlots(t *testing.T) {
	m := load(t, Options{}, backend.LoadSpec{Slots: 2, NBatch: 64})
	a, _ := m.Tokenize("[[fake:echo]]<|turn>user\naaa<turn|>", false, true)
	b, _ := m.Tokenize("<|turn>user\nbbb [[fake:echo]]<turn|>", false, true)
	steps := []backend.SeqStep{
		{Seq: 0, Tokens: a, Sample: true, Logprobs: 2},
		{Seq: 1, Tokens: b, Sample: true, Logprobs: -1},
	}
	outs, err := m.Step(context.Background(), steps)
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 2 || outs[0].Seq != 0 || outs[1].Seq != 1 {
		t.Fatalf("outs %+v", outs)
	}
	if len(outs[0].Logprobs) != 1 || len(outs[0].Logprobs[0].Top) != 2 || outs[1].Logprobs != nil {
		t.Fatalf("logprobs only where asked: %+v", outs)
	}
	if p, _ := m.TokenPiece(outs[1].Tokens[0]); p != "b" {
		t.Fatalf("slot 1 answered %q", p)
	}
	long := make([]backend.Token, 65)
	if _, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: long, Pos: len(a)}}); err == nil || !strings.Contains(err.Error(), "n_batch") {
		t.Fatalf("over-budget step: %v", err)
	}
	if _, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 1, Tokens: a, Pos: 3}}); err == nil {
		t.Fatal("wrong position must be refused")
	}
}

func TestTrimAndSWA(t *testing.T) {
	m := load(t, Options{}, backend.LoadSpec{NCtx: 4096, NBatch: 2048})
	toks := make([]backend.Token, 1000)
	if _, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: toks}}); err != nil {
		t.Fatal(err)
	}
	if m.TrimSeq(0, 100) {
		t.Fatal("trim past the sliding window without swa_full must fail")
	}
	if !m.TrimSeq(0, 900) {
		t.Fatal("trim within the window must work")
	}
	full := load(t, Options{}, backend.LoadSpec{NCtx: 4096, NBatch: 2048, SWAFull: true})
	full.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: toks}})
	if !full.TrimSeq(0, 1) {
		t.Fatal("swa_full trims anywhere")
	}
}

func TestStateRoundTrip(t *testing.T) {
	m := load(t, Options{}, backend.LoadSpec{Slots: 2})
	prompt := "<|turn>user\n[[fake:long 40]]<turn|>\n<|turn>model\n"
	want, _ := generate(t, m, 0, prompt, 0, 1000)
	m.ClearSeq(0)

	toks, _ := m.Tokenize(prompt, true, true)
	res, _ := m.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: toks, Sample: true, Logprobs: -1}})
	first := res[0].Tokens[0]
	state, err := m.SaveSeq(0)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.RestoreSeq(1, state); err != nil {
		t.Fatal(err)
	}
	got := []backend.Token{first}
	pos := len(toks)
	next := first
	for {
		res, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 1, Tokens: []backend.Token{next}, Pos: pos, Sample: true, Logprobs: -1}})
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, res[0].Tokens...)
		if res[0].EOG {
			break
		}
		pos++
		next = res[0].Tokens[0]
	}
	if s, _ := m.Detokenize(got); s != want {
		t.Fatalf("restored slot continued with %q, want %q", s, want)
	}
}

func TestCancelAndFailure(t *testing.T) {
	m := load(t, Options{DecodeRate: 1}, backend.LoadSpec{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Step(ctx, []backend.SeqStep{{Seq: 0, Tokens: []backend.Token{1}, Sample: true}}); err == nil {
		t.Fatal("cancelled step must fail")
	}
	f := load(t, Options{}, backend.LoadSpec{})
	toks, _ := f.Tokenize("[[fake:error]]", false, true)
	if _, err := f.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: toks, Sample: true}}); !errors.Is(err, errScenario) {
		t.Fatalf("scripted failure: %v", err)
	}
}

func TestLoopNeverEnds(t *testing.T) {
	m := load(t, Options{}, backend.LoadSpec{})
	got, outs := generate(t, m, 0, "[[fake:loop]]", 0, 50)
	if outs[len(outs)-1].EOG || !strings.HasPrefix(got, "la la ") {
		t.Fatalf("loop: %q", got)
	}
}
