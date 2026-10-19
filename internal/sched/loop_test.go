package sched

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
)

func newLoop(t *testing.T, opts fake.Options, spec backend.LoadSpec) (*Loop, backend.Model) {
	t.Helper()
	m, err := fake.New(opts).Load(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	l := New(m)
	t.Cleanup(func() { l.Close(); m.Close() })
	return l, m
}

func prompt(m backend.Model, user string) []backend.Token {
	toks, _ := m.Tokenize("<|turn>user\n"+user+"<turn|>\n<|turn>model\n", true, true)
	return toks
}

func run(t *testing.T, l *Loop, m backend.Model, req Request) (string, Event) {
	t.Helper()
	h, err := l.Submit(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	var toks []backend.Token
	for {
		ev, err := h.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		toks = append(toks, ev.Tokens...)
		if ev.Done {
			s, _ := m.Detokenize(toks)
			return s, ev
		}
	}
}

func TestGenerateAndFinish(t *testing.T) {
	l, m := newLoop(t, fake.Options{}, backend.LoadSpec{})
	text, ev := run(t, l, m, Request{Prompt: prompt(m, "hello [[fake:echo]]"), Logprobs: -1})
	if text != "hello" || ev.Finish != FinishEOG || ev.Err != nil {
		t.Fatalf("%q %+v", text, ev)
	}
	if ev.Stats.Completion != 5 || ev.Stats.PromptTokens == 0 {
		t.Fatalf("stats %+v", ev.Stats)
	}
	text, ev = run(t, l, m, Request{Prompt: prompt(m, "[[fake:loop]]"), MaxTokens: 7, Logprobs: -1})
	if text != "la la l" || ev.Finish != FinishLength {
		t.Fatalf("max tokens: %q %+v", text, ev)
	}
}

func TestPromptCacheReuse(t *testing.T) {
	l, m := newLoop(t, fake.Options{}, backend.LoadSpec{Slots: 2})
	first := prompt(m, "a long shared conversation prefix [[fake:echo]]")
	run(t, l, m, Request{Prompt: first, Logprobs: -1})
	second := append(append([]backend.Token(nil), first...), prompt(m, "next")...)
	_, ev := run(t, l, m, Request{Prompt: second, Logprobs: -1})
	if ev.Stats.CachedTokens != len(first) {
		t.Fatalf("cached %d of %d shared tokens", ev.Stats.CachedTokens, len(first))
	}
	_, ev = run(t, l, m, Request{Prompt: second, NoCache: true, Logprobs: -1})
	if ev.Stats.CachedTokens != 0 {
		t.Fatalf("NoCache reused %d tokens", ev.Stats.CachedTokens)
	}
	// The same prompt again reuses all but the last token.
	_, ev = run(t, l, m, Request{Prompt: second, Logprobs: -1})
	if ev.Stats.CachedTokens != len(second)-1 {
		t.Fatalf("identical prompt cached %d of %d", ev.Stats.CachedTokens, len(second))
	}
}

func TestSessionPinsSlot(t *testing.T) {
	l, m := newLoop(t, fake.Options{}, backend.LoadSpec{Slots: 3})
	_, a := run(t, l, m, Request{Prompt: prompt(m, "x"), SessionID: "s1", Logprobs: -1})
	run(t, l, m, Request{Prompt: prompt(m, "y"), Logprobs: -1})
	_, b := run(t, l, m, Request{Prompt: prompt(m, "z"), SessionID: "s1", Logprobs: -1})
	if a.Stats.Slot != b.Stats.Slot {
		t.Fatalf("session moved from slot %d to %d", a.Stats.Slot, b.Stats.Slot)
	}
}

func TestConcurrentSlotsAndQueue(t *testing.T) {
	l, m := newLoop(t, fake.Options{DecodeRate: 2000}, backend.LoadSpec{Slots: 3, NBatch: 64})
	var wg sync.WaitGroup
	results := make([]string, 8)
	for i := range results {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i], _ = run(t, l, m, Request{Prompt: prompt(m, strings.Repeat("abc ", 30)+" [[fake:long 60]]"), Logprobs: -1})
		}()
	}
	wg.Wait()
	for i, r := range results {
		if len(r) != 60 {
			t.Fatalf("request %d: %d bytes", i, len(r))
		}
	}
}

func TestDraftingAndBudget(t *testing.T) {
	l, m := newLoop(t, fake.Options{MaxDraft: 3, Acceptance: 1}, backend.LoadSpec{DrafterPath: "d", NBatch: 8})
	text, ev := run(t, l, m, Request{Prompt: prompt(m, "[[fake:long 40]]"), Logprobs: -1})
	if len(text) != 40 || ev.Stats.Accepted == 0 {
		t.Fatalf("%d bytes, stats %+v", len(text), ev.Stats)
	}
	_, ev = run(t, l, m, Request{Prompt: prompt(m, "[[fake:long 40]]"), NoDraft: true, Logprobs: -1})
	if ev.Stats.Drafted != 0 {
		t.Fatalf("NoDraft drafted %d", ev.Stats.Drafted)
	}
	// Drafting never overshoots max_tokens.
	text, ev = run(t, l, m, Request{Prompt: prompt(m, "[[fake:loop]]"), MaxTokens: 10, Logprobs: -1})
	if len(text) != 10 || ev.Finish != FinishLength {
		t.Fatalf("%q %+v", text, ev)
	}
}

func TestCancel(t *testing.T) {
	l, m := newLoop(t, fake.Options{DecodeRate: 200}, backend.LoadSpec{})
	ctx, cancel := context.WithCancel(context.Background())
	h, _ := l.Submit(ctx, Request{Prompt: prompt(m, "[[fake:loop]]"), Logprobs: -1})
	ev, _ := h.Next(context.Background())
	if ev.Done {
		t.Fatal("ended before cancel")
	}
	cancel()
	deadline := time.After(2 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("cancel did not end the request")
		default:
		}
		ev, _ := h.Next(context.Background())
		if ev.Done {
			if !errors.Is(ev.Err, context.Canceled) {
				t.Fatalf("err %v", ev.Err)
			}
			break
		}
	}
	// The slot is free again.
	text, _ := run(t, l, m, Request{Prompt: prompt(m, "ok [[fake:echo]]"), Logprobs: -1})
	if text != "ok" {
		t.Fatalf("after cancel: %q", text)
	}
}

func TestContextLimits(t *testing.T) {
	l, m := newLoop(t, fake.Options{}, backend.LoadSpec{NCtx: 64})
	if _, err := l.Submit(context.Background(), Request{Prompt: make([]backend.Token, 64)}); err == nil {
		t.Fatal("prompt filling the context must be refused")
	}
	text, ev := run(t, l, m, Request{Prompt: prompt(m, "[[fake:loop]]"), Logprobs: -1})
	if ev.Finish != FinishLength || len(text) == 0 {
		t.Fatalf("context exhaustion: %q %+v", text, ev)
	}
}

func TestBackendError(t *testing.T) {
	l, m := newLoop(t, fake.Options{}, backend.LoadSpec{})
	_, ev := run(t, l, m, Request{Prompt: prompt(m, "[[fake:error]]"), Logprobs: -1})
	if ev.Err == nil {
		t.Fatal("expected the step error")
	}
	text, _ := run(t, l, m, Request{Prompt: prompt(m, "fine [[fake:echo]]"), Logprobs: -1})
	if text != "fine" {
		t.Fatalf("loop did not recover: %q", text)
	}
}

func TestCloseEndsRequests(t *testing.T) {
	m, _ := fake.New(fake.Options{DecodeRate: 100}).Load(context.Background(), backend.LoadSpec{})
	l := New(m)
	h, _ := l.Submit(context.Background(), Request{Prompt: prompt(m, "[[fake:loop]]"), Logprobs: -1})
	time.Sleep(30 * time.Millisecond)
	l.Close()
	for {
		ev, _ := h.Next(context.Background())
		if ev.Done {
			if !errors.Is(ev.Err, ErrClosed) {
				t.Fatalf("err %v", ev.Err)
			}
			return
		}
	}
}

// A request cancelled between steps (a client that left, a stop sequence, the
// reasoning budget) keeps the slot's cache: the next request over the same
// prompt, as the continuation after a budget is, reuses it.
func TestCancelKeepsTheCache(t *testing.T) {
	l, m := newLoop(t, fake.Options{DecodeRate: 200}, backend.LoadSpec{Slots: 1})
	p := prompt(m, "a prompt that is long enough to be worth caching [[fake:loop]]")
	ctx, cancel := context.WithCancel(context.Background())
	h, err := l.Submit(ctx, Request{Prompt: p, Logprobs: -1})
	if err != nil {
		t.Fatal(err)
	}
	if ev, err := h.Next(context.Background()); err != nil || ev.Done {
		t.Fatalf("first event %+v, %v", ev, err)
	}
	cancel()
	var last Event
	for !last.Done {
		if last, err = h.Next(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if !errors.Is(last.Err, context.Canceled) {
		t.Fatalf("cancelled request ended with %v", last.Err)
	}
	_, ev := run(t, l, m, Request{Prompt: p, MaxTokens: 3, Logprobs: -1})
	if ev.Stats.CachedTokens < len(p)-1 {
		t.Fatalf("the next request reused %d of %d prompt tokens: the cancel cleared the slot", ev.Stats.CachedTokens, len(p))
	}
}

// Past the bound on waiting requests a Submit is refused, not queued: the
// queue is for a burst, not for a server that cannot keep up.
func TestMaxPending(t *testing.T) {
	l, m := newLoop(t, fake.Options{DecodeRate: 50}, backend.LoadSpec{Slots: 1})
	l.SetMaxPending(2)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var handles []*Handle
	var refused int
	for i := 0; i < 6; i++ {
		h, err := l.Submit(ctx, Request{Prompt: prompt(m, "a request that runs for a while [[fake:loop]]"), Logprobs: -1})
		switch {
		case errors.Is(err, ErrOverloaded):
			refused++
		case err != nil:
			t.Fatal(err)
		default:
			handles = append(handles, h)
		}
	}
	// One runs, at most two wait; whatever the loop had not yet admitted when
	// the submissions came counts as waiting, so the exact split varies.
	if refused < 2 || len(handles) > 4 {
		t.Fatalf("%d refused, %d accepted of 6 submissions with one slot and two waiting", refused, len(handles))
	}
	cancel()
}
