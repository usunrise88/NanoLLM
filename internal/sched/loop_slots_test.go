package sched

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
)

// What the loop does to a slot's cache when a request ends early, is cut short
// before it computes anything, or asks for a cache that cannot go back far
// enough. The invariant behind all of it: the loop's record of a slot
// (slot.cached) is exactly what the model holds for that sequence.

// hookModel wraps a model to watch what the loop asks of it and to break it on
// purpose. afterStep runs on the loop's goroutine, right after a real Step.
type hookModel struct {
	backend.Model

	mu         sync.Mutex
	afterStep  func(steps []backend.SeqStep)
	samplerErr error
	trims      int
	clears     int
}

func (h *hookModel) onStep(f func(steps []backend.SeqStep)) {
	h.mu.Lock()
	h.afterStep = f
	h.mu.Unlock()
}

func (h *hookModel) failSampler(err error) {
	h.mu.Lock()
	h.samplerErr = err
	h.mu.Unlock()
}

func (h *hookModel) calls() (trims, clears int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.trims, h.clears
}

func (h *hookModel) Step(ctx context.Context, steps []backend.SeqStep) ([]backend.SeqOut, error) {
	outs, err := h.Model.Step(ctx, steps)
	h.mu.Lock()
	f := h.afterStep
	h.mu.Unlock()
	if f != nil {
		f(steps)
	}
	return outs, err
}

func (h *hookModel) SetSampler(seq int, spec backend.SamplerSpec, history []backend.Token) error {
	h.mu.Lock()
	err := h.samplerErr
	h.mu.Unlock()
	if err != nil {
		return err
	}
	return h.Model.SetSampler(seq, spec, history)
}

func (h *hookModel) TrimSeq(seq, pos int) bool {
	h.mu.Lock()
	h.trims++
	h.mu.Unlock()
	return h.Model.TrimSeq(seq, pos)
}

func (h *hookModel) ClearSeq(seq int) {
	h.mu.Lock()
	h.clears++
	h.mu.Unlock()
	h.Model.ClearSeq(seq)
}

// newHookedLoop is newLoop with the model behind a hookModel. The returned
// model is the fake itself, for tokenizing and for reading slot state.
func newHookedLoop(t *testing.T, opts fake.Options, spec backend.LoadSpec) (*Loop, backend.Model, *hookModel) {
	t.Helper()
	m, err := fake.New(opts).Load(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	hm := &hookModel{Model: m}
	l := New(hm)
	t.Cleanup(func() { l.Close(); m.Close() })
	return l, m, hm
}

// mirrored checks the loop's record of a slot against the model's own and
// returns it. It reads loop state, so the slot must be idle and nothing may be
// assigned to it meanwhile.
func mirrored(t *testing.T, l *Loop, m backend.Model, seq int) []backend.Token {
	t.Helper()
	raw, err := m.SaveSeq(seq)
	if err != nil {
		t.Fatal(err)
	}
	var saved struct {
		KV []backend.Token `json:"kv"`
	}
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	cached := l.slots[seq].cached
	if !slices.Equal(saved.KV, cached) {
		t.Fatalf("slot %d: the loop believes it holds %d tokens, the model holds %d", seq, len(cached), len(saved.KV))
	}
	return cached
}

// drain reads a handle's events to the end and returns the tokens and the
// final event; it fails the test rather than hang.
func drain(t *testing.T, h *Handle) ([]backend.Token, Event) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var toks []backend.Token
	for {
		ev, err := h.Next(ctx)
		if err != nil {
			t.Fatalf("waiting for the request to end: %v", err)
		}
		toks = append(toks, ev.Tokens...)
		if ev.Done {
			return toks, ev
		}
	}
}

// start submits a request that is expected to run until it is cancelled, and
// returns once its first token arrived (its slot is decoding).
func start(t *testing.T, l *Loop, req Request) (*Handle, context.CancelFunc) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	h, err := l.Submit(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	wctx, wcancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer wcancel()
	if ev, err := h.Next(wctx); err != nil || ev.Done {
		t.Fatalf("the request did not start: %+v, %v", ev, err)
	}
	return h, cancel
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

// A request cancelled while its prompt is still being prefilled frees its
// slot, leaves the cache holding exactly the chunks that were evaluated, and
// the next request over the same prompt picks up from there. A neighbour
// decoding in the same batches never notices.
func TestCancelDuringPrefill(t *testing.T) {
	l, m, hm := newHookedLoop(t, fake.Options{DecodeRate: 500}, backend.LoadSpec{Slots: 2, NBatch: 16})
	neighbour, stopNeighbour := start(t, l, Request{Prompt: prompt(m, "keeps decoding [[fake:loop]]"), Logprobs: -1})

	body := strings.Repeat("abc ", 30)
	p := prompt(m, body+"[[fake:echo]]")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var chunks, evaluated atomic.Int32
	hm.onStep(func(steps []backend.SeqStep) {
		for _, st := range steps {
			if !st.Sample { // a prefill chunk that does not finish the prompt
				evaluated.Add(int32(len(st.Tokens)))
				if chunks.Add(1) == 3 {
					cancel()
				}
			}
		}
	})
	h, err := l.Submit(ctx, Request{Prompt: p, Logprobs: -1})
	if err != nil {
		t.Fatal(err)
	}
	toks, done := drain(t, h)
	if !errors.Is(done.Err, context.Canceled) || len(toks) != 0 {
		t.Fatalf("cancelled during prefill: %d tokens, %+v", len(toks), done)
	}
	if done.Stats == nil || done.Stats.Completion != 0 || done.Stats.CachedTokens != 0 || done.Stats.PromptTokens != len(p) {
		t.Fatalf("stats %+v", done.Stats)
	}
	if n := chunks.Load(); n != 3 {
		t.Fatalf("%d chunks were evaluated, want the 3 before the cancel", n)
	}
	seq, partial := done.Stats.Slot, int(evaluated.Load())
	if partial <= 0 || partial >= len(p) {
		t.Fatalf("%d of %d prompt tokens evaluated: not a partial prefill", partial, len(p))
	}
	if cached := mirrored(t, l, m, seq); !slices.Equal(cached, p[:partial]) {
		t.Fatalf("the cache holds %d tokens that are not the first %d of the prompt", len(cached), partial)
	}
	if got := l.Snapshot().Busy; got != 1 {
		t.Fatalf("%d slots busy after the cancel, want only the neighbour's", got)
	}

	// The neighbour went on decoding through the cancel.
	for range 5 {
		if ev, err := neighbour.Next(context.Background()); err != nil || ev.Done {
			t.Fatalf("the neighbour stopped: %+v, %v", ev, err)
		}
	}

	// The freed slot is the only one there is: the same prompt resumes from
	// the evaluated chunks and answers.
	text, ev := run(t, l, m, Request{Prompt: p, Logprobs: -1})
	if ev.Stats.Slot != seq || ev.Stats.CachedTokens != partial || text != strings.TrimSpace(body) {
		t.Fatalf("the retry: slot %d (want %d), %d cached (want %d), %q", ev.Stats.Slot, seq, ev.Stats.CachedTokens, partial, text)
	}
	mirrored(t, l, m, seq)
	stopNeighbour()
	if _, ev := drain(t, neighbour); !errors.Is(ev.Err, context.Canceled) {
		t.Fatalf("the neighbour ended with %v", ev.Err)
	}
}

// A request cancelled while it waits for a slot never gets one: it ends at
// once, leaves the queue, and the request holding the slot is undisturbed.
func TestCancelWhileWaiting(t *testing.T) {
	l, m := newLoop(t, fake.Options{DecodeRate: 200}, backend.LoadSpec{Slots: 1})
	holder, stopHolder := start(t, l, Request{Prompt: prompt(m, "holds the only slot [[fake:loop]]"), Logprobs: -1})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	waiting, err := l.Submit(ctx, Request{Prompt: prompt(m, "never runs [[fake:echo]]"), Logprobs: -1})
	if err != nil {
		t.Fatal(err)
	}
	// The loop has gone round a few times with the request waiting.
	steps := l.Snapshot().Steps
	eventually(t, "the loop to step with the request waiting", func() bool { return l.Snapshot().Steps >= steps+3 })
	eventually(t, "the holder and the waiting request to be counted", func() bool { return l.Busy() == 2 })
	cancel()
	toks, ev := drain(t, waiting)
	if !errors.Is(ev.Err, context.Canceled) || len(toks) != 0 {
		t.Fatalf("a request cancelled in the queue: %d tokens, %+v", len(toks), ev)
	}
	eventually(t, "the queue to empty", func() bool { return l.Snapshot().Pending == 0 })
	if got := l.Snapshot().Busy; got != 1 {
		t.Fatalf("%d slots busy, want the holder's only", got)
	}
	for range 3 { // the holder still decodes
		if ev, err := holder.Next(context.Background()); err != nil || ev.Done {
			t.Fatalf("the holder was disturbed: %+v, %v", ev, err)
		}
	}
	stopHolder()
	drain(t, holder)
	text, _ := run(t, l, m, Request{Prompt: prompt(m, "now it runs [[fake:echo]]"), Logprobs: -1})
	if text != "now it runs" {
		t.Fatalf("after the cancels: %q", text)
	}
}

// The operator's views of a loop (Busy, Snapshot) are read from other
// goroutines while it runs requests. Under -race this fails if they touch
// state the loop's goroutine owns.
func TestStatsAreSafeToReadUnderLoad(t *testing.T) {
	l, m := newLoop(t, fake.Options{DecodeRate: 5000}, backend.LoadSpec{Slots: 2, NBatch: 32})
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			run(t, l, m, Request{Prompt: prompt(m, strings.Repeat("abc ", 20)+" [[fake:long 40]]"), Logprobs: -1})
		}()
	}
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	for {
		select {
		case <-done:
			if got := l.Busy(); got != 0 {
				t.Fatalf("%d requests left after they all finished", got)
			}
			return
		default:
			_, _ = l.Busy(), l.Snapshot()
			time.Sleep(50 * time.Microsecond)
		}
	}
}

// A cancel between steps leaves the cache as the prompt plus every token the
// slot evaluated, so the continuation the engine builds after a stop sequence
// or a reasoning budget (the prompt, what was generated, a closing token)
// reuses all of it and evaluates only the pending token and the new ones.
func TestCancelThenContinue(t *testing.T) {
	l, m, hm := newHookedLoop(t, fake.Options{}, backend.LoadSpec{Slots: 1})
	p := prompt(m, "a reasoning model that thinks and thinks [[fake:loop]]")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	const decodeSteps = 6
	var decoded atomic.Int32
	hm.onStep(func(steps []backend.SeqStep) {
		for _, st := range steps {
			if st.Sample && len(st.Tokens) == 1 && decoded.Add(1) == decodeSteps {
				cancel()
			}
		}
	})
	h, err := l.Submit(ctx, Request{Prompt: p, Logprobs: -1})
	if err != nil {
		t.Fatal(err)
	}
	gen, done := drain(t, h)
	if !errors.Is(done.Err, context.Canceled) || len(gen) != decodeSteps+1 {
		t.Fatalf("%d tokens after %d decode steps, %+v", len(gen), decodeSteps, done)
	}
	// The newest token has not been evaluated yet; everything before it has.
	want := append(slices.Clone(p), gen[:len(gen)-1]...)
	if cached := mirrored(t, l, m, 0); !slices.Equal(cached, want) {
		t.Fatalf("the cache holds %d tokens, want the prompt and the %d evaluated ones (%d)", len(cached), decodeSteps, len(want))
	}
	closing, _ := m.Tokenize("<channel|>", false, true)
	next := append(append(slices.Clone(p), gen...), closing...)
	_, ev := run(t, l, m, Request{Prompt: next, MaxTokens: 3, Logprobs: -1})
	if ev.Stats.CachedTokens != len(want) || ev.Stats.PromptTokens != len(next) {
		t.Fatalf("the continuation reused %d of %d tokens, want %d", ev.Stats.CachedTokens, ev.Stats.PromptTokens, len(want))
	}
	mirrored(t, l, m, 0)
}

// A session keeps one slot. While that slot is busy a second request of the
// session runs elsewhere, but must not take the label with it: when both are
// done, the session's next turn belongs on the slot that holds its
// conversation, not on the sibling's.
func TestBusySessionDoesNotSpreadItsLabel(t *testing.T) {
	l, m := newLoop(t, fake.Options{DecodeRate: 200}, backend.LoadSpec{Slots: 3})
	// A filler holds slot 0, so the session starts on slot 1.
	filler, stopFiller := start(t, l, Request{Prompt: prompt(m, "filler [[fake:loop]]"), Logprobs: -1})
	_, first := run(t, l, m, Request{Prompt: prompt(m, "the session starts here [[fake:echo]]"), SessionID: "s1", Logprobs: -1})
	home := first.Stats.Slot
	if home != 1 {
		t.Fatalf("the session started on slot %d, want 1", home)
	}
	conv := prompt(m, "the conversation so far [[fake:loop]]")
	turn, stopTurn := start(t, l, Request{Prompt: conv, SessionID: "s1", Logprobs: -1})

	stopFiller()
	drain(t, filler)
	text, sibling := run(t, l, m, Request{Prompt: prompt(m, "a sibling request [[fake:echo]]"), SessionID: "s1", Logprobs: -1})
	if text != "a sibling request" || sibling.Stats.Slot != 0 {
		t.Fatalf("the sibling ran on slot %d (want the free slot 0): %q", sibling.Stats.Slot, text)
	}

	stopTurn()
	drain(t, turn)
	followUp := append(slices.Clone(conv), prompt(m, "and the next turn [[fake:echo]]")...)
	_, ev := run(t, l, m, Request{Prompt: followUp, SessionID: "s1", Logprobs: -1})
	if ev.Stats.Slot != home || ev.Stats.CachedTokens != len(conv) {
		t.Fatalf("the next turn ran on slot %d with %d tokens cached; want slot %d with the conversation's %d",
			ev.Stats.Slot, ev.Stats.CachedTokens, home, len(conv))
	}
}

// What the loop does when the cache cannot go back as far as a request needs:
// with swa_full the shared prefix stays; without it, a trim past the sliding
// window fails and the slot starts over. Either way the loop's record and the
// model's cache agree afterwards, and the answer is right.
func TestTrimThroughTheLoop(t *testing.T) {
	for _, tc := range []struct {
		name    string
		swaFull bool
	}{
		{"swa_full keeps the shared prefix", true},
		{"a trim past the window clears the slot", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l, m, hm := newHookedLoop(t, fake.Options{}, backend.LoadSpec{Slots: 1, SWAFull: tc.swaFull})
			// Far longer than the fake's 512-token window.
			long := prompt(m, strings.Repeat("x", 700)+" [[fake:echo]]")
			run(t, l, m, Request{Prompt: long, Logprobs: -1})
			if got := len(mirrored(t, l, m, 0)); got <= len(long) {
				t.Fatalf("the cache holds %d tokens, want the prompt and its answer", got)
			}
			// Shares only the turn header with it, so the trim goes back ~700.
			other := prompt(m, "y"+strings.Repeat("z", 20)+" [[fake:echo]]")
			shared := lcp(long, other)
			trims0, clears0 := hm.calls()
			text, ev := run(t, l, m, Request{Prompt: other, Logprobs: -1})
			if text != "y"+strings.Repeat("z", 20) {
				t.Fatalf("answer %q", text)
			}
			wantCached, wantClears := shared, 0
			if !tc.swaFull {
				wantCached, wantClears = 0, 1
			}
			trims, clears := hm.calls()
			if ev.Stats.CachedTokens != wantCached || trims-trims0 != 1 || clears-clears0 != wantClears {
				t.Fatalf("%d tokens cached (want %d), %d trims (want 1), %d clears (want %d)",
					ev.Stats.CachedTokens, wantCached, trims-trims0, clears-clears0, wantClears)
			}
			cached := mirrored(t, l, m, 0)
			if len(cached) < len(other) || !slices.Equal(cached[:len(other)], other) {
				t.Fatalf("the cache does not start with the prompt it just served")
			}
		})
	}
}

// A request that fails leaves nothing behind: after an error struck mid-step,
// or the sampler could not be set, the cache may differ from what the slot
// mirrors, so the slot is emptied on both sides and the next request over
// the same prompt prefills it whole.
func TestFailureClearsTheSlot(t *testing.T) {
	const warm = "a prompt that is worth caching [[fake:echo]]"
	for _, tc := range []struct {
		name   string
		prompt string
		setup  func(hm *hookModel)
	}{
		{"a step error", "a prompt that is worth caching [[fake:error]]", func(*hookModel) {}},
		{"a sampler that cannot be set", warm, func(hm *hookModel) { hm.failSampler(errors.New("bad grammar")) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l, m, hm := newHookedLoop(t, fake.Options{}, backend.LoadSpec{Slots: 1})
			run(t, l, m, Request{Prompt: prompt(m, warm), Logprobs: -1})
			if len(mirrored(t, l, m, 0)) == 0 {
				t.Fatal("setup left the slot empty")
			}
			tc.setup(hm)
			_, ev := run(t, l, m, Request{Prompt: prompt(m, tc.prompt), Logprobs: -1})
			if ev.Err == nil {
				t.Fatal("the request did not fail")
			}
			if cached := mirrored(t, l, m, 0); len(cached) != 0 {
				t.Fatalf("a failed request left %d tokens in the slot", len(cached))
			}
			hm.failSampler(nil)
			text, ev := run(t, l, m, Request{Prompt: prompt(m, warm), Logprobs: -1})
			if ev.Err != nil || ev.Stats.CachedTokens != 0 || text != "a prompt that is worth caching" {
				t.Fatalf("after the failure: %d tokens cached, %q, %v", ev.Stats.CachedTokens, text, ev.Err)
			}
			mirrored(t, l, m, 0)
		})
	}
}
