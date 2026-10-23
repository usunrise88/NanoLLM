package sched

import (
	"bytes"
	"context"
	"errors"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
)

// goid is the goroutine number from the stack header; the test compares the
// goroutine that steps the model with the one Do runs on.
func goid() int {
	var buf [64]byte
	b := bytes.TrimPrefix(buf[:runtime.Stack(buf[:], false)], []byte("goroutine "))
	n, _ := strconv.Atoi(string(b[:bytes.IndexByte(b, ' ')]))
	return n
}

// stepSpy records the goroutine of every Step.
type stepSpy struct {
	backend.Model
	mu  sync.Mutex
	ids map[int]bool
}

func (s *stepSpy) Step(ctx context.Context, steps []backend.SeqStep) ([]backend.SeqOut, error) {
	s.mu.Lock()
	s.ids[goid()] = true
	s.mu.Unlock()
	return s.Model.Step(ctx, steps)
}

func TestDoRunsOnTheLoopGoroutine(t *testing.T) {
	m, err := fake.New(fake.Options{}).Load(context.Background(), backend.LoadSpec{})
	if err != nil {
		t.Fatal(err)
	}
	spy := &stepSpy{Model: m, ids: map[int]bool{}}
	l := New(spy)
	t.Cleanup(func() { l.Close(); m.Close() })

	run(t, l, spy, Request{Prompt: prompt(spy, "hello [[fake:echo]]"), Logprobs: -1})
	var did int
	if err := l.Do(context.Background(), func(backend.Model) error { did = goid(); return nil }); err != nil {
		t.Fatal(err)
	}
	spy.mu.Lock()
	defer spy.mu.Unlock()
	if len(spy.ids) != 1 || !spy.ids[did] {
		t.Fatalf("Do ran on goroutine %d, Step on %v", did, spy.ids)
	}
}

func TestDoWaitsForTheRunningRequests(t *testing.T) {
	l, m := newLoop(t, fake.Options{DecodeRate: 200}, backend.LoadSpec{})
	h, err := l.Submit(context.Background(), Request{Prompt: prompt(m, "[[fake:loop]]"), MaxTokens: 30, Logprobs: -1})
	if err != nil {
		t.Fatal(err)
	}
	var busy atomicInt
	done := make(chan error, 1)
	go func() {
		done <- l.Do(context.Background(), func(backend.Model) error { busy.set(l.Busy()); return nil })
	}()
	for {
		ev, err := h.Next(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if ev.Done {
			break
		}
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if n := busy.get(); n != 0 {
		t.Fatalf("Do ran with %d requests in the loop", n)
	}
}

type atomicInt struct {
	mu sync.Mutex
	v  int
}

func (a *atomicInt) set(v int) { a.mu.Lock(); a.v = v; a.mu.Unlock() }
func (a *atomicInt) get() int  { a.mu.Lock(); defer a.mu.Unlock(); return a.v }

func TestDoReturnsWhatFnReturnsAndHonoursCancel(t *testing.T) {
	l, m := newLoop(t, fake.Options{DecodeRate: 100}, backend.LoadSpec{})
	boom := errors.New("boom")
	if err := l.Do(context.Background(), func(backend.Model) error { return boom }); err != boom {
		t.Fatalf("got %v", err)
	}
	// A request keeps the loop busy; a Do whose caller gives up before the
	// loop is idle never runs.
	if _, err := l.Submit(context.Background(), Request{Prompt: prompt(m, "[[fake:loop]]"), MaxTokens: 40, Logprobs: -1}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	ran := false
	if err := l.Do(ctx, func(backend.Model) error { ran = true; return nil }); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	time.Sleep(600 * time.Millisecond)
	if ran {
		t.Fatal("an abandoned Do ran")
	}
}

func TestDoOnAClosedLoop(t *testing.T) {
	m, _ := fake.New(fake.Options{}).Load(context.Background(), backend.LoadSpec{})
	l := New(m)
	l.Close()
	if err := l.Do(context.Background(), func(backend.Model) error { return nil }); !errors.Is(err, ErrClosed) {
		t.Fatalf("got %v", err)
	}
}

func TestSetDraftThroughDo(t *testing.T) {
	l, m := newLoop(t, fake.Options{MaxDraft: 3, Acceptance: 0.7}, backend.LoadSpec{DrafterPath: "drafter.gguf", DraftMax: 3})
	ask := func() Event {
		_, ev := run(t, l, m, Request{Prompt: prompt(m, "[[fake:loop]]"), MaxTokens: 60, Logprobs: -1})
		return ev
	}
	tuner := m.(backend.Tuner)
	set := func(max int) error {
		return l.Do(context.Background(), func(backend.Model) error { return tuner.SetDraft(max, 0) })
	}

	if ev := ask(); len(ev.Stats.DraftTried) != 3 {
		t.Fatalf("limit 3: tried %v", ev.Stats.DraftTried)
	}
	if err := set(1); err != nil {
		t.Fatal(err)
	}
	ev := ask()
	if len(ev.Stats.DraftTried) != 1 || ev.Stats.Drafted == 0 {
		t.Fatalf("limit 1: tried %v, drafted %d", ev.Stats.DraftTried, ev.Stats.Drafted)
	}
	if err := set(4); err == nil {
		t.Fatal("a limit above the loaded one was accepted")
	}
	if err := set(0); err != nil {
		t.Fatal(err)
	}
	if ev := ask(); ev.Stats.Drafted != 0 {
		t.Fatalf("drafting off, drafted %d", ev.Stats.Drafted)
	}
	if err := set(3); err != nil {
		t.Fatal(err)
	}
	if ev := ask(); ev.Stats.Drafted == 0 {
		t.Fatal("drafting did not come back")
	}
}

func TestAcceptedRunsAreConditionalPerPosition(t *testing.T) {
	l, m := newLoop(t, fake.Options{MaxDraft: 3, Acceptance: 0.7}, backend.LoadSpec{DrafterPath: "drafter.gguf", DraftMax: 3})
	_, ev := run(t, l, m, Request{Prompt: prompt(m, "[[fake:loop]]"), MaxTokens: 90, Logprobs: -1})
	st := ev.Stats
	if len(st.DraftTried) == 0 || len(st.DraftTried) != len(st.AcceptedRuns) {
		t.Fatalf("tried %v accepted %v", st.DraftTried, st.AcceptedRuns)
	}
	sum := 0
	for j := range st.DraftTried {
		if st.AcceptedRuns[j] > st.DraftTried[j] {
			t.Fatalf("position %d: %d accepted of %d tried", j+1, st.AcceptedRuns[j], st.DraftTried[j])
		}
		if j > 0 && st.DraftTried[j] > st.AcceptedRuns[j-1] {
			t.Fatalf("position %d was tried %d times, the one before was accepted %d", j+1, st.DraftTried[j], st.AcceptedRuns[j-1])
		}
		sum += st.AcceptedRuns[j]
	}
	if sum != st.Accepted {
		t.Fatalf("per-position accepted %d, total %d", sum, st.Accepted)
	}
}
