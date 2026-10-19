package engine

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/pool"
)

// A request the engine ends itself (a stop sequence, the reasoning budget) or
// the client abandons is cancelled in the scheduler, which keeps the slot's
// cache: the next turn, or the continuation after a budget, starts from it.

// waitIdle waits until no request holds the (only) instance and returns its
// status. A request that ended from the client's side returns before the loop
// has let go of its slot.
func waitIdle(t *testing.T, s *Service) pool.Status {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		st := s.pool.Status()
		if len(st) == 1 && st[0].Refs == 0 && st[0].Busy == 0 {
			return st[0]
		}
		if time.Now().After(deadline) {
			t.Fatalf("the instance never went idle: %+v", st)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestStopSequenceKeepsTheCache(t *testing.T) {
	s := newService(t, Config{}, fake.Options{DecodeRate: 2000}, backend.LoadSpec{SWAFull: true})
	ctx := context.Background()
	first := user("an answer that never ends [[fake:loop]]")
	one, err := s.Chat(ctx, core.ChatRequest{Messages: first, Generation: core.Generation{Stop: []string{" la la"}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The script is "la la la ..." forever: only the stop sequence ends it.
	if one.Message.Content != "la" || one.FinishReason != core.FinishStop {
		t.Fatalf("first turn: %q (%s)", one.Message.Content, one.FinishReason)
	}
	waitIdle(t, s)

	// The second turn renders the answer back into the history; the cache
	// holds the prompt and the answer, so only the new turn is evaluated.
	turn := append(first, core.Message{Role: core.RoleAssistant, Content: one.Message.Content},
		core.Message{Role: core.RoleUser, Content: "go on [[fake:echo]]"})
	two, err := s.Chat(ctx, core.ChatRequest{Messages: turn}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := one.Usage.PromptTokens + len(one.Message.Content); two.Usage.CachedTokens != want || two.Message.Content != "go on" {
		t.Fatalf("second turn: %d tokens cached, want %d (the first prompt and its answer); %q", two.Usage.CachedTokens, want, two.Message.Content)
	}
}

// The budget ends the request and a second one continues it with the channel
// closed. If the cancel cleared the slot, the continuation would read the
// whole prompt again; the loop's step counter shows what was evaluated.
func TestReasoningBudgetContinuationReusesTheCache(t *testing.T) {
	s := newService(t, Config{}, fake.Options{DecodeRate: 2000}, backend.LoadSpec{SWAFull: true})
	ctx := context.Background()
	req := core.ChatRequest{
		Messages:           user(strings.Repeat("x", 300) + " [[fake:raw <|channel>thought\nthinking forever and ever and ever<channel|>answer]]"),
		MaxReasoningTokens: 5,
	}
	res, err := s.Chat(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.Content != "answer" || res.Usage.ReasoningTokens != 5 {
		t.Fatalf("budget not enforced: %+v %+v", res.Message, res.Usage)
	}
	prompt := res.Usage.PromptTokens
	// The prompt once, a decode step per generated token, the closing token:
	// a few dozen tokens beyond the prompt. A second read of it is 300 more.
	const slack = 60
	st := waitIdle(t, s)
	if st.Loop.StepTokens > int64(prompt+slack) {
		t.Fatalf("%d tokens evaluated for a prompt of %d: the continuation read it again", st.Loop.StepTokens, prompt)
	}

	// The same request again starts from the cache the first one left, and
	// reports that start, not the continuation's.
	again, err := s.Chat(ctx, req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if again.Usage.CachedTokens != prompt-1 {
		t.Fatalf("the repeat reused %d of %d prompt tokens", again.Usage.CachedTokens, prompt)
	}
	if used := waitIdle(t, s).Loop.StepTokens - st.Loop.StepTokens; used > slack {
		t.Fatalf("the repeat evaluated %d tokens", used)
	}
}

// A client that leaves mid-answer (its connection closed, or the sink failed)
// does not cost the next request its prefix.
func TestClientGoneKeepsTheCache(t *testing.T) {
	boom := errors.New("client went away")
	for _, tc := range []struct {
		name string
		gone func(cancel context.CancelFunc) core.Sink
		want error
	}{
		{"the sink fails", func(context.CancelFunc) core.Sink { return func(core.Delta) error { return boom } }, boom},
		{"the context is cancelled", func(cancel context.CancelFunc) core.Sink {
			return func(core.Delta) error { cancel(); return nil }
		}, context.Canceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newService(t, Config{}, fake.Options{DecodeRate: 1000}, backend.LoadSpec{SWAFull: true})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			msgs := user(strings.Repeat("a long conversation ", 10) + "[[fake:loop]]")
			if _, err := s.Chat(ctx, core.ChatRequest{Messages: msgs}, tc.gone(cancel)); !errors.Is(err, tc.want) {
				t.Fatalf("got %v, want %v", err, tc.want)
			}
			waitIdle(t, s)
			res, err := s.Chat(context.Background(), core.ChatRequest{Messages: msgs, Generation: core.Generation{MaxTokens: 3}}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if res.Usage.CachedTokens != res.Usage.PromptTokens-1 {
				t.Fatalf("the next request reused %d of %d prompt tokens", res.Usage.CachedTokens, res.Usage.PromptTokens)
			}
		})
	}
}

// Requests that end every way at once on a two-slot model: nothing is left
// holding a slot or the model, and the model still answers.
func TestEndingsDoNotLeakSlots(t *testing.T) {
	s := newService(t, Config{}, fake.Options{DecodeRate: 3000}, backend.LoadSpec{Slots: 2, SWAFull: true})
	boom := errors.New("client went away")
	kinds := []struct {
		name string
		run  func(i int) error
	}{
		{"complete", func(i int) error {
			res, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("fine [[fake:echo]]")}, nil)
			if err == nil && res.Message.Content != "fine" {
				err = errors.New("content " + res.Message.Content)
			}
			return err
		}},
		{"stop sequence", func(i int) error {
			res, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("go [[fake:loop]]"), Generation: core.Generation{Stop: []string{" la la"}}}, nil)
			if err == nil && res.Message.Content != "la" {
				err = errors.New("content " + res.Message.Content)
			}
			return err
		}},
		{"reasoning budget", func(i int) error {
			res, err := s.Chat(context.Background(), core.ChatRequest{
				Messages:           user("think [[fake:raw <|channel>thought\nthinking forever and ever and ever<channel|>answer]]"),
				MaxReasoningTokens: 4,
			}, nil)
			if err == nil && res.Message.Content != "answer" {
				err = errors.New("content " + res.Message.Content)
			}
			return err
		}},
		{"context cancelled", func(i int) error {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			_, err := s.Chat(ctx, core.ChatRequest{Messages: user("leave [[fake:loop]]")}, func(core.Delta) error { cancel(); return nil })
			if !errors.Is(err, context.Canceled) {
				return errors.New("a cancelled request ended with " + errString(err))
			}
			return nil
		}},
		{"sink fails", func(i int) error {
			_, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("leave [[fake:loop]]")}, func(core.Delta) error { return boom })
			if !errors.Is(err, boom) {
				return errors.New("a failing sink ended with " + errString(err))
			}
			return nil
		}},
		{"session turns", func(i int) error {
			sess := core.Generation{SessionID: "s" + string(rune('0'+i%3)), MaxTokens: 8}
			_, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("turn " + string(rune('a'+i)) + " [[fake:loop]]"), Generation: sess}, nil)
			return err
		}},
	}
	var wg sync.WaitGroup
	for i := range 4 {
		for _, k := range kinds {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if err := k.run(i); err != nil {
					t.Errorf("%s: %v", k.name, err)
				}
			}()
		}
	}
	wg.Wait()
	st := waitIdle(t, s)
	if st.Loop.Pending != 0 || st.Loop.Busy != 0 {
		t.Fatalf("loop left holding requests: %+v", st.Loop)
	}
	res, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("still alive [[fake:echo]]")}, nil)
	if err != nil || res.Message.Content != "still alive" {
		t.Fatalf("after the churn: %v %+v", err, res)
	}
}

func errString(err error) string {
	if err == nil {
		return "no error"
	}
	return err.Error()
}
