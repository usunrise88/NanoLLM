package engine

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/pool"
)

func newService(t *testing.T, cfg Config, opts fake.Options, spec backend.LoadSpec) *Service {
	t.Helper()
	p := pool.New(func(ctx context.Context, k pool.Key) (backend.Model, any, error) {
		s := spec
		if s.NCtx == 0 {
			s.NCtx = k.NCtx
		}
		m, err := fake.New(opts).Load(ctx, s)
		return m, nil, err
	}, 2)
	t.Cleanup(p.Close)
	if cfg.DefaultModel == "" {
		cfg.DefaultModel = "tiny"
	}
	cfg.PromptCache = true
	return New(cfg, p, func(ctx context.Context, model, device string) (Target, error) {
		if model != "tiny" {
			return Target{}, core.Errorf(core.CodeModelNotFound, "no model %q", model)
		}
		return Target{Key: pool.Key{Model: model, Backend: "fake", Device: "cpu", NCtx: 4096}}, nil
	})
}

func user(s string) []core.Message { return []core.Message{{Role: core.RoleUser, Content: s}} }

func TestChatStreamsAndCollects(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	var streamed strings.Builder
	res, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("Привет, мир [[fake:echo]]")}, func(d core.Delta) error {
		streamed.WriteString(d.Content)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.Content != "Привет, мир" || streamed.String() != res.Message.Content {
		t.Fatalf("content %q streamed %q", res.Message.Content, streamed.String())
	}
	if res.FinishReason != core.FinishStop || res.Usage.CompletionTokens != len("Привет, мир") {
		t.Fatalf("finish %s usage %+v", res.FinishReason, res.Usage)
	}
	if res.Stats.Backend != "fake" || res.Stats.StagesMS[core.StageTemplate] < 0 {
		t.Fatalf("stats %+v", res.Stats)
	}
}

func TestChatReasoningAndTools(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	res, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("[[fake:think]]")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.ReasoningContent == "" || res.Message.Content != "Here is a short answer." || res.Usage.ReasoningTokens == 0 {
		t.Fatalf("%+v %+v", res.Message, res.Usage)
	}

	tools := []core.Tool{{Name: "get_weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}}
	res, err = s.Chat(context.Background(), core.ChatRequest{
		Messages: user(`[[fake:tool get_weather {city:<|"|>Paris<|"|>}]]`), Tools: tools,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != core.FinishToolCalls || len(res.Message.ToolCalls) != 1 {
		t.Fatalf("%+v", res)
	}
	c := res.Message.ToolCalls[0]
	if c.Name != "get_weather" || c.Arguments != `{"city":"Paris"}` || !strings.HasPrefix(c.ID, "call_") {
		t.Fatalf("call %+v", c)
	}
}

func TestStopSequences(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	var streamed strings.Builder
	res, err := s.Chat(context.Background(), core.ChatRequest{
		Messages:   user("alpha beta gamma [[fake:echo]]"),
		Generation: core.Generation{Stop: []string{"xx", " gam"}},
	}, func(d core.Delta) error { streamed.WriteString(d.Content); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.Content != "alpha beta" || streamed.String() != "alpha beta" || res.FinishReason != core.FinishStop {
		t.Fatalf("%q %q %s", res.Message.Content, streamed.String(), res.FinishReason)
	}
}

func TestMaxTokensAndCap(t *testing.T) {
	s := newService(t, Config{MaxTokensCap: 12}, fake.Options{}, backend.LoadSpec{})
	res, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("[[fake:loop]]")}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != core.FinishLength || res.Usage.CompletionTokens != 12 {
		t.Fatalf("%s %+v", res.FinishReason, res.Usage)
	}
	res, _ = s.Chat(context.Background(), core.ChatRequest{Messages: user("[[fake:loop]]"), Generation: core.Generation{MaxTokens: 4}}, nil)
	if res.Message.Content != "la l" {
		t.Fatalf("%q", res.Message.Content)
	}
}

func TestReasoningBudget(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	res, err := s.Chat(context.Background(), core.ChatRequest{
		Messages: user("[[fake:raw <|channel>thought\nthinking forever and ever and ever<channel|>answer]]"), MaxReasoningTokens: 5,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Message.ReasoningContent) > 6 || res.Usage.ReasoningTokens != 5 || res.Message.Content != "answer" {
		t.Fatalf("budget not enforced: %q (%d tokens), answer %q", res.Message.ReasoningContent, res.Usage.ReasoningTokens, res.Message.Content)
	}
}

func TestContextOverflow(t *testing.T) {
	long := user(strings.Repeat("x", 5000))
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{NBatch: 8192})
	_, err := s.Chat(context.Background(), core.ChatRequest{Messages: long}, nil)
	var ce *core.Error
	if !errors.As(err, &ce) || ce.Code != core.CodeContextOverflow {
		t.Fatalf("expected context_overflow, got %v", err)
	}
	s = newService(t, Config{ContextOverflow: OverflowTruncateMiddle, MaxTokensCap: 16}, fake.Options{}, backend.LoadSpec{NBatch: 8192})
	res, err := s.Chat(context.Background(), core.ChatRequest{Messages: long}, nil)
	if err != nil || len(res.Warnings) == 0 {
		t.Fatalf("truncate: %v %+v", err, res)
	}
}

func TestLogprobsAndErrors(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	res, err := s.Chat(context.Background(), core.ChatRequest{
		Messages:   user("ab [[fake:echo]]"),
		Generation: core.Generation{Logprobs: true, TopLogprobs: 2},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Logprobs) != 2 || res.Logprobs[0].Token != "a" || len(res.Logprobs[0].Top) != 2 {
		t.Fatalf("logprobs %+v", res.Logprobs)
	}
	_, err = s.Chat(context.Background(), core.ChatRequest{Messages: user("x"), Generation: core.Generation{Model: "nope"}}, nil)
	var ce *core.Error
	if !errors.As(err, &ce) || ce.Code != core.CodeModelNotFound {
		t.Fatalf("unknown model: %v", err)
	}
	_, err = s.Chat(context.Background(), core.ChatRequest{Messages: user("x"), Generation: core.Generation{
		ResponseFormat: &core.ResponseFormat{Type: core.FormatJSONSchema, Schema: json.RawMessage(`{"type":"kaboom"}`)}}}, nil)
	if !errors.As(err, &ce) || ce.Code != core.CodeInvalidRequest || ce.Param != "response_format" {
		t.Fatalf("a bad schema: %v", err)
	}
	res, err = s.Chat(context.Background(), core.ChatRequest{Messages: user("[[fake:raw ```json\n{\"a\":1}```]]"), Generation: core.Generation{
		ResponseFormat: &core.ResponseFormat{Type: core.FormatJSONSchema, Schema: json.RawMessage(`{"type":"object","properties":{"a":{"type":"integer"}}}`)}}}, nil)
	if err != nil || res.Message.Content != `{"a":1}` {
		t.Fatalf("schema answer: %v %+v", err, res)
	}
	_, err = s.Chat(context.Background(), core.ChatRequest{Messages: user("[[fake:error]]")}, nil)
	if !errors.As(err, &ce) || ce.Code != core.CodeInternal {
		t.Fatalf("backend failure: %v", err)
	}
}

func TestJSONObjectFence(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	res, err := s.Chat(context.Background(), core.ChatRequest{
		Messages:   user("[[fake:raw ```json\n{\"a\": 1}\n```]]"),
		Generation: core.Generation{ResponseFormat: &core.ResponseFormat{Type: core.FormatJSONObject}},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Message.Content != `{"a": 1}` {
		t.Fatalf("%q", res.Message.Content)
	}
}

func TestSinkErrorStops(t *testing.T) {
	s := newService(t, Config{}, fake.Options{DecodeRate: 1000}, backend.LoadSpec{})
	boom := errors.New("client went away")
	_, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("[[fake:loop]]")}, func(core.Delta) error { return boom })
	if !errors.Is(err, boom) {
		t.Fatalf("%v", err)
	}
	// The slot is free for the next request.
	res, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("ok [[fake:echo]]")}, nil)
	if err != nil || res.Message.Content != "ok" {
		t.Fatalf("%v %+v", err, res)
	}
}

func TestCompleteAndTokenize(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	res, err := s.Complete(context.Background(), core.CompletionRequest{Prompt: "raw [[fake:json]]"}, nil)
	if err != nil || res.Text != `{"ok":true,"items":[1,2,3],"name":"fake"}` {
		t.Fatalf("%v %+v", err, res)
	}
	tr, err := s.Tokenize(context.Background(), core.TokenizeRequest{Text: "abc"})
	if err != nil || tr.Count != 3 {
		t.Fatalf("%v %+v", err, tr)
	}
	dr, err := s.Detokenize(context.Background(), core.DetokenizeRequest{Tokens: tr.Tokens})
	if err != nil || dr.Text != "abc" {
		t.Fatalf("%v %+v", err, dr)
	}
	if _, err := s.Detokenize(context.Background(), core.DetokenizeRequest{Tokens: []int{-1}}); err == nil {
		t.Fatal("negative token accepted")
	}
}

func TestUTF8Stream(t *testing.T) {
	u := &utf8Stream{}
	b := []byte("aПр€😀")
	var got string
	for _, c := range b {
		got += u.write(string([]byte{c}))
	}
	got += u.flush()
	if got != string(b) {
		t.Fatalf("%q", got)
	}
}

// Finished requests are counted by outcome, with their tokens: the
// operator's /metrics and the soak test read these.
func TestMetricsCountRequests(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	if _, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("count me [[fake:echo]]")}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Complete(context.Background(), core.CompletionRequest{Prompt: "a raw prompt [[fake:echo]]"}, nil); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Chat(ctx, core.ChatRequest{Messages: user("never runs")}, nil); err == nil {
		t.Fatal("a cancelled request succeeded")
	}
	if _, err := s.Chat(context.Background(), core.ChatRequest{Generation: core.Generation{Model: "nope"}, Messages: user("x")}, nil); err == nil {
		t.Fatal("an unknown model succeeded")
	}
	snap := s.Metrics.Snapshot()
	if snap.Requests["chat/ok"] != 1 || snap.Requests["complete/ok"] != 1 {
		t.Errorf("requests %v", snap.Requests)
	}
	if snap.Tokens.Prompt == 0 || snap.Tokens.Completion == 0 {
		t.Errorf("tokens %+v", snap.Tokens)
	}
	if snap.Requests["chat/cancelled"] != 1 {
		t.Errorf("the cancelled request is counted as cancelled: %v", snap.Requests)
	}
	if snap.Requests["chat/error"] != 0 {
		t.Errorf("a cancelled or refused request counted as an error: %v", snap.Requests)
	}
	if snap.LatencyMS["total"].Count != 3 {
		t.Errorf("total latency observations %d, want the 3 that reached generation", snap.LatencyMS["total"].Count)
	}
}
