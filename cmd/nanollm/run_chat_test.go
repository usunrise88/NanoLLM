package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/core"
)

type echoService struct {
	core.Service
	reqs []core.ChatRequest
}

func (e *echoService) Chat(_ context.Context, req core.ChatRequest, sink core.Sink) (*core.ChatResult, error) {
	e.reqs = append(e.reqs, req)
	if *req.EnableThinking {
		_ = sink(core.Delta{ReasoningContent: "hmm"})
	}
	answer := "re: " + req.Messages[len(req.Messages)-1].Content
	_ = sink(core.Delta{Content: answer})
	return &core.ChatResult{Message: core.Message{Role: core.RoleAssistant, Content: answer}}, nil
}

func TestREPLKeepsHistoryAndCommands(t *testing.T) {
	svc := &echoService{}
	var out, errw bytes.Buffer
	s := &chatSession{svc: svc, model: "m", opts: chatOptions{system: "be brief"}, out: &out, errw: &errw}
	s.history = []core.Message{{Role: core.RoleSystem, Content: "be brief"}}
	in := strings.NewReader("one\n/think on\ntwo\n/reset\nthree\n/nope\n")
	if err := s.repl(context.Background(), in); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); got != "re: one\nre: two\nre: three\n" {
		t.Fatalf("answers %q", got)
	}
	if len(svc.reqs) != 3 {
		t.Fatalf("%d requests", len(svc.reqs))
	}
	// The second turn carries the first; after /reset only the system message
	// is left.
	if n := len(svc.reqs[1].Messages); n != 4 || !*svc.reqs[1].EnableThinking {
		t.Fatalf("second turn: %+v", svc.reqs[1])
	}
	if m := svc.reqs[2].Messages; len(m) != 2 || m[0].Role != core.RoleSystem {
		t.Fatalf("after reset: %+v", m)
	}
	if !strings.Contains(errw.String(), "hmm") || !strings.Contains(errw.String(), "unknown command /nope") {
		t.Fatalf("stderr %q", errw.String())
	}
}
