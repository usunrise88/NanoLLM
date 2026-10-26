package engine

import (
	"context"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/core"
)

var harnessTools = []core.Tool{{Name: "get_time", Description: "The current time."}}

// malformedCall is a tool call whose arguments do not parse.
const malformedCall = "[[fake:raw <|tool_call>call:get_time{ broken<tool_call|>]]"

func TestRescueReadsACallFromTheTextAndReportsOK(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	res, err := s.Chat(context.Background(), core.ChatRequest{
		Messages: user("[[fake:tool get_time {}]]"), Tools: harnessTools, ToolFormat: core.ToolFormatRescue,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Parse == nil || res.Parse.Status != core.ParseOK || len(res.Message.ToolCalls) != 1 {
		t.Fatalf("parse %+v, calls %v", res.Parse, res.Message.ToolCalls)
	}
}

func TestRescueKeepsAMalformedCallAsTextAndReportsIt(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	res, err := s.Chat(context.Background(), core.ChatRequest{
		Messages: user(malformedCall), Tools: harnessTools, ToolFormat: core.ToolFormatRescue,
	}, nil)
	if err != nil {
		t.Fatalf("rescue failed the request: %v", err)
	}
	p := res.Parse
	if p == nil || p.Status != core.ParseFailed || p.Error == "" || len(res.Message.ToolCalls) != 0 {
		t.Fatalf("parse %+v, calls %v", p, res.Message.ToolCalls)
	}
	if p.Raw == "" || res.Message.Content != p.Raw {
		t.Fatalf("the reply holds %q and the report %q, want the same whole text", res.Message.Content, p.Raw)
	}
}

func TestTheDefaultFormatStillFailsOnAMalformedCall(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	if _, err := s.Chat(context.Background(), core.ChatRequest{Messages: user(malformedCall), Tools: harnessTools}, nil); err == nil {
		t.Fatal("a malformed call did not fail the request outside rescue")
	}
}

func TestRescueRefusesARequiredToolChoice(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	_, err := s.Chat(context.Background(), core.ChatRequest{
		Messages: user("Look it up."), Tools: harnessTools, ToolFormat: core.ToolFormatRescue,
		ToolChoice: &core.ToolChoice{Mode: core.ToolChoiceRequired},
	}, nil)
	if err == nil {
		t.Fatal("rescue accepted tool_choice required, which needs the grammar it does not apply")
	}
}

func TestAnUnknownToolFormatIsRefused(t *testing.T) {
	s := newService(t, Config{}, fake.Options{}, backend.LoadSpec{})
	if _, err := s.Chat(context.Background(), core.ChatRequest{Messages: user("Hi."), ToolFormat: "loose"}, nil); err == nil {
		t.Fatal("an unknown tool format was accepted")
	}
}

func TestRescueAppliesNoToolGrammarAndStrictDoes(t *testing.T) {
	rescue := &gen{rescue: true}
	if err := rescue.constrain(core.Generation{}, harnessTools, core.ToolChoiceAuto); err != nil {
		t.Fatal(err)
	}
	if rescue.sampler.Grammar != "" {
		t.Fatal("rescue applied a tool grammar")
	}
	strict := &gen{}
	if err := strict.constrain(core.Generation{}, harnessTools, core.ToolChoiceAuto); err != nil {
		t.Fatal(err)
	}
	if strict.sampler.Grammar == "" {
		t.Fatal("strict applied no tool grammar")
	}
}
