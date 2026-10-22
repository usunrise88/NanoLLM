package main

import "testing"

func TestAnswerAndWhy(t *testing.T) {
	for _, c := range []struct {
		name, body string
		calls      int
		why        string
	}{
		{"a call", `{"choices":[{"finish_reason":"tool_calls","message":{"content":"","tool_calls":[{"id":"1"}]}}]}`, 1, "empty"},
		{"a null list", `{"choices":[{"finish_reason":"stop","message":{"content":"Sunny.","tool_calls":null}}]}`, 0, "text instead of a call"},
		{"reasoning without room left", `{"choices":[{"finish_reason":"length","message":{"content":"","reasoning_content":"1. Analyse"}}]}`, 0, "budget spent on reasoning"},
		{"an answer cut short", `{"choices":[{"finish_reason":"length","message":{"content":"The weather"}}]}`, 0, "budget spent on the answer"},
		{"not JSON", `oops`, 0, "unreadable"},
		{"no choices", `{"choices":[]}`, 0, "unreadable"},
	} {
		a := answerOf([]byte(c.body))
		if a.toolCalls != c.calls || a.why() != c.why {
			t.Errorf("%s: %d calls, why %q", c.name, a.toolCalls, a.why())
		}
	}
}

func TestClaimSharesTheBudget(t *testing.T) {
	g := &gen{limit: 3}
	n := 0
	for g.claim() {
		n++
	}
	if n != 3 || g.claim() {
		t.Fatalf("claimed %d", n)
	}
	g = &gen{} // no -n: never runs out
	for range 1000 {
		if !g.claim() {
			t.Fatal("an unlimited run ran out")
		}
	}
}
