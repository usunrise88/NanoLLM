package grammar

import (
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/core"
)

func TestGemma4JSONWrapsAndRenamesRoot(t *testing.T) {
	for _, inner := range []Spec{JSONObject(), mustSchema(t, `{"type":"object","properties":{"root":{"type":"string"},"ws":{"type":"integer"}}}`)} {
		s := Gemma4JSON(inner)
		if strings.Count(s.GBNF, "\nroot ::=")+strings.Count(s.GBNF[:10], "root ::=") != 1 {
			t.Fatalf("exactly one root rule:\n%s", s.GBNF)
		}
		if !strings.HasPrefix(s.GBNF, "root ::= \"```json\" nanollm-ws nanollm-answer nanollm-ws \"```\"\n") {
			t.Fatalf("%s", s.GBNF)
		}
		if !strings.Contains(s.GBNF, "\nnanollm-answer ::= ") {
			t.Fatalf("inner root not renamed:\n%s", s.GBNF)
		}
	}
}

func mustSchema(t *testing.T, schema string) Spec {
	t.Helper()
	s, err := FromSchema([]byte(schema))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestGemma4Tools(t *testing.T) {
	tools := []core.Tool{{Name: "get_weather"}, {Name: "search.web"}}
	lazy, err := Gemma4Tools(tools, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(lazy.Triggers) != 1 || lazy.Triggers[0] != "<|tool_call>" || !strings.HasPrefix(lazy.GBNF, "root ::= g4-call+\n") {
		t.Fatalf("%+v", lazy)
	}
	if !strings.Contains(lazy.GBNF, `g4-name ::= "get_weather" | "search.web"`) {
		t.Fatal(lazy.GBNF)
	}
	req, _ := Gemma4Tools(tools, true, false)
	if req.Triggers != nil || !strings.HasPrefix(req.GBNF, "root ::= g4-thought? g4-call\n") {
		t.Fatalf("%+v", req)
	}
	if _, err := Gemma4Tools([]core.Tool{{Name: `bad"name`}}, true, true); err == nil {
		t.Fatal("a name that would break the grammar must be refused")
	}
}

func TestFromSchemaWarnsOnUnsupportedPattern(t *testing.T) {
	s := mustSchema(t, `{"type":"string","pattern":"^(?=a)b$"}`)
	if len(s.Warnings) != 1 {
		t.Fatalf("%+v", s)
	}
}
