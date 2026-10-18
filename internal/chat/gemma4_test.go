package chat

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/usunrise88/nanollm/internal/core"
)

// fixtureCase is one OpenAI-shaped conversation from testdata/gemma4_cases.json.
type fixtureCase struct {
	Name                string `json:"name"`
	EnableThinking      bool   `json:"enable_thinking"`
	AddGenerationPrompt *bool  `json:"add_generation_prompt"`
	Messages            []struct {
		Role             string  `json:"role"`
		Content          *string `json:"content"`
		ReasoningContent string  `json:"reasoning_content"`
		ToolCallID       string  `json:"tool_call_id"`
		Name             string  `json:"name"`
		ToolCalls        []struct {
			ID       string `json:"id"`
			Function struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"messages"`
	Tools []struct {
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	} `json:"tools"`
}

func (c fixtureCase) convert() ([]core.Message, []core.Tool, Options) {
	var msgs []core.Message
	for _, m := range c.Messages {
		cm := core.Message{Role: m.Role, ReasoningContent: m.ReasoningContent, ToolCallID: m.ToolCallID, Name: m.Name}
		if m.Content != nil {
			cm.Content = *m.Content
		}
		for _, tc := range m.ToolCalls {
			cm.ToolCalls = append(cm.ToolCalls, core.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
		msgs = append(msgs, cm)
	}
	var tools []core.Tool
	for _, t := range c.Tools {
		tools = append(tools, core.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters})
	}
	opts := Options{EnableThinking: c.EnableThinking, AddGenerationPrompt: c.AddGenerationPrompt == nil || *c.AddGenerationPrompt}
	return msgs, tools, opts
}

// TestGemma4MatchesTemplate renders every case and compares it byte for byte
// with the model's own template rendered by jinja2. Regenerate the expected
// file with tools/chatfixtures/gen.py after changing the cases.
func TestGemma4MatchesTemplate(t *testing.T) {
	var cases []fixtureCase
	readJSON(t, "testdata/gemma4_cases.json", &cases)
	var want map[string]struct{ Template, Prompt string }
	readJSON(t, "testdata/gemma4_expected.json", &want)
	if len(want) != len(cases) {
		t.Fatalf("%d cases, %d expected renders: regenerate testdata", len(cases), len(want))
	}
	for _, c := range cases {
		msgs, tools, opts := c.convert()
		if got := RenderGemma4(msgs, tools, opts); got != want[c.Name].Template {
			t.Errorf("%s template:\n got %q\nwant %q", c.Name, got, want[c.Name].Template)
		}
		if got := PromptGemma4(msgs, tools, opts); got != want[c.Name].Prompt {
			t.Errorf("%s prompt:\n got %q\nwant %q", c.Name, got, want[c.Name].Prompt)
		}
	}
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatal(err)
	}
}

func TestPyFloat(t *testing.T) {
	for in, want := range map[float64]string{
		0: "0.0", 1: "1.0", 1.5: "1.5", 0.0001: "0.0001", 0.00001: "1e-05", 1e15: "1000000000000000.0",
		1e16: "1e+16", 1.2345e20: "1.2345e+20", -2.5: "-2.5", 123.456: "123.456",
	} {
		if got := pyFloat(in); got != want {
			t.Errorf("pyFloat(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestPyFloatRuntimeSum(t *testing.T) {
	a, b := 0.1, 0.2
	if got := pyFloat(a + b); got != "0.30000000000000004" {
		t.Fatalf("got %q", got)
	}
}
