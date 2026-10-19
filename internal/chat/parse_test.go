package chat

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/core"
)

type parsed struct {
	content, reasoning string
	calls              []core.ToolCall
}

func collect(t *testing.T, opts ParseOptions, chunks []string) (parsed, error) {
	t.Helper()
	p := NewParser(opts)
	var out parsed
	add := func(ds []core.Delta) {
		for _, d := range ds {
			if d.Content != "" && d.ReasoningContent != "" {
				t.Fatalf("a delta mixes content and reasoning: %+v", d)
			}
			out.content += d.Content
			out.reasoning += d.ReasoningContent
			out.calls = append(out.calls, d.ToolCalls...)
		}
	}
	for _, c := range chunks {
		ds, err := p.Feed(c)
		add(ds)
		if err != nil {
			return out, err
		}
	}
	ds, err := p.Close()
	add(ds)
	return out, err
}

// every runs the parser on the whole text and on every two-way split of it,
// and requires the same answer: streaming must not change what is parsed.
func every(t *testing.T, opts ParseOptions, text string) parsed {
	t.Helper()
	want, err := collect(t, opts, []string{text})
	if err != nil {
		t.Fatalf("%q: %v", text, err)
	}
	for i := 1; i < len(text); i++ {
		got, err := collect(t, opts, []string{text[:i], text[i:]})
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("split at %d of %q: %+v (%v), want %+v", i, text, got, err, want)
		}
	}
	return want
}

func TestParsePlain(t *testing.T) {
	for in, want := range map[string]parsed{
		"Hello <b>world</b>": {content: "Hello <b>world</b>"},
		"<|turn>model\nHi":   {content: "Hi"},
		"<|channel>thought\n  Let me think.<channel|>Answer.": {content: "Answer.", reasoning: "Let me think."},
		"<|channel><|channel>thought x<channel|>A":            {content: "A", reasoning: "x"},
		"Answer.<channel|>":                {content: "Answer."},
		"A<|channel>thought\n<channel|>":   {content: "A"},
		"<|tool_call>call:f{}<tool_call|>": {content: "<|tool_call>call:f{}<tool_call|>"},
	} {
		if got := every(t, ParseOptions{}, in); !reflect.DeepEqual(got, want) {
			t.Errorf("%q: got %+v want %+v", in, got, want)
		}
	}
}

func TestParseToolCalls(t *testing.T) {
	in := `<|channel>thought
need weather<channel|>Checking.<|tool_call>call:get_weather{city:<|"|>Paris, "FR"<|"|>,days:3,opts:{metric:true,tags:[<|"|>a<|"|>, 1.5e3, null]}}<tool_call|>junk<|tool_call>call:noop{}<tool_call|>`
	got := every(t, ParseOptions{Tools: true}, in)
	if got.reasoning != "need weather" || got.content != "Checking." || len(got.calls) != 2 {
		t.Fatalf("%+v", got)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(got.calls[0].Arguments), &args); err != nil {
		t.Fatalf("arguments are not JSON: %s", got.calls[0].Arguments)
	}
	want := `{"city":"Paris, \"FR\"","days":3,"opts":{"metric":true,"tags":["a",1.5e3,null]}}`
	if got.calls[0].Name != "get_weather" || got.calls[0].Arguments != want {
		t.Fatalf("call %+v", got.calls[0])
	}
	if got.calls[1].Name != "noop" || got.calls[1].Arguments != "{}" {
		t.Fatalf("call %+v", got.calls[1])
	}
}

func TestParseRoundTripsTemplateSyntax(t *testing.T) {
	// What the template renders for a call, the parser reads back.
	args := `{"b": [1, "x", {"Z": null}], "a": "q", "n": -2.5}`
	rendered := callOpen + "call:f{" + callArguments(args) + "}" + callClose
	got := every(t, ParseOptions{Tools: true}, rendered)
	var a, b any
	json.Unmarshal([]byte(args), &a)
	json.Unmarshal([]byte(got.calls[0].Arguments), &b)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("%s → %s", args, got.calls[0].Arguments)
	}
}

func TestParseErrors(t *testing.T) {
	for _, in := range []string{
		"<|tool_call>call:f{a:1",
		"<|tool_call>call:f{a:}<tool_call|>",
		"<|tool_call>f{}<tool_call|>",
		"<|tool_call>call:f{a:<|\"|>x}<tool_call|>",
	} {
		if _, err := collect(t, ParseOptions{Tools: true}, []string{in}); err == nil {
			t.Errorf("%q: expected an error", in)
		}
	}
}

func TestParseJSONFence(t *testing.T) {
	for _, in := range []string{
		"```json\n{\"a\": [1, 2]}\n```",
		"{\"a\": [1, 2]}",
		"<|channel>thought\nok<channel|>```json {\"a\": [1, 2]} ```\n",
	} {
		got := every(t, ParseOptions{JSON: true}, in)
		if strings.TrimSpace(got.content) != `{"a": [1, 2]}` || strings.Contains(got.content, "`") {
			t.Errorf("%q: content %q", in, got.content)
		}
	}
}
