package chat

import (
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/core"
)

func TestAssistantPrefixFollowsTheTurnOpener(t *testing.T) {
	msgs := []core.Message{{Role: core.RoleUser, Content: "Name the report."}}
	got := PromptGemma4(msgs, nil, Options{AddGenerationPrompt: true, AssistantPrefix: "# Report"})
	if !strings.HasSuffix(got, turnOpen+"model\n# Report") {
		t.Fatalf("the prompt ends %q, want the model's turn opened with the prefix", got)
	}
}

func TestWithoutAGenerationPromptThePrefixIsNotAdded(t *testing.T) {
	msgs := []core.Message{{Role: core.RoleUser, Content: "Hi."}}
	if got := PromptGemma4(msgs, nil, Options{AssistantPrefix: "# Report"}); strings.Contains(got, "# Report") {
		t.Fatalf("the prefix was added without a generation prompt: %q", got)
	}
}

func TestParserReportsWordsDroppedAfterCalls(t *testing.T) {
	p := NewParser(ParseOptions{Tools: true})
	if _, err := p.Feed("<|tool_call>call:get_time{}<tool_call|>and some words"); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if !p.Repaired() {
		t.Fatal("words dropped after the calls were not reported as a repair")
	}
}

func TestParserWithOnlyCallsIsNotRepaired(t *testing.T) {
	p := NewParser(ParseOptions{Tools: true})
	if _, err := p.Feed("<|tool_call>call:get_time{}<tool_call|>\n"); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if p.Repaired() {
		t.Fatal("a reply with only calls was reported as repaired")
	}
}
