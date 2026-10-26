package core

import "testing"

func TestGemma4HasTheHarnessFeatures(t *testing.T) {
	c := CapabilitiesOf(ModelInfo{ID: "gemma-4-e4b-qat", Family: "gemma4"})
	if !c.Thinking || !c.ParseReport || !c.ReasoningBudget || !c.AssistantPrefix || !c.PrefixCache || !c.Tokenize {
		t.Fatalf("capabilities %+v", c)
	}
	if len(c.ToolFormats) != 2 {
		t.Fatalf("tool formats %v, want strict and rescue", c.ToolFormats)
	}
}

func TestAnotherFamilyHasOnlyTheEngineFeatures(t *testing.T) {
	c := CapabilitiesOf(ModelInfo{ID: "other", Family: "other"})
	if c.Thinking || c.ParseReport || c.AssistantPrefix || len(c.ToolFormats) != 0 || !c.Tokenize {
		t.Fatalf("capabilities %+v", c)
	}
}
