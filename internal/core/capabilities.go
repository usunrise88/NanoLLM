package core

// Capabilities is what a catalog model can do through this API (RFC R1). The features
// follow the model's family, so a client can probe once and cache the answer.
type Capabilities struct {
	ID              string   `json:"id"`
	Family          string   `json:"family,omitempty"`
	Thinking        bool     `json:"thinking"`
	ToolFormats     []string `json:"tool_formats"`
	ReasoningBudget bool     `json:"reasoning_budget"`
	ParseReport     bool     `json:"parse_report"`
	AssistantPrefix bool     `json:"assistant_prefix"`
	PrefixCache     bool     `json:"prefix_cache"`
	Tokenize        bool     `json:"tokenize"`
}

// CapabilitiesOf reports the features of a catalog entry. Every family has tokenization
// and the prompt cache of cache_prompt; Gemma 4 also has the thinking switch, both tool
// formats, the reasoning budget, the parse report and the assistant prefix.
func CapabilitiesOf(m ModelInfo) Capabilities {
	c := Capabilities{ID: m.ID, Family: m.Family, ToolFormats: []string{}, PrefixCache: true, Tokenize: true}
	if m.Family == "gemma4" {
		c.Thinking = true
		c.ToolFormats = []string{ToolFormatStrict, ToolFormatRescue}
		c.ReasoningBudget = true
		c.ParseReport = true
		c.AssistantPrefix = true
	}
	return c
}
