package core

// Stage is one step of the fixed request pipeline (SPEC §2). Every stage is
// timed, and the timings are returned with every response.
type Stage string

const (
	// StageTemplate renders messages, tools and the thinking flag through the
	// model's chat template.
	StageTemplate Stage = "template"
	// StageTokenize turns text into token ids via the backend tokenizer.
	StageTokenize Stage = "tokenize"
	// StageCache finds the longest reusable prefix in a slot, restoring a
	// saved state snapshot if the model needs one.
	StageCache Stage = "cache"
	// StagePrefill processes the uncached prompt tokens in batches.
	StagePrefill Stage = "prefill"
	// StageDecode is the token loop: sample, constrain, stream.
	StageDecode Stage = "decode"
	// StagePost splits reasoning from the answer, parses tool calls, applies
	// stop sequences and assembles usage.
	StagePost Stage = "post"
)

// Stages is the pipeline in execution order.
var Stages = []Stage{StageTemplate, StageTokenize, StageCache, StagePrefill, StageDecode, StagePost}

// Stats is the `stats` object every response carries (SPEC §12).
type Stats struct {
	StagesMS     map[Stage]float64 `json:"stages_ms"`
	TTFTMS       float64           `json:"ttft_ms"`
	PrefillTokS  float64           `json:"prefill_tok_s"`
	DecodeTokS   float64           `json:"decode_tok_s"`
	PromptTokens int               `json:"prompt_tokens"`
	CachedTokens int               `json:"cached_tokens"`
	// CompletionTokens includes ReasoningTokens.
	CompletionTokens int    `json:"completion_tokens"`
	ReasoningTokens  int    `json:"reasoning_tokens"`
	Backend          string `json:"backend"`
	Device           string `json:"device"`
	Preset           string `json:"preset,omitempty"`
	Mode             string `json:"mode,omitempty"`
	// DraftAcceptance is the share of drafted tokens the target accepted;
	// absent when no drafter ran.
	DraftAcceptance *float64 `json:"draft_acceptance,omitempty"`
	Slot            int      `json:"slot"`
}
