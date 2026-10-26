package core

import "encoding/json"

// Message roles.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Message is one chat turn. Content is text only in v1; image and audio parts
// are refused by the dialects until multimodal input lands in v2.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// ReasoningContent is the thinking text of an earlier assistant turn. The
	// chat template decides whether it is shown to the model again.
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID ties a tool result to the call that asked for it.
	ToolCallID string `json:"tool_call_id,omitempty"`
	Name       string `json:"name,omitempty"`
}

// Tool is a function the model may call.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"` // JSON Schema
}

// ToolCall is a call the model made, parsed out of its output by the family's
// tool-call parser.
type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON text, as the model wrote it
}

// Tool choice modes. A named function is given in ToolChoice.Function.
const (
	ToolChoiceAuto     = "auto"
	ToolChoiceNone     = "none"
	ToolChoiceRequired = "required"
)

type ToolChoice struct {
	Mode     string `json:"mode,omitempty"`
	Function string `json:"function,omitempty"`
}

// Sampling holds the sampler parameters. A nil field means "the model
// manifest's recommended value", which is not the same as zero: temperature 0
// is greedy decoding, and a client that sends nothing should get the model
// card's settings rather than that.
type Sampling struct {
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	TopK             *int            `json:"top_k,omitempty"`
	MinP             *float64        `json:"min_p,omitempty"`
	RepeatPenalty    *float64        `json:"repeat_penalty,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	LogitBias        map[int]float64 `json:"logit_bias,omitempty"`
	Seed             *int64          `json:"seed,omitempty"`
}

// Response format types.
const (
	FormatText       = "text"
	FormatJSONObject = "json_object"
	FormatJSONSchema = "json_schema"
)

// ResponseFormat constrains output. JSON formats are compiled to a GBNF
// grammar and enforced by the sampler, so output is valid by construction.
type ResponseFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema,omitempty"`
}

// Generation is what chat and raw completion share.
type Generation struct {
	Model    string   `json:"model,omitempty"`
	Sampling Sampling `json:"sampling"`
	// MaxTokens is capped by generation.max_tokens_cap. Zero means the cap.
	MaxTokens int      `json:"max_tokens,omitempty"`
	Stop      []string `json:"stop,omitempty"`
	// Logprobs asks for the chosen token's log-probability; TopLogprobs (≤ 20)
	// for that many alternatives as well.
	Logprobs    bool `json:"logprobs,omitempty"`
	TopLogprobs int  `json:"top_logprobs,omitempty"`
	// Grammar is raw GBNF; native dialect only.
	Grammar        string          `json:"grammar,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// SessionID pins a conversation to one slot so its state is reused.
	SessionID string `json:"session_id,omitempty"`
	// CachePrompt=false opts out of prefix reuse for this request.
	CachePrompt *bool `json:"cache_prompt,omitempty"`
	// Device is a selector like devices.default. Fallback overrides
	// devices.fallback for this request.
	Device   string `json:"device,omitempty"`
	Fallback *bool  `json:"fallback,omitempty"`
}

type ChatRequest struct {
	Generation
	Messages   []Message   `json:"messages"`
	Tools      []Tool      `json:"tools,omitempty"`
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`
	// EnableThinking maps to the family's chat-template switch; nil means
	// generation.enable_thinking.
	EnableThinking *bool `json:"enable_thinking,omitempty"`
	// MaxReasoningTokens closes the thinking block when reached, and the
	// answer follows.
	MaxReasoningTokens int `json:"max_reasoning_tokens,omitempty"`

	// ToolFormat says how tool calls are read. "strict", the default, constrains the
	// output with the tool grammar when tools are offered. "rescue" applies no
	// grammar: the calls are read from the text, and what cannot be read is reported
	// in the parse report (RFC R4).
	ToolFormat string `json:"tool_format,omitempty"`
	// AssistantPrefix is text that the model continues in its own turn. The reply
	// holds only what the model adds after it (RFC R5).
	AssistantPrefix string `json:"assistant_prefix,omitempty"`
}

type CompletionRequest struct {
	Generation
	Prompt string `json:"prompt"`
}

// Finish reasons.
const (
	FinishStop      = "stop"
	FinishLength    = "length"
	FinishToolCalls = "tool_calls"
)

type ChatResult struct {
	Message      Message   `json:"message"`
	FinishReason string    `json:"finish_reason"`
	Logprobs     []Logprob `json:"logprobs,omitempty"`
	Usage        Usage     `json:"usage"`
	Stats        Stats     `json:"stats"`
	Warnings     []Warning `json:"warnings,omitempty"`
	// Parse says how the reply was read from the model's text. It is nil for a
	// request that the parser does not read.
	Parse *ParseReport `json:"parse,omitempty"`
}

type CompletionResult struct {
	Text         string    `json:"text"`
	FinishReason string    `json:"finish_reason"`
	Logprobs     []Logprob `json:"logprobs,omitempty"`
	Usage        Usage     `json:"usage"`
	Stats        Stats     `json:"stats"`
	Warnings     []Warning `json:"warnings,omitempty"`
}

type Logprob struct {
	Token   string     `json:"token"`
	Logprob float64    `json:"logprob"`
	Top     []TopToken `json:"top_logprobs,omitempty"`
}

type TopToken struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CachedTokens     int `json:"cached_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens"`
}

// Warning is a non-fatal note about how a request was served, such as a
// device fallback.
type Warning struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Warning codes.
const (
	WarnDeviceFallback = "device_fallback"
)

type TokenizeRequest struct {
	Model string `json:"model,omitempty"`
	// Exactly one of Text and Messages. Messages are rendered through the chat
	// template first, which is what makes the count useful.
	Text     string    `json:"text,omitempty"`
	Messages []Message `json:"messages,omitempty"`
	Tools    []Tool    `json:"tools,omitempty"`
}

type TokenizeResult struct {
	Tokens []int `json:"tokens"`
	Count  int   `json:"count"`
}

type DetokenizeRequest struct {
	Model  string `json:"model,omitempty"`
	Tokens []int  `json:"tokens"`
}

type DetokenizeResult struct {
	Text string `json:"text"`
}

// Tool formats (ChatRequest.ToolFormat).
const (
	ToolFormatStrict = "strict"
	ToolFormatRescue = "rescue"
)

// Parse statuses (ParseReport.Status).
const (
	ParseOK       = "ok"       // every call and channel was read as written
	ParseRepaired = "repaired" // the parser dropped words that followed the calls
	ParseFailed   = "failed"   // a call could not be read; Raw holds the whole reply text
)

// ParseReport says how the reply was read from the model's text (RFC R4).
type ParseReport struct {
	Status string `json:"status"`
	Raw    string `json:"raw,omitempty"`
	Error  string `json:"error,omitempty"`
}
