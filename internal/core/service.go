// Package core is the only entry point into NanoLLM business logic (SPEC §2).
//
// API dialects depend on Service and on nothing below it: they translate
// request and response shapes and must never reach into the model pool, the
// scheduler, the registry or the queue directly. The CLI's `run` verb talks to
// the same interface, so what the terminal shows is what the API returns.
package core

import "context"

// Service generates text.
//
// Streaming and non-streaming calls are the same call: a nil sink collects the
// whole answer, a non-nil one additionally receives every piece as it is
// decoded. One code path means a streamed answer and a buffered one cannot
// drift apart in their stop handling, reasoning split or usage accounting.
type Service interface {
	// Chat renders the messages through the model's chat template and
	// generates the assistant turn.
	Chat(ctx context.Context, req ChatRequest, sink Sink) (*ChatResult, error)

	// Complete generates a continuation of a raw prompt, without a template.
	Complete(ctx context.Context, req CompletionRequest, sink Sink) (*CompletionResult, error)

	// Tokenize counts and returns tokens for text, or for a chat payload with
	// the template applied.
	Tokenize(ctx context.Context, req TokenizeRequest) (*TokenizeResult, error)

	Detokenize(ctx context.Context, req DetokenizeRequest) (*DetokenizeResult, error)
}

// Sink receives each piece of a streamed answer. Returning an error stops
// generation — that is how a disconnected client frees its slot — and the
// error is returned from the call that was streaming.
type Sink func(Delta) error

// Delta is one streamed piece. Reasoning and content never share a delta: the
// post stage has already decided which side of the thinking block it is on.
type Delta struct {
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	Logprobs         []Logprob  `json:"logprobs,omitempty"`
}
