// Package openai is OpenAI Chat Completions compatibility under /v1 (SPEC §6),
// so existing clients work by changing base_url and nothing else.
//
// Divergences from the original API, documented here and to callers:
//   - n > 1 returns 400.
//   - Image and audio content parts return 400 until vision lands in v2.
//   - Responses carry extra fields OpenAI clients ignore: reasoning_content on
//     messages and deltas, stats (timings, device) and warnings.
//   - enable_thinking, top_k, min_p and repeat_penalty are accepted as
//     extensions, and so is chat_template_kwargs.enable_thinking, which is
//     where llama-server and vLLM clients put it.
//
// Errors use the OpenAI error envelope rather than problem+json: OpenAI SDKs
// parse error.message and error.type strictly, and a client that cannot read
// the error is not compatible however good the successes are.
package openai

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/api/adapter"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
)

func init() { adapter.Register(config.DialectOpenAI, &Adapter{}) }

type Adapter struct{}

func (*Adapter) Mount(mux *http.ServeMux, deps adapter.Deps) error {
	h := &handler{deps: deps}
	mux.HandleFunc("POST /v1/chat/completions", h.chat)
	mux.HandleFunc("POST /v1/completions", h.complete)
	mux.HandleFunc("GET /v1/models", h.models)
	mux.HandleFunc("GET /v1/models/{id}", h.model)
	mux.HandleFunc("POST /v1/embeddings", func(w http.ResponseWriter, _ *http.Request) {
		writeError(w, httpx.NotYet("embeddings", "v2"))
	})
	return nil
}

type handler struct{ deps adapter.Deps }

func (h *handler) service(w http.ResponseWriter) (core.Service, bool) {
	if h.deps.Service == nil {
		writeError(w, core.Errorf(core.CodeModelUnavailable, "no backend can serve models in this configuration"))
		return nil, false
	}
	return h.deps.Service, true
}

// --- requests ---

type chatRequest struct {
	Model    string    `json:"model"`
	Messages []message `json:"messages"`
	Tools    []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	} `json:"tools"`
	ToolChoice          json.RawMessage `json:"tool_choice"`
	ParallelToolCalls   *bool           `json:"parallel_tool_calls"`
	Stream              bool            `json:"stream"`
	StreamOptions       *streamOptions  `json:"stream_options"`
	MaxTokens           int             `json:"max_tokens"`
	MaxCompletionTokens int             `json:"max_completion_tokens"`
	N                   *int            `json:"n"`
	Stop                stopList        `json:"stop"`
	Logprobs            bool            `json:"logprobs"`
	TopLogprobs         int             `json:"top_logprobs"`
	ResponseFormat      *struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	} `json:"response_format"`
	sampling
	EnableThinking     *bool `json:"enable_thinking"`
	ChatTemplateKwargs *struct {
		EnableThinking *bool `json:"enable_thinking"`
	} `json:"chat_template_kwargs"`
	MaxReasoningTokens int `json:"max_reasoning_tokens"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type sampling struct {
	Temperature      *float64           `json:"temperature"`
	TopP             *float64           `json:"top_p"`
	TopK             *int               `json:"top_k"`
	MinP             *float64           `json:"min_p"`
	RepeatPenalty    *float64           `json:"repeat_penalty"`
	PresencePenalty  *float64           `json:"presence_penalty"`
	FrequencyPenalty *float64           `json:"frequency_penalty"`
	LogitBias        map[string]float64 `json:"logit_bias"`
	Seed             *int64             `json:"seed"`
}

func (s sampling) core() (core.Sampling, error) {
	out := core.Sampling{
		Temperature: s.Temperature, TopP: s.TopP, TopK: s.TopK, MinP: s.MinP,
		RepeatPenalty: s.RepeatPenalty, PresencePenalty: s.PresencePenalty,
		FrequencyPenalty: s.FrequencyPenalty, Seed: s.Seed,
	}
	if len(s.LogitBias) > 0 {
		out.LogitBias = map[int]float64{}
		for k, v := range s.LogitBias {
			var id int
			if _, err := fmt.Sscan(k, &id); err != nil {
				return out, core.Errorf(core.CodeInvalidRequest, "logit_bias key %q is not a token id", k).WithParam("logit_bias")
			}
			out.LogitBias[id] = v
		}
	}
	return out, nil
}

// stopList accepts a string or an array of strings.
type stopList []string

func (s *stopList) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		return nil
	}
	var one string
	if err := json.Unmarshal(b, &one); err == nil {
		*s = []string{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(b, &many); err != nil {
		return errors.New("stop must be a string or an array of strings")
	}
	*s = many
	return nil
}

type message struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	Name             string          `json:"name"`
	ReasoningContent string          `json:"reasoning_content"`
	ToolCallID       string          `json:"tool_call_id"`
	ToolCalls        []toolCall      `json:"tool_calls"`
}

type toolCall struct {
	Index    *int   `json:"index,omitempty"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// text reads a content field: a string, null, or an array of parts of which
// only text parts are accepted in v1.
func (m message) text(i int) (string, error) {
	c := strings.TrimSpace(string(m.Content))
	if c == "" || c == "null" {
		return "", nil
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err != nil {
		return "", core.Errorf(core.CodeInvalidRequest, "messages[%d].content must be a string or an array of parts", i).WithParam("messages")
	}
	var sb strings.Builder
	for _, p := range parts {
		switch p.Type {
		case "text":
			sb.WriteString(p.Text)
		case "image_url", "input_audio", "image", "audio", "file":
			return "", core.Errorf(core.CodeInvalidRequest, "messages[%d]: %s content arrives with multimodal input in v2", i, p.Type).WithParam("messages")
		default:
			return "", core.Errorf(core.CodeInvalidRequest, "messages[%d]: unknown content part %q", i, p.Type).WithParam("messages")
		}
	}
	return sb.String(), nil
}

func decode(r *http.Request, v any) error {
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			return core.Errorf(core.CodeInvalidRequest, "the request body is larger than %d bytes", mbe.Limit)
		}
		if errors.Is(err, io.EOF) {
			return core.Errorf(core.CodeInvalidRequest, "the request body is empty")
		}
		return core.Errorf(core.CodeInvalidRequest, "the request body is not valid JSON: %v", err)
	}
	return nil
}

func (req chatRequest) core() (core.ChatRequest, error) {
	var out core.ChatRequest
	if req.N != nil && *req.N > 1 {
		return out, core.Errorf(core.CodeInvalidRequest, "n > 1 is not supported").WithParam("n")
	}
	smp, err := req.sampling.core()
	if err != nil {
		return out, err
	}
	out.Generation = core.Generation{
		Model: req.Model, Sampling: smp, MaxTokens: req.MaxTokens, Stop: req.Stop,
		Logprobs: req.Logprobs, TopLogprobs: req.TopLogprobs,
	}
	if req.MaxCompletionTokens > 0 {
		out.MaxTokens = req.MaxCompletionTokens
	}
	if rf := req.ResponseFormat; rf != nil {
		switch rf.Type {
		case "", core.FormatText:
		case core.FormatJSONObject:
			out.ResponseFormat = &core.ResponseFormat{Type: core.FormatJSONObject}
		case core.FormatJSONSchema:
			if rf.JSONSchema == nil || len(rf.JSONSchema.Schema) == 0 {
				return out, core.Errorf(core.CodeInvalidRequest, "response_format json_schema needs json_schema.schema").WithParam("response_format")
			}
			out.ResponseFormat = &core.ResponseFormat{Type: core.FormatJSONSchema, Schema: rf.JSONSchema.Schema}
		default:
			return out, core.Errorf(core.CodeInvalidRequest, "unknown response_format type %q", rf.Type).WithParam("response_format")
		}
	}
	for i, m := range req.Messages {
		text, err := m.text(i)
		if err != nil {
			return out, err
		}
		cm := core.Message{Role: m.Role, Content: text, ReasoningContent: m.ReasoningContent, ToolCallID: m.ToolCallID, Name: m.Name}
		for _, tc := range m.ToolCalls {
			cm.ToolCalls = append(cm.ToolCalls, core.ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
		}
		out.Messages = append(out.Messages, cm)
	}
	for i, t := range req.Tools {
		if t.Type != "" && t.Type != "function" {
			return out, core.Errorf(core.CodeInvalidRequest, "tools[%d]: only function tools are supported", i).WithParam("tools")
		}
		if t.Function.Name == "" {
			return out, core.Errorf(core.CodeInvalidRequest, "tools[%d] has no function name", i).WithParam("tools")
		}
		out.Tools = append(out.Tools, core.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters})
	}
	if tc := strings.TrimSpace(string(req.ToolChoice)); tc != "" && tc != "null" {
		var mode string
		if err := json.Unmarshal(req.ToolChoice, &mode); err == nil {
			switch mode {
			case core.ToolChoiceAuto, core.ToolChoiceNone, core.ToolChoiceRequired:
				out.ToolChoice = &core.ToolChoice{Mode: mode}
			default:
				return out, core.Errorf(core.CodeInvalidRequest, "tool_choice %q is not auto, none or required", mode).WithParam("tool_choice")
			}
		} else {
			var named struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			}
			if err := json.Unmarshal(req.ToolChoice, &named); err != nil || named.Function.Name == "" {
				return out, core.Errorf(core.CodeInvalidRequest, "tool_choice must be a mode or {\"type\":\"function\",\"function\":{\"name\":…}}").WithParam("tool_choice")
			}
			out.ToolChoice = &core.ToolChoice{Function: named.Function.Name}
		}
	}
	out.EnableThinking = req.EnableThinking
	if req.ChatTemplateKwargs != nil && req.ChatTemplateKwargs.EnableThinking != nil {
		out.EnableThinking = req.ChatTemplateKwargs.EnableThinking
	}
	out.MaxReasoningTokens = req.MaxReasoningTokens
	return out, nil
}

// --- responses ---

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *usage       `json:"usage,omitempty"`
	Stats   *core.Stats  `json:"stats,omitempty"`
	// Warnings is an extension; OpenAI clients ignore it.
	Warnings []core.Warning `json:"warnings,omitempty"`
}

type chatChoice struct {
	Index        int           `json:"index"`
	Message      *respMessage  `json:"message,omitempty"`
	Delta        *respMessage  `json:"delta,omitempty"`
	Logprobs     *logprobBlock `json:"logprobs"`
	FinishReason *string       `json:"finish_reason"`
}

type respMessage struct {
	Role             string     `json:"role,omitempty"`
	Content          *string    `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []toolCall `json:"tool_calls,omitempty"`
}

type logprobBlock struct {
	Content []logprobEntry `json:"content"`
}

type logprobEntry struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes"`
	TopLogprobs []topLogprob `json:"top_logprobs"`
}

type topLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

type usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func toUsage(u core.Usage) *usage {
	out := &usage{PromptTokens: u.PromptTokens, CompletionTokens: u.CompletionTokens, TotalTokens: u.PromptTokens + u.CompletionTokens}
	out.PromptTokensDetails.CachedTokens = u.CachedTokens
	out.CompletionTokensDetails.ReasoningTokens = u.ReasoningTokens
	return out
}

func byteList(s string) []int {
	out := make([]int, len(s))
	for i := range len(s) {
		out[i] = int(s[i])
	}
	return out
}

func toLogprobs(lps []core.Logprob) *logprobBlock {
	if len(lps) == 0 {
		return nil
	}
	b := &logprobBlock{Content: []logprobEntry{}}
	for _, lp := range lps {
		e := logprobEntry{Token: lp.Token, Logprob: lp.Logprob, Bytes: byteList(lp.Token), TopLogprobs: []topLogprob{}}
		for _, t := range lp.Top {
			e.TopLogprobs = append(e.TopLogprobs, topLogprob{Token: t.Token, Logprob: t.Logprob, Bytes: byteList(t.Token)})
		}
		b.Content = append(b.Content, e)
	}
	return b
}

func toToolCalls(calls []core.ToolCall, indexFrom int, stream bool) []toolCall {
	var out []toolCall
	for i, c := range calls {
		tc := toolCall{ID: c.ID, Type: "function"}
		tc.Function.Name, tc.Function.Arguments = c.Name, c.Arguments
		if stream {
			idx := indexFrom + i
			tc.Index = &idx
		}
		out = append(out, tc)
	}
	return out
}

func newID(prefix string) string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return prefix + hex.EncodeToString(b[:])
}

func ptr[T any](v T) *T { return &v }

func (h *handler) chat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := decode(r, &req); err != nil {
		writeError(w, err)
		return
	}
	creq, err := req.core()
	if err != nil {
		writeError(w, err)
		return
	}
	svc, ok := h.service(w)
	if !ok {
		return
	}
	id, created := newID("chatcmpl-"), time.Now().Unix()
	model := req.Model
	if model == "" {
		model = h.deps.DefaultModel
	}

	if !req.Stream {
		res, err := svc.Chat(r.Context(), creq, nil)
		if err != nil {
			writeError(w, err)
			return
		}
		msg := &respMessage{Role: core.RoleAssistant, ReasoningContent: res.Message.ReasoningContent,
			ToolCalls: toToolCalls(res.Message.ToolCalls, 0, false)}
		// OpenAI sends content null on a pure tool-call answer.
		if res.Message.Content != "" || len(res.Message.ToolCalls) == 0 {
			msg.Content = ptr(res.Message.Content)
		}
		httpx.WriteJSON(w, http.StatusOK, chatResponse{
			ID: id, Object: "chat.completion", Created: created, Model: model,
			Choices: []chatChoice{{Message: msg, Logprobs: toLogprobs(res.Logprobs), FinishReason: ptr(res.FinishReason)}},
			Usage:   toUsage(res.Usage), Stats: &res.Stats, Warnings: res.Warnings,
		})
		return
	}

	chunk := func(delta *respMessage, lps []core.Logprob, finish *string) chatResponse {
		return chatResponse{ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []chatChoice{{Delta: delta, Logprobs: toLogprobs(lps), FinishReason: finish}}}
	}
	calls := 0
	stream, res, err := runStreaming(w, r.Context(), func(ctx context.Context, sink core.Sink) (any, error) {
		return svc.Chat(ctx, creq, sink)
	}, func(st *httpx.Stream) error {
		// The role arrives first, on its own, as OpenAI's stream does.
		return st.Send("", "", chunk(&respMessage{Role: core.RoleAssistant, Content: ptr("")}, nil, nil))
	}, func(d core.Delta) any {
		m := &respMessage{ReasoningContent: d.ReasoningContent, ToolCalls: toToolCalls(d.ToolCalls, calls, true)}
		calls += len(d.ToolCalls)
		if d.Content != "" {
			m.Content = ptr(d.Content)
		}
		return chunk(m, d.Logprobs, nil)
	})
	if err != nil {
		return
	}
	defer stream.Close()
	cr := res.(*core.ChatResult)
	final := chunk(&respMessage{}, nil, ptr(cr.FinishReason))
	final.Stats, final.Warnings = &cr.Stats, cr.Warnings
	if err := stream.Send("", "", final); err != nil {
		return
	}
	if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
		_ = stream.Send("", "", chatResponse{ID: id, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []chatChoice{}, Usage: toUsage(cr.Usage)})
	}
	_ = stream.Data("[DONE]")
}

// runStreaming runs a generation that writes deltas to an SSE stream.
//
// The stream opens at the first delta, or when the heartbeat is due during a
// long prefill: until then nothing is written, so a request that fails early
// (unknown model, context overflow) still gets its HTTP status and an error
// body. An error after the stream has opened is sent as an OpenAI error event,
// the only way left to report it. On success the open stream is returned for
// the closing chunks; the caller closes it.
func runStreaming(w http.ResponseWriter, ctx context.Context, gen func(context.Context, core.Sink) (any, error),
	onOpen func(*httpx.Stream) error, toChunk func(core.Delta) any) (*httpx.Stream, any, error) {
	type result struct {
		v   any
		err error
	}
	deltas := make(chan core.Delta, 64)
	done := make(chan result, 1)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		v, err := gen(ctx, func(d core.Delta) error {
			select {
			case deltas <- d:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		close(deltas)
		done <- result{v, err}
	}()

	var stream *httpx.Stream
	open := func() error {
		if stream != nil {
			return nil
		}
		st, err := httpx.NewStream(w)
		if err != nil {
			return err
		}
		stream = st
		return onOpen(st)
	}
	fail := func(err error) (*httpx.Stream, any, error) {
		cancel()
		<-done
		if stream != nil {
			stream.Close()
		}
		return nil, nil, err
	}
	early := time.NewTimer(15 * time.Second)
	defer early.Stop()
	var heartbeat <-chan time.Time
	for {
		select {
		case d, ok := <-deltas:
			if !ok {
				res := <-done
				if res.err != nil {
					if stream == nil {
						writeError(w, res.err)
					} else {
						writeStreamError(stream, res.err)
						stream.Close()
					}
					return nil, nil, res.err
				}
				if err := open(); err != nil {
					return nil, nil, err
				}
				return stream, res.v, nil
			}
			if err := open(); err != nil {
				return fail(err)
			}
			heartbeat = stream.Heartbeat()
			if err := stream.Send("", "", toChunk(d)); err != nil {
				return fail(err)
			}
		case <-early.C:
			if err := open(); err != nil {
				return fail(err)
			}
			heartbeat = stream.Heartbeat()
		case <-heartbeat:
			if err := stream.Comment("keep-alive"); err != nil {
				return fail(err)
			}
		}
	}
}

func writeStreamError(stream *httpx.Stream, err error) {
	if errors.Is(err, context.Canceled) {
		return
	}
	_ = stream.Send("", "", envelope(err))
	_ = stream.Data("[DONE]")
}

// --- completions ---

type completionRequest struct {
	Model       string          `json:"model"`
	Prompt      json.RawMessage `json:"prompt"`
	MaxTokens   *int            `json:"max_tokens"`
	Stream      bool            `json:"stream"`
	Stop        stopList        `json:"stop"`
	N           *int            `json:"n"`
	Logprobs    *int            `json:"logprobs"`
	Echo        bool            `json:"echo"`
	Suffix      string          `json:"suffix"`
	BestOf      *int            `json:"best_of"`
	StreamOpts  *streamOptions  `json:"stream_options"`
	CachePrompt *bool           `json:"cache_prompt"`
	sampling
}

type completionResponse struct {
	ID       string             `json:"id"`
	Object   string             `json:"object"`
	Created  int64              `json:"created"`
	Model    string             `json:"model"`
	Choices  []completionChoice `json:"choices"`
	Usage    *usage             `json:"usage,omitempty"`
	Stats    *core.Stats        `json:"stats,omitempty"`
	Warnings []core.Warning     `json:"warnings,omitempty"`
}

type completionChoice struct {
	Text         string          `json:"text"`
	Index        int             `json:"index"`
	Logprobs     *legacyLogprobs `json:"logprobs"`
	FinishReason *string         `json:"finish_reason"`
}

type legacyLogprobs struct {
	Tokens        []string             `json:"tokens"`
	TokenLogprobs []float64            `json:"token_logprobs"`
	TopLogprobs   []map[string]float64 `json:"top_logprobs"`
	TextOffset    []int                `json:"text_offset"`
}

func toLegacy(lps []core.Logprob, offset int) *legacyLogprobs {
	if len(lps) == 0 {
		return nil
	}
	out := &legacyLogprobs{}
	for _, lp := range lps {
		out.Tokens = append(out.Tokens, lp.Token)
		out.TokenLogprobs = append(out.TokenLogprobs, lp.Logprob)
		top := map[string]float64{}
		for _, t := range lp.Top {
			top[t.Token] = t.Logprob
		}
		out.TopLogprobs = append(out.TopLogprobs, top)
		out.TextOffset = append(out.TextOffset, offset)
		offset += len(lp.Token)
	}
	return out
}

func (h *handler) complete(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := decode(r, &req); err != nil {
		writeError(w, err)
		return
	}
	var prompt string
	if err := json.Unmarshal(req.Prompt, &prompt); err != nil {
		var many []string
		if json.Unmarshal(req.Prompt, &many) != nil || len(many) != 1 {
			writeError(w, core.Errorf(core.CodeInvalidRequest, "prompt must be a string (or an array of one)").WithParam("prompt"))
			return
		}
		prompt = many[0]
	}
	switch {
	case req.N != nil && *req.N > 1, req.BestOf != nil && *req.BestOf > 1:
		writeError(w, core.Errorf(core.CodeInvalidRequest, "n and best_of > 1 are not supported").WithParam("n"))
		return
	case req.Echo:
		writeError(w, core.Errorf(core.CodeInvalidRequest, "echo is not supported").WithParam("echo"))
		return
	case req.Suffix != "":
		writeError(w, core.Errorf(core.CodeInvalidRequest, "suffix is not supported").WithParam("suffix"))
		return
	}
	smp, err := req.core()
	if err != nil {
		writeError(w, err)
		return
	}
	// The legacy endpoint's documented default is 16 tokens.
	maxTokens := 16
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	creq := core.CompletionRequest{Prompt: prompt, Generation: core.Generation{
		Model: req.Model, Sampling: smp, MaxTokens: maxTokens, Stop: req.Stop, CachePrompt: req.CachePrompt,
	}}
	if req.Logprobs != nil {
		creq.Logprobs, creq.TopLogprobs = true, *req.Logprobs
	}
	svc, ok := h.service(w)
	if !ok {
		return
	}
	id, created := newID("cmpl-"), time.Now().Unix()
	model := req.Model
	if model == "" {
		model = h.deps.DefaultModel
	}
	if !req.Stream {
		res, err := svc.Complete(r.Context(), creq, nil)
		if err != nil {
			writeError(w, err)
			return
		}
		httpx.WriteJSON(w, http.StatusOK, completionResponse{
			ID: id, Object: "text_completion", Created: created, Model: model,
			Choices: []completionChoice{{Text: res.Text, Logprobs: toLegacy(res.Logprobs, 0), FinishReason: ptr(res.FinishReason)}},
			Usage:   toUsage(res.Usage), Stats: &res.Stats, Warnings: res.Warnings,
		})
		return
	}
	offset := 0
	stream, res, err := runStreaming(w, r.Context(), func(ctx context.Context, sink core.Sink) (any, error) {
		return svc.Complete(ctx, creq, sink)
	}, func(*httpx.Stream) error { return nil }, func(d core.Delta) any {
		c := completionResponse{ID: id, Object: "text_completion", Created: created, Model: model,
			Choices: []completionChoice{{Text: d.Content, Logprobs: toLegacy(d.Logprobs, offset)}}}
		offset += len(d.Content)
		return c
	})
	if err != nil {
		return
	}
	defer stream.Close()
	cr := res.(*core.CompletionResult)
	final := completionResponse{ID: id, Object: "text_completion", Created: created, Model: model,
		Choices: []completionChoice{{FinishReason: ptr(cr.FinishReason)}}, Stats: &cr.Stats, Warnings: cr.Warnings}
	if req.StreamOpts != nil && req.StreamOpts.IncludeUsage {
		final.Usage = toUsage(cr.Usage)
	}
	if err := stream.Send("", "", final); err != nil {
		return
	}
	_ = stream.Data("[DONE]")
}

// --- models ---

type modelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func (h *handler) models(w http.ResponseWriter, _ *http.Request) {
	data := []modelEntry{}
	for _, m := range h.deps.Catalog {
		data = append(data, modelEntry{ID: m.ID, Object: "model", OwnedBy: "nanollm"})
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (h *handler) model(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	for _, m := range h.deps.Catalog {
		if m.ID == id {
			httpx.WriteJSON(w, http.StatusOK, modelEntry{ID: m.ID, Object: "model", OwnedBy: "nanollm"})
			return
		}
	}
	writeError(w, core.Errorf(core.CodeModelNotFound, "no model %q in the catalog", id).WithParam("model"))
}

// --- errors ---

// errorEnvelope is the OpenAI error shape. Clients parse it strictly, so the
// field names are part of the contract.
type errorEnvelope struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param,omitempty"`
		Code    string `json:"code"`
	} `json:"error"`
}

func envelope(err error) errorEnvelope {
	e := core.AsError(err)
	var env errorEnvelope
	env.Error.Message = e.Message
	env.Error.Type = errorType(e.Code)
	env.Error.Param = e.Param
	env.Error.Code = string(e.Code)
	return env
}

func writeError(w http.ResponseWriter, err error) {
	e := core.AsError(err)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.Code.HTTPStatus())
	_ = json.NewEncoder(w).Encode(envelope(err))
}

// errorType maps our codes onto the small vocabulary OpenAI clients branch on.
func errorType(c core.Code) string {
	switch c {
	case core.CodeInvalidRequest, core.CodeContextOverflow, core.CodeModelNotFound,
		core.CodeDeviceUnavailable, core.CodeCapabilityUnavailable:
		return "invalid_request_error"
	case core.CodeUnauthorized:
		return "authentication_error"
	case core.CodeForbidden:
		return "permission_error"
	case core.CodeQueueFull, core.CodeRateLimited:
		return "rate_limit_error"
	default:
		return "server_error"
	}
}
