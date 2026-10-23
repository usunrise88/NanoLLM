package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/registry"
)

// --- requests ---

// common is what /api/chat and /api/generate share.
type common struct {
	Model     string          `json:"model"`
	Format    json.RawMessage `json:"format"`
	Options   json.RawMessage `json:"options"`
	Stream    *bool           `json:"stream"`
	Think     json.RawMessage `json:"think"`
	KeepAlive json.RawMessage `json:"keep_alive"`
	Logprobs  bool            `json:"logprobs"`
}

type chatRequest struct {
	common
	Messages []chatMessage `json:"messages"`
	Tools    []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	} `json:"tools"`
}

type chatMessage struct {
	Role       string            `json:"role"`
	Content    string            `json:"content"`
	Thinking   string            `json:"thinking"`
	Images     []json.RawMessage `json:"images"`
	ToolCalls  []requestToolCall `json:"tool_calls"`
	ToolName   string            `json:"tool_name"`
	ToolCallID string            `json:"tool_call_id"`
}

type requestToolCall struct {
	ID       string `json:"id"`
	Function struct {
		Name string `json:"name"`
		// Arguments is an object, where OpenAI's is a string of JSON.
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type generateRequest struct {
	common
	Prompt   string            `json:"prompt"`
	Suffix   string            `json:"suffix"`
	System   string            `json:"system"`
	Template string            `json:"template"`
	Context  json.RawMessage   `json:"context"`
	Raw      bool              `json:"raw"`
	Images   []json.RawMessage `json:"images"`
}

const noImages = "this model does not accept images: image input arrives with multimodal support in v2"

// base is the generation a request describes, without its messages, and what
// the request asked of keep_alive.
func (c common) base(m registry.Manifest) (core.ChatRequest, keepAlive, []core.Warning, error) {
	var req core.ChatRequest
	var ka keepAlive
	if c.Logprobs {
		return req, ka, nil, core.Errorf(core.CodeInvalidRequest, "logprobs are not supported by the ollama API of this server; /v1/chat/completions has them").WithParam("logprobs")
	}
	opts, err := parseOptions(c.Options)
	if err != nil {
		return req, ka, nil, err
	}
	format, err := parseFormat(c.Format)
	if err != nil {
		return req, ka, nil, err
	}
	think, err := parseThink(c.Think)
	if err != nil {
		return req, ka, nil, err
	}
	if ka, err = parseKeepAlive(c.KeepAlive); err != nil {
		return req, ka, nil, err
	}
	req.Generation = core.Generation{
		Model: m.ID, Sampling: opts.Sampling, MaxTokens: opts.NumPredict, Stop: opts.Stop, ResponseFormat: format,
	}
	req.EnableThinking = think
	return req, ka, opts.warning(), nil
}

func (c common) streaming() bool { return c.Stream == nil || *c.Stream }

// messages converts the conversation. Ollama gives a tool call's arguments as an
// object and does not number its calls, so the calls get ids here and a tool
// result is tied to the call it answers: by id when the client has one, else by
// tool_name, else to the first call still waiting for a result.
func messages(in []chatMessage) ([]core.Message, error) {
	out := make([]core.Message, 0, len(in))
	var waiting []core.ToolCall
	for i, m := range in {
		if len(m.Images) > 0 {
			return nil, core.Errorf(core.CodeInvalidRequest, "messages[%d]: %s", i, noImages).WithParam("images")
		}
		cm := core.Message{Role: m.Role, Content: m.Content, ReasoningContent: m.Thinking}
		switch m.Role {
		case core.RoleAssistant:
			waiting = waiting[:0]
			for j, tc := range m.ToolCalls {
				args, err := argumentsText(tc.Function.Arguments)
				if err != nil {
					return nil, core.Errorf(core.CodeInvalidRequest, "messages[%d].tool_calls[%d]: %v", i, j, err).WithParam("messages")
				}
				id := tc.ID
				if id == "" {
					id = fmt.Sprintf("call_%d_%d", i, j)
				}
				cm.ToolCalls = append(cm.ToolCalls, core.ToolCall{ID: id, Name: tc.Function.Name, Arguments: args})
			}
			waiting = append(waiting, cm.ToolCalls...)
		case core.RoleTool:
			cm.Name, cm.ToolCallID = m.ToolName, m.ToolCallID
			for k, c := range waiting {
				var answers bool
				switch {
				case cm.ToolCallID != "":
					answers = c.ID == cm.ToolCallID
				case cm.Name != "":
					answers = c.Name == cm.Name
				default:
					answers = true
				}
				if !answers {
					continue
				}
				cm.ToolCallID = c.ID
				if cm.Name == "" {
					cm.Name = c.Name
				}
				waiting = append(waiting[:k], waiting[k+1:]...)
				break
			}
		}
		out = append(out, cm)
	}
	return out, nil
}

// argumentsText is a call's arguments as the JSON text core carries. A string
// that holds an object is accepted as well: some clients send OpenAI's form.
func argumentsText(raw json.RawMessage) (string, error) {
	if isNull(raw) {
		return "{}", nil
	}
	if t := bytes.TrimSpace(raw); t[0] == '"' {
		var s string
		if err := json.Unmarshal(t, &s); err != nil {
			return "", errors.New("arguments is not valid JSON")
		}
		raw = json.RawMessage(s)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return "", errors.New("arguments must be a JSON object")
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return "", errors.New("arguments is not valid JSON")
	}
	return buf.String(), nil
}

// --- responses ---

type respToolCall struct {
	ID       string `json:"id,omitempty"`
	Function struct {
		Index     int             `json:"index"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

func toolCalls(calls []core.ToolCall, from int) []respToolCall {
	var out []respToolCall
	for i, c := range calls {
		tc := respToolCall{ID: c.ID}
		tc.Function.Index, tc.Function.Name = from+i, c.Name
		tc.Function.Arguments = json.RawMessage("{}")
		if json.Valid([]byte(c.Arguments)) {
			tc.Function.Arguments = json.RawMessage(c.Arguments)
		}
		out = append(out, tc)
	}
	return out
}

// timings are the durations (nanoseconds) and counts the last line carries.
type timings struct {
	TotalDuration      int64 `json:"total_duration"`
	LoadDuration       int64 `json:"load_duration"`
	PromptEvalCount    int   `json:"prompt_eval_count"`
	PromptEvalDuration int64 `json:"prompt_eval_duration"`
	EvalCount          int   `json:"eval_count"`
	EvalDuration       int64 `json:"eval_duration"`
}

func nanos(ms float64) int64 { return int64(ms * float64(time.Millisecond)) }

func timingsOf(start time.Time, res *core.ChatResult) *timings {
	return &timings{
		TotalDuration:      time.Since(start).Nanoseconds(),
		PromptEvalCount:    res.Usage.PromptTokens,
		PromptEvalDuration: nanos(res.Stats.StagesMS[core.StagePrefill]),
		EvalCount:          res.Usage.CompletionTokens,
		EvalDuration:       nanos(res.Stats.StagesMS[core.StageDecode]),
	}
}

// fragment is a piece of the answer, or all of it.
type fragment struct {
	content, thinking string
	calls             []respToolCall
}

func (f fragment) empty() bool { return f.content == "" && f.thinking == "" && len(f.calls) == 0 }

// final is what closes an answer.
type final struct {
	reason   string
	timings  *timings
	warnings []core.Warning
}

// doneReason is core's finish reason in Ollama's words: a turn that ends in a
// tool call is "stop" there.
func doneReason(finish string) string {
	if finish == core.FinishLength {
		return "length"
	}
	return "stop"
}

// chatResponse is one line of /api/chat: a piece of the answer, or the last line.
type chatResponse struct {
	Model      string    `json:"model"`
	CreatedAt  time.Time `json:"created_at"`
	Message    chatReply `json:"message"`
	Done       bool      `json:"done"`
	DoneReason string    `json:"done_reason,omitempty"`
	*timings
	// Warnings is an extension; clients ignore it.
	Warnings []core.Warning `json:"warnings,omitempty"`
}

type chatReply struct {
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	Thinking  string         `json:"thinking,omitempty"`
	ToolCalls []respToolCall `json:"tool_calls,omitempty"`
}

// generateResponse is one line of /api/generate.
type generateResponse struct {
	Model      string    `json:"model"`
	CreatedAt  time.Time `json:"created_at"`
	Response   string    `json:"response"`
	Thinking   string    `json:"thinking,omitempty"`
	Done       bool      `json:"done"`
	DoneReason string    `json:"done_reason,omitempty"`
	*timings
	Warnings []core.Warning `json:"warnings,omitempty"`
}

// --- /api/chat and /api/generate ---

// runSpec is one generation, whichever endpoint asked for it.
type runSpec struct {
	req      core.ChatRequest
	stream   bool
	keep     keepAlive
	warnings []core.Warning
	// render builds a line of the endpoint's response; fin is nil for a
	// piece in the middle of the answer.
	render func(f fragment, fin *final) any
}

func (h *handler) chat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := decode(r, &req); err != nil {
		writeError(w, err)
		return
	}
	m, err := h.resolve(req.Model)
	if err != nil {
		writeError(w, err)
		return
	}
	creq, keep, warns, err := req.base(m)
	if err != nil {
		writeError(w, err)
		return
	}
	name := req.Model
	spec := &runSpec{req: creq, stream: req.streaming(), keep: keep, warnings: warns}
	spec.render = func(f fragment, fin *final) any {
		resp := chatResponse{Model: name, CreatedAt: time.Now().UTC(),
			Message: chatReply{Role: core.RoleAssistant, Content: f.content, Thinking: f.thinking, ToolCalls: f.calls}}
		if fin != nil {
			resp.Done, resp.DoneReason, resp.timings, resp.Warnings = true, fin.reason, fin.timings, fin.warnings
		}
		return resp
	}
	if len(req.Messages) == 0 {
		h.preload(w, r, m, spec)
		return
	}
	if spec.req.Messages, err = messages(req.Messages); err != nil {
		writeError(w, err)
		return
	}
	for i, t := range req.Tools {
		if t.Type != "" && t.Type != "function" {
			writeError(w, core.Errorf(core.CodeInvalidRequest, "tools[%d]: only function tools are supported", i).WithParam("tools"))
			return
		}
		if t.Function.Name == "" {
			writeError(w, core.Errorf(core.CodeInvalidRequest, "tools[%d] has no function name", i).WithParam("tools"))
			return
		}
		spec.req.Tools = append(spec.req.Tools, core.Tool{Name: t.Function.Name, Description: t.Function.Description, Parameters: t.Function.Parameters})
	}
	h.serve(w, r, m, spec)
}

func (h *handler) generate(w http.ResponseWriter, r *http.Request) {
	var req generateRequest
	if err := decode(r, &req); err != nil {
		writeError(w, err)
		return
	}
	m, err := h.resolve(req.Model)
	if err != nil {
		writeError(w, err)
		return
	}
	var refuse string
	switch {
	case len(req.Images) > 0:
		refuse = noImages
	case req.Raw:
		refuse = `raw mode is not supported: the prompt always goes through the model's chat template; /v1/completions takes a raw prompt`
	case req.Template != "":
		refuse = "a custom template is not supported: the chat template is the model's own"
	case req.Suffix != "":
		refuse = "suffix (fill in the middle) is not supported"
	case !isNull(req.Context):
		refuse = "context is deprecated in Ollama and not supported: send the conversation to /api/chat"
	}
	if refuse != "" {
		writeError(w, core.Errorf(core.CodeInvalidRequest, "%s", refuse))
		return
	}
	creq, keep, warns, err := req.base(m)
	if err != nil {
		writeError(w, err)
		return
	}
	name := req.Model
	spec := &runSpec{req: creq, stream: req.streaming(), keep: keep, warnings: warns}
	spec.render = func(f fragment, fin *final) any {
		resp := generateResponse{Model: name, CreatedAt: time.Now().UTC(), Response: f.content, Thinking: f.thinking}
		if fin != nil {
			resp.Done, resp.DoneReason, resp.timings, resp.Warnings = true, fin.reason, fin.timings, fin.warnings
		}
		return resp
	}
	if req.Prompt == "" && req.System == "" {
		h.preload(w, r, m, spec)
		return
	}
	if req.System != "" {
		spec.req.Messages = append(spec.req.Messages, core.Message{Role: core.RoleSystem, Content: req.System})
	}
	if req.Prompt != "" {
		spec.req.Messages = append(spec.req.Messages, core.Message{Role: core.RoleUser, Content: req.Prompt})
	}
	h.serve(w, r, m, spec)
}

// preload answers a request with nothing to say, which in Ollama loads the
// model, or unloads it when keep_alive is 0: how clients warm a model up.
func (h *handler) preload(w http.ResponseWriter, r *http.Request, m registry.Manifest, spec *runSpec) {
	reason := "load"
	switch {
	case spec.keep.Unload:
		reason = "unload"
		if h.deps.Admin != nil {
			if !h.mayManage(r) {
				writeError(w, core.Errorf(core.CodeForbidden, "unloading a model needs an administrative key"))
				return
			}
			// A model that is not resident is already unloaded.
			if err := h.deps.Admin.Unload(r.Context(), m.ID); err != nil && core.AsError(err).Code != core.CodeInvalidRequest {
				writeError(w, err)
				return
			}
		}
	case h.deps.Admin != nil:
		h.pin(r, m, spec.keep)
		if _, err := h.deps.Admin.Load(r.Context(), m.ID); err != nil {
			writeError(w, err)
			return
		}
	}
	httpx.WriteJSON(w, http.StatusOK, spec.render(fragment{}, &final{reason: reason}))
}

// pin applies a negative keep_alive before the request runs, so that the
// instance it loads starts pinned.
func (h *handler) pin(r *http.Request, m registry.Manifest, keep keepAlive) {
	if keep.Forever && h.mayManage(r) {
		_ = h.deps.Admin.Pin(r.Context(), m.ID, true)
	}
}

// unloadAfter applies keep_alive 0 once the answer has gone out.
func (h *handler) unloadAfter(r *http.Request, m registry.Manifest, keep keepAlive) {
	if keep.Unload && h.mayManage(r) {
		_ = h.deps.Admin.Unload(context.WithoutCancel(r.Context()), m.ID)
	}
}

func (h *handler) serve(w http.ResponseWriter, r *http.Request, m registry.Manifest, spec *runSpec) {
	svc, ok := h.service(w)
	if !ok {
		return
	}
	h.pin(r, m, spec.keep)
	start := time.Now()
	finish := func(res *core.ChatResult) *final {
		return &final{reason: doneReason(res.FinishReason), timings: timingsOf(start, res),
			warnings: append(append([]core.Warning(nil), res.Warnings...), spec.warnings...)}
	}
	if !spec.stream {
		res, err := svc.Chat(r.Context(), spec.req, nil)
		if err != nil {
			writeError(w, err)
			return
		}
		httpx.WriteJSON(w, http.StatusOK, spec.render(fragment{
			content: res.Message.Content, thinking: res.Message.ReasoningContent, calls: toolCalls(res.Message.ToolCalls, 0),
		}, finish(res)))
		h.unloadAfter(r, m, spec.keep)
		return
	}
	if h.stream(w, r.Context(), svc, spec, finish) {
		h.unloadAfter(r, m, spec.keep)
	}
}

// stream runs a generation that writes its pieces as lines of JSON. It reports
// whether the answer was delivered whole.
//
// The response opens at the first piece, or when the keep-alive is due during a
// long prefill: until then nothing is written, so a request that fails early
// (unknown model, context overflow, a full queue) still gets its HTTP status. A
// failure after that is a last line {"error": …}, the only way left to report
// it, and a client that went away is not reported at all. A stream that is quiet
// is kept open with an empty piece, which every client reads as a piece: a blank
// line would not parse.
func (h *handler) stream(w http.ResponseWriter, ctx context.Context, svc core.Service, spec *runSpec, finish func(*core.ChatResult) *final) bool {
	type result struct {
		res *core.ChatResult
		err error
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	deltas := make(chan core.Delta, 64)
	done := make(chan result, 1)
	go func() {
		res, err := svc.Chat(ctx, spec.req, func(d core.Delta) error {
			select {
			case deltas <- d:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		close(deltas)
		done <- result{res, err}
	}()

	var st *httpx.Stream
	defer func() {
		if st != nil {
			st.Close()
		}
	}()
	open := func() error {
		if st != nil {
			return nil
		}
		var err error
		st, err = httpx.NewNDJSON(w)
		return err
	}
	abandon := func() bool {
		cancel()
		<-done
		return false
	}
	calls := 0
	early := time.NewTimer(15 * time.Second)
	defer early.Stop()
	var beat <-chan time.Time
	for {
		select {
		case d, ok := <-deltas:
			if !ok {
				res := <-done
				if res.err != nil {
					switch {
					case st == nil:
						writeError(w, res.err)
					case !errors.Is(res.err, context.Canceled):
						_ = st.Line(errorBody{Error: core.AsError(res.err).Message})
					}
					return false
				}
				if err := open(); err != nil {
					writeError(w, err)
					return false
				}
				return st.Line(spec.render(fragment{}, finish(res.res))) == nil
			}
			f := fragment{content: d.Content, thinking: d.ReasoningContent, calls: toolCalls(d.ToolCalls, calls)}
			calls += len(d.ToolCalls)
			if f.empty() {
				continue
			}
			if err := open(); err != nil {
				cancel()
				<-done
				writeError(w, err)
				return false
			}
			beat = st.Heartbeat()
			if st.Line(spec.render(f, nil)) != nil {
				return abandon()
			}
		case <-early.C:
			if st != nil {
				continue
			}
			if err := open(); err != nil {
				cancel()
				<-done
				writeError(w, err)
				return false
			}
			beat = st.Heartbeat()
			if st.Line(spec.render(fragment{}, nil)) != nil {
				return abandon()
			}
		case <-beat:
			if st.Line(spec.render(fragment{}, nil)) != nil {
				return abandon()
			}
		}
	}
}
