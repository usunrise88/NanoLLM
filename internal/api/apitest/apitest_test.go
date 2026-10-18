// Package apitest exercises both dialects end to end over HTTP, against the
// engine on the fake backend.
package apitest

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/api/adapter"
	_ "github.com/usunrise88/nanollm/internal/api/native"
	_ "github.com/usunrise88/nanollm/internal/api/openai"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/engine"
	"github.com/usunrise88/nanollm/internal/pool"
	"github.com/usunrise88/nanollm/internal/registry"
)

func server(t *testing.T) *httptest.Server {
	t.Helper()
	p := pool.New(func(ctx context.Context, k pool.Key) (backend.Model, any, error) {
		m, err := fake.New(fake.Options{}).Load(ctx, backend.LoadSpec{NCtx: k.NCtx})
		return m, nil, err
	}, 1)
	t.Cleanup(p.Close)
	svc := engine.New(engine.Config{DefaultModel: "tiny", PromptCache: true, MaxTokensCap: 256}, p,
		func(ctx context.Context, model, device string) (engine.Target, error) {
			if model != "tiny" {
				return engine.Target{}, core.Errorf(core.CodeModelNotFound, "no model %q", model).WithParam("model")
			}
			return engine.Target{Key: pool.Key{Model: model, Backend: "fake", Device: "cpu", NCtx: 4096}}, nil
		})
	mux := http.NewServeMux()
	if err := adapter.MountAll(mux, []string{"openai", "native"}, adapter.Deps{
		Service: svc, Catalog: []registry.Manifest{{ID: "tiny"}}, ConfigSnapshot: func() any { return nil },
	}); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, srv *httptest.Server, path, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(srv.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func decodeJSON(t *testing.T, r io.Reader, v any) {
	t.Helper()
	if err := json.NewDecoder(r).Decode(v); err != nil {
		t.Fatal(err)
	}
}

// events reads an SSE body into (event, data) pairs.
func events(t *testing.T, r io.Reader) [][2]string {
	t.Helper()
	var out [][2]string
	sc := bufio.NewScanner(r)
	name := ""
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			out = append(out, [2]string{name, strings.TrimPrefix(line, "data: ")})
			name = ""
		}
	}
	return out
}

func TestOpenAIChat(t *testing.T) {
	srv := server(t)
	resp := post(t, srv, "/v1/chat/completions", `{"model":"tiny","messages":[{"role":"user","content":"hi there [[fake:echo]]"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var body struct {
		Object  string
		Choices []struct {
			Message struct {
				Role    string
				Content *string
			}
			FinishReason string `json:"finish_reason"`
		}
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}
	}
	decodeJSON(t, resp.Body, &body)
	if body.Object != "chat.completion" || *body.Choices[0].Message.Content != "hi there" || body.Choices[0].FinishReason != "stop" {
		t.Fatalf("%+v", body)
	}
	if body.Usage.TotalTokens != body.Usage.PromptTokens+body.Usage.CompletionTokens {
		t.Fatalf("usage %+v", body.Usage)
	}
}

func TestOpenAIStreamAndTools(t *testing.T) {
	srv := server(t)
	resp := post(t, srv, "/v1/chat/completions", `{"messages":[{"role":"user","content":"[[fake:tool f {a:1}]]"}],
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
		"stream":true,"stream_options":{"include_usage":true}}`)
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content type %q", ct)
	}
	evs := events(t, resp.Body)
	if evs[len(evs)-1][1] != "[DONE]" {
		t.Fatalf("no [DONE]: %v", evs)
	}
	var sawRole, sawCall, sawFinish, sawUsage bool
	for _, ev := range evs[:len(evs)-1] {
		var c struct {
			Choices []struct {
				Delta struct {
					Role      string
					ToolCalls []struct {
						Index    int
						ID       string
						Function struct{ Name, Arguments string }
					} `json:"tool_calls"`
				}
				FinishReason *string `json:"finish_reason"`
			}
			Usage *struct {
				TotalTokens int `json:"total_tokens"`
			}
		}
		json.Unmarshal([]byte(ev[1]), &c)
		if c.Usage != nil {
			sawUsage = true
			continue
		}
		d := c.Choices[0]
		sawRole = sawRole || d.Delta.Role == "assistant"
		if len(d.Delta.ToolCalls) == 1 && d.Delta.ToolCalls[0].Function.Arguments == `{"a":1}` && d.Delta.ToolCalls[0].ID != "" {
			sawCall = true
		}
		if d.FinishReason != nil && *d.FinishReason == "tool_calls" {
			sawFinish = true
		}
	}
	if !sawRole || !sawCall || !sawFinish || !sawUsage {
		t.Fatalf("role %v call %v finish %v usage %v: %v", sawRole, sawCall, sawFinish, sawUsage, evs)
	}
}

func TestOpenAIErrors(t *testing.T) {
	srv := server(t)
	for _, tc := range []struct {
		body   string
		status int
		typ    string
	}{
		{`{"model":"nope","messages":[{"role":"user","content":"x"}]}`, 404, "invalid_request_error"},
		{`{"messages":[{"role":"user","content":"x"}],"n":2}`, 400, "invalid_request_error"},
		{`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}`, 400, "invalid_request_error"},
		{`not json`, 400, "invalid_request_error"},
		// A streamed request that fails before the first token still gets its status.
		{`{"model":"nope","stream":true,"messages":[{"role":"user","content":"x"}]}`, 404, "invalid_request_error"},
	} {
		resp := post(t, srv, "/v1/chat/completions", tc.body)
		var env struct {
			Error struct{ Message, Type, Code string }
		}
		decodeJSON(t, resp.Body, &env)
		if resp.StatusCode != tc.status || env.Error.Type != tc.typ || env.Error.Message == "" {
			t.Errorf("%s: %d %+v", tc.body, resp.StatusCode, env)
		}
	}
}

func TestOpenAICompletionsAndModels(t *testing.T) {
	srv := server(t)
	resp := post(t, srv, "/v1/completions", `{"prompt":"[[fake:loop]]","max_tokens":5}`)
	var body struct {
		Object  string
		Choices []struct {
			Text         string
			FinishReason string `json:"finish_reason"`
		}
	}
	decodeJSON(t, resp.Body, &body)
	if body.Object != "text_completion" || body.Choices[0].Text != "la la" || body.Choices[0].FinishReason != "length" {
		t.Fatalf("%+v", body)
	}
	r, _ := http.Get(srv.URL + "/v1/models")
	var list struct {
		Data []struct{ ID string }
	}
	decodeJSON(t, r.Body, &list)
	r.Body.Close()
	if len(list.Data) != 1 || list.Data[0].ID != "tiny" {
		t.Fatalf("%+v", list)
	}
}

func TestNativeChatAndStream(t *testing.T) {
	srv := server(t)
	resp := post(t, srv, "/api/v1/chat", `{"messages":[{"role":"user","content":"[[fake:think]]"}],"session_id":"s"}`)
	var res core.ChatResult
	decodeJSON(t, resp.Body, &res)
	if res.Message.ReasoningContent == "" || res.Message.Content == "" || res.Stats.Backend != "fake" {
		t.Fatalf("%+v", res)
	}
	resp = post(t, srv, "/api/v1/chat", `{"messages":[{"role":"user","content":"ab [[fake:echo]]"}],"stream":true}`)
	evs := events(t, resp.Body)
	text := ""
	for _, ev := range evs {
		if ev[0] == "delta" {
			var d core.Delta
			json.Unmarshal([]byte(ev[1]), &d)
			text += d.Content
		}
	}
	if text != "ab" || evs[len(evs)-1][0] != "done" {
		t.Fatalf("%q %v", text, evs)
	}
	// The native dialect is strict about field names.
	resp = post(t, srv, "/api/v1/chat", `{"messages":[{"role":"user","content":"x"}],"session_idd":"s"}`)
	if resp.StatusCode != 400 || resp.Header.Get("Content-Type") != "application/problem+json" {
		t.Fatalf("unknown field: %d %s", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
}

func TestNativeTokenize(t *testing.T) {
	srv := server(t)
	resp := post(t, srv, "/api/v1/tokenize", `{"text":"hello"}`)
	var tr core.TokenizeResult
	decodeJSON(t, resp.Body, &tr)
	if tr.Count != 5 {
		t.Fatalf("%+v", tr)
	}
	b, _ := json.Marshal(map[string]any{"tokens": tr.Tokens})
	resp = post(t, srv, "/api/v1/detokenize", string(b))
	var dr core.DetokenizeResult
	decodeJSON(t, resp.Body, &dr)
	if dr.Text != "hello" {
		t.Fatalf("%+v", dr)
	}
}
