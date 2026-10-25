package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/api/adapter"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/engine"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/pool"
	"github.com/usunrise88/nanollm/internal/registry"
)

const modelID = "gemma-4-e4b-qat"

var modelDigest = strings.Repeat("ab", 32)

func ptr[T any](v T) *T { return &v }

var testCatalog = []registry.Manifest{{
	ID: modelID, Family: "gemma4", Arch: "gemma4", Params: "E4B", Quant: "UD-Q4_K_XL", Format: registry.FormatGGUF,
	ContextMax: 131072, License: "Apache-2.0",
	Files:            []registry.File{{Role: registry.RoleModel, Path: "dir/model.gguf", Size: 1000, SHA256: modelDigest}},
	SamplingDefaults: &registry.SamplingDefaults{Temperature: ptr(1.0), TopK: ptr(64), TopP: ptr(0.95)},
}}

// spy wraps the engine to record what a dialect hands to core, and to script
// a service that fails or streams in ways the fake backend does not.
type spy struct {
	core.Service
	mu     sync.Mutex
	got    []core.ChatRequest
	script func(ctx context.Context, req core.ChatRequest, sink core.Sink) (*core.ChatResult, error)
}

func (s *spy) Chat(ctx context.Context, req core.ChatRequest, sink core.Sink) (*core.ChatResult, error) {
	s.mu.Lock()
	s.got = append(s.got, req)
	script := s.script
	s.mu.Unlock()
	if script != nil {
		return script(ctx, req, sink)
	}
	return s.Service.Chat(ctx, req, sink)
}

func (s *spy) last() core.ChatRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.got[len(s.got)-1]
}

func (s *spy) setScript(f func(context.Context, core.ChatRequest, core.Sink) (*core.ChatResult, error)) {
	s.mu.Lock()
	s.script = f
	s.mu.Unlock()
}

// testAdmin is core.Admin over the same pool the engine uses, like the
// server's runtime is.
type testAdmin struct {
	p *pool.Pool
	// disk is the state of each model's files; absent means complete.
	mu        sync.Mutex
	disk      map[string]string
	pins      map[string]bool
	unloads   int
	downloads []string
	progress  []core.DownloadProgress
	dlErr     error
}

func (a *testAdmin) key(id string) pool.Key {
	return pool.Key{Model: id, Backend: "fake", Device: "cpu", NCtx: 4096}
}

func (a *testAdmin) Models(context.Context) ([]core.ModelInfo, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	var out []core.ModelInfo
	for _, m := range testCatalog {
		info := core.ModelInfo{ID: m.ID, Family: m.Family, Params: m.Params, Quant: m.Quant, Disk: registry.StateComplete, SizeBytes: 1000}
		if s, ok := a.disk[m.ID]; ok {
			info.Disk = s
		}
		for _, s := range a.p.Status() {
			if s.Key.Model == m.ID {
				info.Loaded = append(info.Loaded, s)
			}
		}
		out = append(out, info)
	}
	return out, nil
}

func (a *testAdmin) Download(_ context.Context, id string, progress func(core.DownloadProgress)) error {
	a.mu.Lock()
	a.downloads = append(a.downloads, id)
	steps, err := a.progress, a.dlErr
	a.mu.Unlock()
	for _, p := range steps {
		if progress != nil {
			progress(p)
		}
	}
	return err
}

func (a *testAdmin) Load(ctx context.Context, id string) (*core.ModelInfo, error) {
	_, release, err := a.p.Acquire(ctx, a.key(id))
	if err != nil {
		return nil, err
	}
	release()
	return &core.ModelInfo{ID: id}, nil
}

func (a *testAdmin) Unload(_ context.Context, id string) error {
	a.mu.Lock()
	a.unloads++
	a.mu.Unlock()
	if err := a.p.Unload(a.key(id)); err != nil {
		return core.Errorf(core.CodeInvalidRequest, "model %s is not loaded", id)
	}
	return nil
}

func (a *testAdmin) Pin(_ context.Context, id string, pin bool) error {
	a.mu.Lock()
	a.pins[id] = pin
	a.mu.Unlock()
	a.p.Pin(a.key(id), pin)
	return nil
}

func (a *testAdmin) Reload(context.Context, string) (*core.ModelInfo, error) { return nil, nil }
func (a *testAdmin) Devices(context.Context) ([]core.DeviceInfo, error)      { return nil, nil }

type setup struct {
	fake    fake.Options
	noAdmin bool
	keys    []httpx.KeySpec
	log     *slog.Logger
}

type env struct {
	srv   *httptest.Server
	h     http.Handler
	spy   *spy
	admin *testAdmin
}

func newEnv(t *testing.T, s setup) *env {
	t.Helper()
	p := pool.New(func(ctx context.Context, k pool.Key) (backend.Model, any, error) {
		m, err := fake.New(s.fake).Load(ctx, backend.LoadSpec{NCtx: k.NCtx})
		return m, nil, err
	}, 1)
	t.Cleanup(p.Close)
	// The engine knows the catalog id and nothing else, so a request that
	// reaches it with an alias or a tag was not resolved by the dialect.
	svc := engine.New(engine.Config{DefaultModel: modelID, PromptCache: true, MaxTokensCap: 100000}, p,
		func(_ context.Context, model, _ string) (engine.Target, error) {
			if model != modelID {
				return engine.Target{}, core.Errorf(core.CodeModelNotFound, "no model %q", model)
			}
			return engine.Target{Key: pool.Key{Model: model, Backend: "fake", Device: "cpu", NCtx: 4096}, Manifest: testCatalog[0]}, nil
		})
	e := &env{spy: &spy{Service: svc}, admin: &testAdmin{p: p, disk: map[string]string{}, pins: map[string]bool{}}}
	deps := adapter.Deps{Service: e.spy, Catalog: testCatalog, DefaultModel: func() string { return modelID }, ConfigSnapshot: func() any { return nil }}
	if !s.noAdmin {
		deps.Admin = e.admin
	}
	mux := http.NewServeMux()
	if err := adapter.MountAll(mux, []string{"ollama"}, deps); err != nil {
		t.Fatal(err)
	}
	var mw []httpx.Middleware
	if s.log != nil {
		mw = append(mw, httpx.AccessLog(s.log))
	}
	if s.keys != nil {
		store, err := httpx.NewKeyStore(s.keys)
		if err != nil {
			t.Fatal(err)
		}
		mw = append(mw, httpx.Auth(store))
	}
	e.h = httpx.Chain(mux, mw...)
	e.srv = httptest.NewServer(e.h)
	t.Cleanup(e.srv.Close)
	return e
}

func (e *env) do(t *testing.T, method, path, body, token string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(t.Context(), method, e.srv.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func (e *env) post(t *testing.T, path, body string) *http.Response {
	t.Helper()
	return e.do(t, http.MethodPost, path, body, "")
}

func (e *env) get(t *testing.T, path string) *http.Response {
	t.Helper()
	return e.do(t, http.MethodGet, path, "", "")
}

// line is any line a chat, generate or pull response can hold. A pointer says
// that the key was there.
type line struct {
	Model      string    `json:"model"`
	CreatedAt  time.Time `json:"created_at"`
	Done       bool      `json:"done"`
	DoneReason string    `json:"done_reason"`
	Message    struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		Thinking  string `json:"thinking"`
		ToolCalls []struct {
			ID       string `json:"id"`
			Function struct {
				Index     int            `json:"index"`
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"message"`
	Response           string `json:"response"`
	Thinking           string `json:"thinking"`
	TotalDuration      *int64 `json:"total_duration"`
	LoadDuration       *int64 `json:"load_duration"`
	PromptEvalCount    *int   `json:"prompt_eval_count"`
	PromptEvalDuration *int64 `json:"prompt_eval_duration"`
	EvalCount          *int   `json:"eval_count"`
	EvalDuration       *int64 `json:"eval_duration"`
	Status             string `json:"status"`
	Digest             string `json:"digest"`
	Total              int64  `json:"total"`
	Completed          int64  `json:"completed"`
	Error              string `json:"error"`
	Warnings           []warn `json:"warnings"`
}

type warn struct{ Code, Message string }

func decodeJSON[T any](t *testing.T, resp *http.Response) T {
	t.Helper()
	var v T
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		t.Fatalf("status %d: %v", resp.StatusCode, err)
	}
	return v
}

// lines reads a response as newline-delimited JSON, failing on a line that is
// not one JSON value: that is what a client does.
func lines(t *testing.T, resp *http.Response) []line {
	t.Helper()
	var out []line
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var l line
		if err := json.Unmarshal(sc.Bytes(), &l); err != nil {
			t.Fatalf("line %q: %v", sc.Text(), err)
		}
		out = append(out, l)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

func wantStatus(t *testing.T, resp *http.Response, status int) {
	t.Helper()
	if resp.StatusCode != status {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d, want %d: %s", resp.StatusCode, status, body)
	}
}

func wantError(t *testing.T, resp *http.Response, status int, contains string) {
	t.Helper()
	body, _ := io.ReadAll(resp.Body)
	var out struct{ Error string }
	if err := json.Unmarshal(body, &out); err != nil || out.Error == "" {
		t.Fatalf("not an Ollama error: %s", body)
	}
	if resp.StatusCode != status || !strings.Contains(out.Error, contains) {
		t.Fatalf("status %d (want %d), error %q (want it to contain %q)", resp.StatusCode, status, out.Error, contains)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("content type %q", ct)
	}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func msgs(content string) string {
	b, _ := json.Marshal([]map[string]string{{"role": "user", "content": content}})
	return `"messages":` + string(b)
}

// --- basics ---

func TestRootAndVersion(t *testing.T) {
	e := newEnv(t, setup{})
	for _, method := range []string{http.MethodGet, http.MethodHead} {
		resp := e.do(t, method, "/", "", "")
		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/plain") ||
			(method == http.MethodGet && string(body) != "Ollama is running") {
			t.Errorf("%s /: %d %q %q", method, resp.StatusCode, resp.Header.Get("Content-Type"), body)
		}
	}
	// The root is the root and nothing under it.
	wantStatus(t, e.get(t, "/nonesuch"), 404)

	v := decodeJSON[map[string]string](t, e.get(t, "/api/version"))
	if v["version"] != CompatVersion || CompatVersion == "" {
		t.Errorf("version %v", v)
	}
}

func TestChat(t *testing.T) {
	e := newEnv(t, setup{})
	for _, body := range []string{
		`{"model":"gemma-4-e4b-qat","stream":false,` + msgs("hi there [[fake:echo]]") + `}`,
		// Ollama clients send these as a matter of course.
		`{"model":"gemma-4-e4b-qat","stream":false,"keep_alive":"5m","options":{},"format":"",` + msgs("hi there [[fake:echo]]") + `}`,
	} {
		resp := e.post(t, "/api/chat", body)
		wantStatus(t, resp, 200)
		if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("content type %q", ct)
		}
		l := decodeJSON[line](t, resp)
		if l.Model != modelID || l.Message.Role != "assistant" || l.Message.Content != "hi there" || !l.Done || l.DoneReason != "stop" {
			t.Fatalf("%+v", l)
		}
		// A duration is present, not necessarily above zero: the instant fake
		// finishes inside one tick of the Windows clock.
		if l.CreatedAt.IsZero() || l.TotalDuration == nil || *l.TotalDuration < 0 || l.LoadDuration == nil ||
			l.PromptEvalCount == nil || *l.PromptEvalCount <= 0 || l.PromptEvalDuration == nil ||
			l.EvalCount == nil || *l.EvalCount <= 0 || l.EvalDuration == nil {
			t.Errorf("the last line lacks its timings: %+v", l)
		}
		if len(l.Warnings) != 0 {
			t.Errorf("warnings %v", l.Warnings)
		}
	}
}

// Streaming is the default, unlike OpenAI's.
func TestChatStreamsByDefault(t *testing.T) {
	e := newEnv(t, setup{})
	whole := decodeJSON[line](t, e.post(t, "/api/chat", `{"model":"gemma-4-e4b-qat","stream":false,`+msgs("[[fake:long 120]]")+`}`))

	for _, stream := range []string{``, `"stream":true,`, `"stream":null,`} {
		resp := e.post(t, "/api/chat", `{"model":"gemma-4-e4b-qat",`+stream+msgs("[[fake:long 120]]")+`}`)
		wantStatus(t, resp, 200)
		if ct := resp.Header.Get("Content-Type"); ct != "application/x-ndjson" {
			t.Fatalf("content type %q", ct)
		}
		ls := lines(t, resp)
		if len(ls) < 3 {
			t.Fatalf("%d lines", len(ls))
		}
		var text string
		for i, l := range ls {
			last := i == len(ls)-1
			if l.Done != last || l.Message.Role != "assistant" || l.Model != modelID {
				t.Fatalf("line %d: %+v", i, l)
			}
			if !last && (l.TotalDuration != nil || l.DoneReason != "") {
				t.Errorf("line %d carries final fields: %+v", i, l)
			}
			text += l.Message.Content
		}
		final := ls[len(ls)-1]
		if text != whole.Message.Content || final.Message.Content != "" || final.DoneReason != "stop" ||
			final.TotalDuration == nil || final.EvalCount == nil || *final.EvalCount != *whole.EvalCount {
			t.Errorf("streamed %q (%+v), buffered %q", text, final, whole.Message.Content)
		}
	}
}

func TestGenerate(t *testing.T) {
	e := newEnv(t, setup{})
	resp := e.post(t, "/api/generate", `{"model":"gemma4","stream":false,"system":"be brief","prompt":"hello [[fake:echo]]"}`)
	wantStatus(t, resp, 200)
	l := decodeJSON[line](t, resp)
	if l.Model != "gemma4" || l.Response != "hello" || !l.Done || l.DoneReason != "stop" || l.TotalDuration == nil {
		t.Fatalf("%+v", l)
	}
	got := e.spy.last()
	if len(got.Messages) != 2 || got.Messages[0].Role != "system" || got.Messages[0].Content != "be brief" ||
		got.Messages[1].Role != "user" || got.Model != modelID {
		t.Errorf("sent to core: %+v", got)
	}

	// Streaming by default, the answer in "response".
	ls := lines(t, e.post(t, "/api/generate", `{"model":"gemma-4-e4b-qat","prompt":"[[fake:long 60]]"}`))
	var text string
	for _, l := range ls[:len(ls)-1] {
		if l.Done || l.Response == "" {
			t.Fatalf("%+v", l)
		}
		text += l.Response
	}
	if last := ls[len(ls)-1]; !last.Done || last.Response != "" || len(text) != 60 {
		t.Errorf("%d bytes, last %+v", len(text), last)
	}
}

func TestModelNames(t *testing.T) {
	e := newEnv(t, setup{})
	for _, tc := range []struct {
		name   string
		status int
		err    string
	}{
		{name: "gemma-4-e4b-qat"},
		{name: "gemma-4-e4b-qat:latest"},
		{name: "Gemma-4-E4B-QAT:Latest"},
		{name: " gemma-4-e4b-qat "},
		{name: "gemma4"},
		{name: "gemma4:latest"},
		{name: "gemma4:e4b"},
		{name: "library/gemma4:e4b"},
		{name: "registry.ollama.ai/library/gemma4:e4b-it-qat"},
		{name: "llama3.2", status: 404, err: "model 'llama3.2' not found"},
		{name: "gemma-4-e4b-qat:q8_0", status: 404, err: "not found"},
		{name: "gemma4:e2b", status: 404, err: "not found"},
		{name: "", status: 400, err: "model is required"},
	} {
		body, _ := json.Marshal(map[string]any{"model": tc.name, "stream": false, "messages": []map[string]string{{"role": "user", "content": "x"}}})
		resp := e.post(t, "/api/chat", string(body))
		if tc.status != 0 {
			wantError(t, resp, tc.status, tc.err)
			continue
		}
		wantStatus(t, resp, 200)
		// The response names the model the way the request did.
		if l := decodeJSON[line](t, resp); l.Model != tc.name {
			t.Errorf("asked for %q, answered as %q", tc.name, l.Model)
		}
		if got := e.spy.last().Model; got != modelID {
			t.Errorf("%q reached core as %q", tc.name, got)
		}
	}
	// Unknown before streaming: still a status, not a stream.
	wantError(t, e.post(t, "/api/chat", `{"model":"nope",`+msgs("x")+`}`), 404, "'nope' not found")
	wantError(t, e.post(t, "/api/generate", `{"model":"nope","prompt":"x"}`), 404, "'nope' not found")
}

// --- request mapping ---

func TestOptions(t *testing.T) {
	e := newEnv(t, setup{})
	for _, tc := range []struct {
		name, options string
		check         func(core.ChatRequest) bool
		ignored       string
	}{
		{"sampling", `{"temperature":0.2,"top_p":0.9,"top_k":40,"min_p":0.05,"repeat_penalty":1.1,"presence_penalty":0.5,"frequency_penalty":0.25,"seed":7,"num_predict":33,"stop":["END","STOP"]}`,
			func(r core.ChatRequest) bool {
				s := r.Sampling
				return *s.Temperature == 0.2 && *s.TopP == 0.9 && *s.TopK == 40 && *s.MinP == 0.05 && *s.RepeatPenalty == 1.1 &&
					*s.PresencePenalty == 0.5 && *s.FrequencyPenalty == 0.25 && *s.Seed == 7 && r.MaxTokens == 33 &&
					len(r.Stop) == 2 && r.Stop[0] == "END" && r.Stop[1] == "STOP"
			}, ""},
		{"integers sent as floats", `{"top_k":40.0,"seed":12345678901,"num_predict":50.0}`,
			func(r core.ChatRequest) bool {
				return *r.Sampling.TopK == 40 && *r.Sampling.Seed == 12345678901 && r.MaxTokens == 50
			}, ""},
		{"a single stop string", `{"stop":"END"}`,
			func(r core.ChatRequest) bool { return len(r.Stop) == 1 && r.Stop[0] == "END" }, ""},
		{"num_predict -1 and -2 mean the cap", `{"num_predict":-1}`,
			func(r core.ChatRequest) bool { return r.MaxTokens == 0 }, ""},
		{"nulls are unset", `{"temperature":null,"top_k":null,"stop":null}`,
			func(r core.ChatRequest) bool {
				return r.Sampling.Temperature == nil && r.Sampling.TopK == nil && r.Stop == nil
			}, ""},
		{"nothing sent, the model card's sampling", `{}`,
			func(r core.ChatRequest) bool {
				return r.Sampling.Temperature == nil && r.Sampling.TopK == nil && r.MaxTokens == 0
			}, ""},
		{"options left out are listed, runtime knobs are not", `{"num_ctx":8192,"mirostat":2,"num_gpu":1,"num_thread":8,"temperature":0.5}`,
			func(r core.ChatRequest) bool { return *r.Sampling.Temperature == 0.5 }, "mirostat, num_ctx"},
	} {
		resp := e.post(t, "/api/chat", `{"model":"gemma4","stream":false,"options":`+tc.options+`,`+msgs("x")+`}`)
		wantStatus(t, resp, 200)
		l := decodeJSON[line](t, resp)
		if got := e.spy.last(); !tc.check(got) {
			t.Errorf("%s: core got %+v", tc.name, got)
		}
		switch {
		case tc.ignored == "" && len(l.Warnings) != 0:
			t.Errorf("%s: warnings %v", tc.name, l.Warnings)
		case tc.ignored != "" && (len(l.Warnings) != 1 || l.Warnings[0].Code != "option_ignored" || !strings.HasSuffix(l.Warnings[0].Message, tc.ignored)):
			t.Errorf("%s: warnings %v", tc.name, l.Warnings)
		}
	}

	// Options of /api/generate go the same way.
	wantStatus(t, e.post(t, "/api/generate", `{"model":"gemma4","stream":false,"prompt":"x","options":{"temperature":0.3,"num_predict":9}}`), 200)
	if r := e.spy.last(); *r.Sampling.Temperature != 0.3 || r.MaxTokens != 9 {
		t.Errorf("generate options: %+v", r)
	}
}

func TestFormat(t *testing.T) {
	e := newEnv(t, setup{})
	schema := `{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}`
	for _, tc := range []struct {
		format string
		want   func(*core.ResponseFormat) bool
	}{
		{`"json"`, func(f *core.ResponseFormat) bool { return f != nil && f.Type == core.FormatJSONObject }},
		{schema, func(f *core.ResponseFormat) bool {
			return f != nil && f.Type == core.FormatJSONSchema && string(f.Schema) == schema
		}},
		{`""`, func(f *core.ResponseFormat) bool { return f == nil }},
		{`null`, func(f *core.ResponseFormat) bool { return f == nil }},
	} {
		resp := e.post(t, "/api/chat", `{"model":"gemma4","stream":false,"format":`+tc.format+`,`+msgs("[[fake:json]]")+`}`)
		wantStatus(t, resp, 200)
		if got := e.spy.last().ResponseFormat; !tc.want(got) {
			t.Errorf("format %s reached core as %+v", tc.format, got)
		}
		if l := decodeJSON[line](t, resp); tc.format == `"json"` && !json.Valid([]byte(l.Message.Content)) {
			t.Errorf("not JSON: %q", l.Message.Content)
		}
	}
	wantStatus(t, e.post(t, "/api/generate", `{"model":"gemma4","stream":false,"format":"json","prompt":"[[fake:json]]"}`), 200)
	if f := e.spy.last().ResponseFormat; f == nil || f.Type != core.FormatJSONObject {
		t.Errorf("generate format %+v", f)
	}
	wantError(t, e.post(t, "/api/chat", `{"model":"gemma4","format":"xml",`+msgs("x")+`}`), 400, "invalid format")
	wantError(t, e.post(t, "/api/chat", `{"model":"gemma4","format":5,`+msgs("x")+`}`), 400, "format")
}

func TestLogprobs(t *testing.T) {
	e := newEnv(t, setup{})
	type entry struct {
		Token       string
		Logprob     float64
		Bytes       []int
		TopLogprobs []struct {
			Token   string
			Logprob float64
		} `json:"top_logprobs"`
	}
	type withLogprobs struct{ Logprobs []entry }

	ask := `"model":"gemma4","logprobs":true,"top_logprobs":2,`
	whole := decodeJSON[withLogprobs](t, e.post(t, "/api/chat", `{`+ask+`"stream":false,`+msgs("abc [[fake:echo]]")+`}`))
	if r := e.spy.last(); !r.Logprobs || r.TopLogprobs != 2 {
		t.Errorf("core got %+v", r.Generation)
	}
	if len(whole.Logprobs) < 3 {
		t.Fatalf("%+v", whole)
	}
	for _, lp := range whole.Logprobs {
		if lp.Token == "" || lp.Logprob > 0 || len(lp.Bytes) == 0 || len(lp.TopLogprobs) != 2 {
			t.Errorf("%+v", lp)
		}
	}

	// Streamed, each line carries the log-probabilities of its own tokens.
	var streamed int
	resp := e.post(t, "/api/chat", `{`+ask+msgs("abc [[fake:echo]]")+`}`)
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		var l withLogprobs
		if err := json.Unmarshal(sc.Bytes(), &l); err != nil {
			t.Fatal(err)
		}
		streamed += len(l.Logprobs)
	}
	if streamed != len(whole.Logprobs) {
		t.Errorf("%d entries streamed, %d buffered", streamed, len(whole.Logprobs))
	}

	gen := decodeJSON[withLogprobs](t, e.post(t, "/api/generate", `{`+ask+`"stream":false,"prompt":"abc [[fake:echo]]"}`))
	if len(gen.Logprobs) != len(whole.Logprobs) {
		t.Errorf("generate: %d entries", len(gen.Logprobs))
	}

	// Not asked for, not sent.
	raw, _ := io.ReadAll(e.post(t, "/api/chat", `{"model":"gemma4","stream":false,`+msgs("abc [[fake:echo]]")+`}`).Body)
	if strings.Contains(string(raw), "logprobs") {
		t.Errorf("logprobs without being asked: %s", raw)
	}
}

func TestThink(t *testing.T) {
	e := newEnv(t, setup{})
	for _, tc := range []struct {
		think string
		want  *bool
	}{
		{``, nil}, {`"think":null,`, nil}, {`"think":true,`, ptr(true)}, {`"think":false,`, ptr(false)},
		{`"think":"high",`, ptr(true)}, {`"think":"low",`, ptr(true)},
	} {
		resp := e.post(t, "/api/chat", `{"model":"gemma4","stream":false,`+tc.think+msgs("[[fake:think]]")+`}`)
		wantStatus(t, resp, 200)
		got := e.spy.last().EnableThinking
		if (got == nil) != (tc.want == nil) || (got != nil && *got != *tc.want) {
			t.Errorf("%q reached core as %v", tc.think, got)
		}
		l := decodeJSON[line](t, resp)
		if l.Message.Thinking == "" || l.Message.Content != "Here is a short answer." {
			t.Errorf("%q: thinking %q, content %q", tc.think, l.Message.Thinking, l.Message.Content)
		}
	}

	// Streamed, the thinking arrives before the answer, in its own field.
	ls := lines(t, e.post(t, "/api/chat", `{"model":"gemma4","think":true,`+msgs("[[fake:think]]")+`}`))
	var thought, answer string
	sawAnswer := false
	for _, l := range ls {
		if l.Message.Thinking != "" && (l.Message.Content != "" || sawAnswer) {
			t.Fatalf("thinking after or with the answer: %+v", l)
		}
		sawAnswer = sawAnswer || l.Message.Content != ""
		thought, answer = thought+l.Message.Thinking, answer+l.Message.Content
	}
	if thought == "" || answer != "Here is a short answer." {
		t.Errorf("thinking %q, answer %q", thought, answer)
	}

	l := decodeJSON[line](t, e.post(t, "/api/generate", `{"model":"gemma4","stream":false,"think":true,"prompt":"[[fake:think]]"}`))
	if l.Thinking == "" || l.Response != "Here is a short answer." {
		t.Errorf("generate: %+v", l)
	}
	wantError(t, e.post(t, "/api/chat", `{"model":"gemma4","think":"extreme",`+msgs("x")+`}`), 400, "think")
	// An earlier turn's thinking goes back to core.
	wantStatus(t, e.post(t, "/api/chat", `{"model":"gemma4","stream":false,"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b","thinking":"hmm"},{"role":"user","content":"c"}]}`), 200)
	if m := e.spy.last().Messages[1]; m.ReasoningContent != "hmm" {
		t.Errorf("assistant turn: %+v", m)
	}
}

const weatherTool = `{"type":"function","function":{"name":"get_weather","description":"Weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}`

func TestToolCallObjectArguments(t *testing.T) {
	e := newEnv(t, setup{})
	ask := `[[fake:tool get_weather {city:<|\"|>Paris<|\"|>}]]`
	resp := e.post(t, "/api/chat", `{"model":"gemma4","stream":false,"tools":[`+weatherTool+`],"messages":[{"role":"user","content":"`+ask+`"}]}`)
	wantStatus(t, resp, 200)
	raw, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(raw), `"arguments":{"city":"Paris"}`) {
		t.Errorf("the arguments are not an object: %s", raw)
	}
	var l line
	if err := json.Unmarshal(raw, &l); err != nil {
		t.Fatal(err)
	}
	if len(l.Message.ToolCalls) != 1 || l.Message.ToolCalls[0].Function.Name != "get_weather" ||
		l.Message.ToolCalls[0].Function.Arguments["city"] != "Paris" || !strings.HasPrefix(l.Message.ToolCalls[0].ID, "call_") ||
		l.DoneReason != "stop" || !l.Done {
		t.Fatalf("%+v", l)
	}
	if tools := e.spy.last().Tools; len(tools) != 1 || tools[0].Name != "get_weather" || tools[0].Description != "Weather in a city" ||
		!strings.Contains(string(tools[0].Parameters), `"required":["city"]`) {
		t.Errorf("tools reached core as %+v", tools)
	}

	// Streamed: a line holds the call, then the closing line.
	ls := lines(t, e.post(t, "/api/chat", `{"model":"gemma4","tools":[`+weatherTool+`],"messages":[{"role":"user","content":"`+ask+`"}]}`))
	var calls int
	for _, l := range ls {
		for _, tc := range l.Message.ToolCalls {
			calls++
			if l.Done || tc.Function.Index != 0 || tc.Function.Arguments["city"] != "Paris" {
				t.Errorf("%+v", l)
			}
		}
	}
	if calls != 1 || !ls[len(ls)-1].Done || ls[len(ls)-1].DoneReason != "stop" {
		t.Errorf("%d calls in %+v", calls, ls)
	}
}

func TestToolResultsAreTiedToTheirCalls(t *testing.T) {
	e := newEnv(t, setup{})
	call := func(id, name, args string) string {
		idField := ""
		if id != "" {
			idField = `"id":"` + id + `",`
		}
		return `{` + idField + `"function":{"name":"` + name + `","arguments":` + args + `}}`
	}
	conversation := func(calls []string, results ...string) string {
		return `{"model":"gemma4","stream":false,"tools":[` + weatherTool + `],"messages":[{"role":"user","content":"weather? [[fake:echo]]"},` +
			`{"role":"assistant","content":"","tool_calls":[` + strings.Join(calls, ",") + `]},` + strings.Join(results, ",") + `]}`
	}
	for _, tc := range []struct {
		name string
		body string
		ids  []string // ToolCallID of each tool message, in order
		tool []string // Name of each
	}{
		{"by tool_name",
			conversation([]string{call("", "get_weather", `{"city":"Paris"}`)}, `{"role":"tool","tool_name":"get_weather","content":"{\"temp\":21}"}`),
			[]string{"call_1_0"}, []string{"get_weather"}},
		{"by position, with no name",
			conversation([]string{call("", "a", `{}`), call("", "b", `{"x":1}`)}, `{"role":"tool","content":"1"}`, `{"role":"tool","content":"2"}`),
			[]string{"call_1_0", "call_1_1"}, []string{"a", "b"}},
		{"by name, out of order",
			conversation([]string{call("", "a", `{}`), call("", "b", `{}`)}, `{"role":"tool","tool_name":"b","content":"1"}`, `{"role":"tool","tool_name":"a","content":"2"}`),
			[]string{"call_1_1", "call_1_0"}, []string{"b", "a"}},
		{"by the client's id",
			conversation([]string{call("abc", "get_weather", `{}`)}, `{"role":"tool","tool_call_id":"abc","content":"1"}`),
			[]string{"abc"}, []string{"get_weather"}},
		{"the same tool twice",
			conversation([]string{call("", "get_weather", `{"city":"A"}`), call("", "get_weather", `{"city":"B"}`)},
				`{"role":"tool","tool_name":"get_weather","content":"1"}`, `{"role":"tool","tool_name":"get_weather","content":"2"}`),
			[]string{"call_1_0", "call_1_1"}, []string{"get_weather", "get_weather"}},
	} {
		wantStatus(t, e.post(t, "/api/chat", tc.body), 200)
		got := e.spy.last().Messages
		var results []core.Message
		for _, m := range got {
			if m.Role == core.RoleTool {
				results = append(results, m)
			}
		}
		if len(results) != len(tc.ids) {
			t.Fatalf("%s: %d tool messages in %+v", tc.name, len(results), got)
		}
		for i, m := range results {
			if m.ToolCallID != tc.ids[i] || m.Name != tc.tool[i] {
				t.Errorf("%s: tool message %d is %q/%q, want %q/%q", tc.name, i, m.ToolCallID, m.Name, tc.ids[i], tc.tool[i])
			}
		}
	}

	// The arguments an assistant turn carries go to core as JSON text.
	wantStatus(t, e.post(t, "/api/chat", conversation([]string{call("", "get_weather", `{ "city" : "Paris" }`)}, `{"role":"tool","content":"1"}`)), 200)
	if c := e.spy.last().Messages[1].ToolCalls; len(c) != 1 || c[0].Arguments != `{"city":"Paris"}` || c[0].Name != "get_weather" {
		t.Errorf("calls %+v", c)
	}
	// OpenAI's string form is accepted too.
	wantStatus(t, e.post(t, "/api/chat", conversation([]string{call("", "get_weather", `"{\"city\":\"Rome\"}"`)}, `{"role":"tool","content":"1"}`)), 200)
	if c := e.spy.last().Messages[1].ToolCalls; len(c) != 1 || c[0].Arguments != `{"city":"Rome"}` {
		t.Errorf("calls %+v", c)
	}
	wantError(t, e.post(t, "/api/chat", conversation([]string{call("", "f", `[1]`)}, `{"role":"tool","content":"1"}`)), 400, "arguments must be a JSON object")
}

func TestUnsupportedAndInvalid(t *testing.T) {
	e := newEnv(t, setup{})
	for _, tc := range []struct {
		name, path, body, err string
		status                int
	}{
		{"image in a message", "/api/chat", `{"model":"gemma4","messages":[{"role":"user","content":"x","images":["aGk="]}]}`, "does not accept images", 400},
		{"images in generate", "/api/generate", `{"model":"gemma4","prompt":"x","images":["aGk="]}`, "does not accept images", 400},
		{"raw", "/api/generate", `{"model":"gemma4","prompt":"x","raw":true}`, "raw mode is not supported", 400},
		{"template", "/api/generate", `{"model":"gemma4","prompt":"x","template":"{{ .Prompt }}"}`, "custom template", 400},
		{"suffix", "/api/generate", `{"model":"gemma4","prompt":"x","suffix":"y"}`, "suffix", 400},
		{"context", "/api/generate", `{"model":"gemma4","prompt":"x","context":[1,2]}`, "context", 400},
		{"too many top_logprobs", "/api/chat", `{"model":"gemma4","logprobs":true,"top_logprobs":99,` + msgs("x") + `}`, "top_logprobs", 400},
		{"bad JSON", "/api/chat", `not json`, "not valid JSON", 400},
		{"empty body", "/api/chat", ``, "empty", 400},
		{"content that is not a string", "/api/chat", `{"model":"gemma4","messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]}`, "not valid JSON", 400},
		{"options that are not an object", "/api/chat", `{"model":"gemma4","options":[1],` + msgs("x") + `}`, "options must be an object", 400},
		{"option of the wrong type", "/api/chat", `{"model":"gemma4","options":{"temperature":"hot"},` + msgs("x") + `}`, "temperature must be a number", 400},
		{"fractional integer option", "/api/chat", `{"model":"gemma4","options":{"top_k":1.5},` + msgs("x") + `}`, "top_k must be an integer", 400},
		{"bad stop", "/api/chat", `{"model":"gemma4","options":{"stop":5},` + msgs("x") + `}`, "stop", 400},
		{"bad keep_alive", "/api/chat", `{"model":"gemma4","keep_alive":"soon",` + msgs("x") + `}`, "keep_alive", 400},
		{"tool of another type", "/api/chat", `{"model":"gemma4","tools":[{"type":"retrieval"}],` + msgs("x") + `}`, "only function tools", 400},
		{"tool without a name", "/api/chat", `{"model":"gemma4","tools":[{"type":"function","function":{}}],` + msgs("x") + `}`, "no function name", 400},
		{"unknown role", "/api/chat", `{"model":"gemma4","messages":[{"role":"wizard","content":"x"}]}`, "unknown role", 400},
		{"too long a prompt", "/api/chat", `{"model":"gemma4","stream":false,` + msgs(strings.Repeat("a", 6000)) + `}`, "context holds", 400},
		{"too long a prompt, streamed", "/api/chat", `{"model":"gemma4",` + msgs(strings.Repeat("a", 6000)) + `}`, "context holds", 400},
		{"show without a model", "/api/show", `{}`, "model is required", 400},
		{"pull without a model", "/api/pull", `{}`, "model is required", 400},
	} {
		wantError(t, e.post(t, tc.path, tc.body), tc.status, tc.err)
	}
}

// --- errors from core ---

func TestCoreErrorsKeepTheirStatus(t *testing.T) {
	e := newEnv(t, setup{})
	for _, tc := range []struct {
		code   core.Code
		status int
	}{
		{core.CodeInvalidRequest, 400}, {core.CodeContextOverflow, 400}, {core.CodeModelNotFound, 404},
		{core.CodeDeviceUnavailable, 409}, {core.CodeQueueFull, 503}, {core.CodeModelUnavailable, 503},
		{core.CodeRateLimited, 429}, {core.CodeNotImplemented, 501}, {core.CodeInternal, 500},
	} {
		e.spy.setScript(func(context.Context, core.ChatRequest, core.Sink) (*core.ChatResult, error) {
			return nil, core.Errorf(tc.code, "scripted %s", tc.code).WithCause(io.ErrUnexpectedEOF)
		})
		for _, body := range []string{
			`{"model":"gemma4","stream":false,` + msgs("x") + `}`,
			`{"model":"gemma4","stream":true,` + msgs("x") + `}`,
		} {
			wantError(t, e.post(t, "/api/chat", body), tc.status, "scripted "+string(tc.code))
		}
	}
	// An error that is not ours is "internal error" and nothing more.
	e.spy.setScript(func(context.Context, core.ChatRequest, core.Sink) (*core.ChatResult, error) {
		return nil, io.ErrUnexpectedEOF
	})
	wantError(t, e.post(t, "/api/chat", `{"model":"gemma4",`+msgs("x")+`}`), 500, "internal error")
}

// A failure after the first line is a last line, with the status already sent.
func TestErrorAfterTheStreamStarted(t *testing.T) {
	e := newEnv(t, setup{})
	e.spy.setScript(func(_ context.Context, _ core.ChatRequest, sink core.Sink) (*core.ChatResult, error) {
		_ = sink(core.Delta{Content: "one "})
		_ = sink(core.Delta{Content: "two"})
		return nil, core.Errorf(core.CodeInternal, "the backend broke").WithCause(io.ErrUnexpectedEOF)
	})
	resp := e.post(t, "/api/chat", `{"model":"gemma4",`+msgs("x")+`}`)
	wantStatus(t, resp, 200)
	ls := lines(t, resp)
	if len(ls) != 3 || ls[0].Message.Content != "one " || ls[1].Message.Content != "two" || ls[2].Error != "the backend broke" || ls[2].Done {
		t.Errorf("%+v", ls)
	}
	// The same for /api/generate.
	ls = lines(t, e.post(t, "/api/generate", `{"model":"gemma4","prompt":"x"}`))
	if len(ls) != 3 || ls[2].Error != "the backend broke" {
		t.Errorf("%+v", ls)
	}
}

func TestErrorFromTheBackend(t *testing.T) {
	e := newEnv(t, setup{})
	wantError(t, e.post(t, "/api/chat", `{"model":"gemma4","stream":false,`+msgs("[[fake:error]]")+`}`), 500, "generation failed")
	wantError(t, e.post(t, "/api/chat", `{"model":"gemma4",`+msgs("[[fake:error]]")+`}`), 500, "generation failed")
}

// --- cancellation ---

// A caller that went away before the first byte gets 499 and the code, never a
// 500.
func TestCancelledBeforeTheFirstByte(t *testing.T) {
	e := newEnv(t, setup{})
	for _, tc := range []struct{ path, body string }{
		{"/api/chat", `{"model":"gemma4","stream":false,` + msgs("x") + `}`},
		{"/api/chat", `{"model":"gemma4",` + msgs("x") + `}`},
		{"/api/generate", `{"model":"gemma4","stream":false,"prompt":"x"}`},
		{"/api/generate", `{"model":"gemma4","prompt":"x"}`},
	} {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		req := httptest.NewRequestWithContext(ctx, http.MethodPost, tc.path, strings.NewReader(tc.body))
		rec := httptest.NewRecorder()
		e.h.ServeHTTP(rec, req)
		if rec.Code != core.StatusClientClosedRequest || !strings.Contains(rec.Body.String(), `"error"`) {
			t.Errorf("%s %s: %d %s", tc.path, tc.body, rec.Code, rec.Body.String())
		}
	}
}

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

// requests is the access log so far: the status of each request, by path.
func (s *syncBuffer) requests() map[string][]int {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := map[string][]int{}
	for _, l := range bytes.Split(s.b.Bytes(), []byte("\n")) {
		var rec struct {
			Path   string
			Status int
		}
		if json.Unmarshal(l, &rec) == nil && rec.Path != "" {
			out[rec.Path] = append(out[rec.Path], rec.Status)
		}
	}
	return out
}

// A client that hangs up mid-generation frees the slot at once, and the
// server logs no fault for it.
func TestClientDisconnect(t *testing.T) {
	// 100 tokens a second and a cap of 100000: only a cancellation ends [[fake:loop]].
	logs := &syncBuffer{}
	e := newEnv(t, setup{fake: fake.Options{DecodeRate: 100}, log: slog.New(slog.NewJSONHandler(logs, nil))})

	t.Run("streaming", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, e.srv.URL+"/api/chat", strings.NewReader(`{"model":"gemma4",`+msgs("[[fake:loop]]")+`}`))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(resp.Body)
		for i := 0; i < 3; i++ {
			if !sc.Scan() {
				t.Fatalf("the stream ended early: %v", sc.Err())
			}
		}
		cancel()
		resp.Body.Close()
		waitFor(t, "the request to be logged", func() bool { return len(logs.requests()["/api/chat"]) == 1 })
		if got := logs.requests()["/api/chat"][0]; got >= 500 {
			t.Errorf("a disconnect was logged as %d", got)
		}
		// The slot is free: with 100000 tokens to go it would be held for ever.
		start := time.Now()
		wantStatus(t, e.post(t, "/api/chat", `{"model":"gemma4","stream":false,`+msgs("again [[fake:echo]]")+`}`), 200)
		if time.Since(start) > 5*time.Second {
			t.Errorf("the next request waited %v", time.Since(start))
		}
	})

	t.Run("buffered", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, e.srv.URL+"/api/generate", strings.NewReader(`{"model":"gemma4","stream":false,"prompt":"[[fake:loop]]"}`))
		done := make(chan error, 1)
		go func() {
			resp, err := http.DefaultClient.Do(req)
			if err == nil {
				resp.Body.Close()
			}
			done <- err
		}()
		time.Sleep(200 * time.Millisecond)
		cancel()
		if err := <-done; err == nil {
			t.Fatal("the request completed")
		}
		waitFor(t, "the request to be logged", func() bool { return len(logs.requests()["/api/generate"]) == 1 })
		if got := logs.requests()["/api/generate"][0]; got != core.StatusClientClosedRequest {
			t.Errorf("a disconnect was logged as %d, want 499", got)
		}
		wantStatus(t, e.post(t, "/api/chat", `{"model":"gemma4","stream":false,`+msgs("again [[fake:echo]]")+`}`), 200)
	})
}

// --- model management ---

type tagList struct {
	Models []struct {
		Name, Model, Digest string
		ModifiedAt          time.Time `json:"modified_at"`
		Size                int64
		Details             modelDetails
	}
}

func TestTagsPsShow(t *testing.T) {
	e := newEnv(t, setup{})

	tags := decodeJSON[tagList](t, e.get(t, "/api/tags"))
	if len(tags.Models) != 1 {
		t.Fatalf("%+v", tags)
	}
	m := tags.Models[0]
	if m.Name != modelID+":latest" || m.Model != m.Name || m.Digest != modelDigest || m.Size != 1000 || m.ModifiedAt.IsZero() ||
		m.Details.Format != "gguf" || m.Details.Family != "gemma4" || len(m.Details.Families) != 1 || m.Details.Families[0] != "gemma4" ||
		m.Details.ParameterSize != "E4B" || m.Details.QuantizationLevel != "Q4_0" {
		t.Errorf("%+v", m)
	}

	// Nothing is resident yet.
	ps := decodeJSON[struct{ Models []json.RawMessage }](t, e.get(t, "/api/ps"))
	if ps.Models == nil || len(ps.Models) != 0 {
		t.Errorf("ps before a request: %+v", ps)
	}
	wantStatus(t, e.post(t, "/api/chat", `{"model":"gemma4","stream":false,`+msgs("x")+`}`), 200)
	run := decodeJSON[struct {
		Models []struct {
			Name, Model, Digest string
			Size                int64
			SizeVRAM            int64     `json:"size_vram"`
			ExpiresAt           time.Time `json:"expires_at"`
			ContextLength       int       `json:"context_length"`
			Details             modelDetails
		}
	}](t, e.get(t, "/api/ps"))
	if len(run.Models) != 1 {
		t.Fatalf("%+v", run)
	}
	r := run.Models[0]
	if r.Name != modelID+":latest" || r.Digest != modelDigest || r.Size <= 0 || r.SizeVRAM != 0 || r.ContextLength != 4096 ||
		!r.ExpiresAt.After(time.Now().AddDate(10, 0, 0)) || r.Details.QuantizationLevel != "Q4_0" {
		t.Errorf("%+v", r)
	}

	// A model that is not on disk and not resident is not local.
	e.admin.mu.Lock()
	e.admin.disk[modelID] = registry.StateAbsent
	e.admin.mu.Unlock()
	if err := e.admin.p.Unload(e.admin.key(modelID)); err != nil {
		t.Fatal(err)
	}
	tags = decodeJSON[tagList](t, e.get(t, "/api/tags"))
	if tags.Models == nil || len(tags.Models) != 0 {
		t.Errorf("tags of an empty store: %+v", tags)
	}
	wantError(t, e.post(t, "/api/show", `{"model":"gemma4"}`), 404, "try pulling")
}

func TestShow(t *testing.T) {
	e := newEnv(t, setup{})
	for _, body := range []string{`{"model":"gemma4:e4b"}`, `{"name":"gemma-4-e4b-qat:latest"}`, `{"model":"gemma4","verbose":true}`} {
		resp := e.post(t, "/api/show", body)
		wantStatus(t, resp, 200)
		s := decodeJSON[struct {
			License, Modelfile, Parameters, Template string
			Details                                  modelDetails
			ModelInfo                                map[string]any `json:"model_info"`
			Capabilities                             []string
			ModifiedAt                               time.Time `json:"modified_at"`
		}](t, resp)
		if s.License != "Apache-2.0" || !strings.Contains(s.Modelfile, "FROM "+modelID) || !strings.Contains(s.Parameters, "top_k") ||
			!strings.Contains(s.Parameters, "temperature") || s.Details.QuantizationLevel != "Q4_0" || s.Details.Family != "gemma4" ||
			s.ModelInfo["general.architecture"] != "gemma4" || s.ModelInfo["gemma4.context_length"] != float64(131072) ||
			s.ModifiedAt.IsZero() {
			t.Errorf("%s: %+v", body, s)
		}
		if got := strings.Join(s.Capabilities, ","); got != "completion,tools,thinking" {
			t.Errorf("capabilities %q", got)
		}
	}
	wantError(t, e.post(t, "/api/show", `{"model":"llama3.2"}`), 404, "model 'llama3.2' not found")
}

func TestWithoutAnAdmin(t *testing.T) {
	e := newEnv(t, setup{noAdmin: true})
	tags := decodeJSON[struct{ Models []struct{ Name string } }](t, e.get(t, "/api/tags"))
	if len(tags.Models) != 1 {
		t.Errorf("%+v", tags)
	}
	if ps := decodeJSON[struct{ Models []json.RawMessage }](t, e.get(t, "/api/ps")); ps.Models == nil || len(ps.Models) != 0 {
		t.Errorf("%+v", ps)
	}
	wantStatus(t, e.post(t, "/api/show", `{"model":"gemma4"}`), 200)
	wantStatus(t, e.post(t, "/api/chat", `{"model":"gemma4","stream":false,`+msgs("x")+`}`), 200)
	wantError(t, e.post(t, "/api/pull", `{"model":"gemma4"}`), 501, "not available")
	// A preload has nothing to load with, and says what it is.
	l := decodeJSON[line](t, e.post(t, "/api/chat", `{"model":"gemma4","stream":false}`))
	if !l.Done || l.DoneReason != "load" {
		t.Errorf("%+v", l)
	}
}

func TestPull(t *testing.T) {
	e := newEnv(t, setup{})
	e.admin.progress = []core.DownloadProgress{
		{File: "model.gguf", Downloaded: 400, Total: 1000},
		{File: "model.gguf", Downloaded: 1000, Total: 1000, Percent: 100},
	}
	resp := e.post(t, "/api/pull", `{"model":"gemma4"}`)
	wantStatus(t, resp, 200)
	if ct := resp.Header.Get("Content-Type"); ct != "application/x-ndjson" {
		t.Errorf("content type %q", ct)
	}
	ls := lines(t, resp)
	var statuses []string
	for _, l := range ls {
		statuses = append(statuses, l.Status)
	}
	if ls[0].Status != "pulling manifest" || ls[len(ls)-1].Status != "success" || len(ls) < 5 {
		t.Fatalf("%v", statuses)
	}
	// The last progress line is the whole file, tied to its digest.
	var sawDone bool
	for _, l := range ls {
		if l.Total == 1000 && l.Completed == 1000 {
			sawDone = true
			if l.Digest != "sha256:"+modelDigest || l.Status != "pulling "+modelDigest[:12] {
				t.Errorf("%+v", l)
			}
		}
	}
	if !sawDone || statuses[len(statuses)-3] != "verifying sha256 digest" || statuses[len(statuses)-2] != "writing manifest" {
		t.Errorf("%v", statuses)
	}
	if len(e.admin.downloads) != 1 || e.admin.downloads[0] != modelID {
		t.Errorf("downloads %v", e.admin.downloads)
	}

	l := decodeJSON[line](t, e.post(t, "/api/pull", `{"model":"gemma4","stream":false}`))
	if l.Status != "success" {
		t.Errorf("%+v", l)
	}
	// Older clients say "name".
	if ls := lines(t, e.post(t, "/api/pull", `{"name":"gemma4"}`)); ls[len(ls)-1].Status != "success" {
		t.Errorf("%+v", ls)
	}

	wantError(t, e.post(t, "/api/pull", `{"model":"llama3.2"}`), 404, "'llama3.2' not found")

	e.admin.mu.Lock()
	e.admin.dlErr = core.Errorf(core.CodeModelUnavailable, "download failed: disk full")
	e.admin.mu.Unlock()
	wantError(t, e.post(t, "/api/pull", `{"model":"gemma4","stream":false}`), 503, "disk full")
	ls = lines(t, e.post(t, "/api/pull", `{"model":"gemma4"}`))
	if last := ls[len(ls)-1]; last.Error != "download failed: disk full" {
		t.Errorf("%+v", ls)
	}
}

// The model is loaded by an empty request, and unloaded by keep_alive 0, as
// Ollama's clients do it.
func TestPreloadAndUnload(t *testing.T) {
	e := newEnv(t, setup{})
	running := func() int {
		return len(decodeJSON[struct{ Models []json.RawMessage }](t, e.get(t, "/api/ps")).Models)
	}
	for _, tc := range []struct{ path, body string }{
		{"/api/chat", `{"model":"gemma4","messages":[]}`},
		{"/api/generate", `{"model":"gemma4"}`},
		{"/api/generate", `{"model":"gemma4","prompt":"","stream":false}`},
	} {
		if running() != 0 {
			t.Fatal("the model is loaded")
		}
		resp := e.post(t, tc.path, tc.body)
		wantStatus(t, resp, 200)
		ls := lines(t, resp)
		if len(ls) != 1 || !ls[0].Done || ls[0].DoneReason != "load" || ls[0].Message.Content != "" || ls[0].Model != "gemma4" {
			t.Errorf("%s %s: %+v", tc.path, tc.body, ls)
		}
		if running() != 1 {
			t.Errorf("%s %s did not load the model", tc.path, tc.body)
		}
		resp = e.post(t, tc.path, strings.Replace(tc.body, `"model":"gemma4"`, `"model":"gemma4","keep_alive":0`, 1))
		wantStatus(t, resp, 200)
		if ls := lines(t, resp); len(ls) != 1 || ls[0].DoneReason != "unload" {
			t.Errorf("%+v", ls)
		}
		if running() != 0 {
			t.Errorf("%s did not unload the model", tc.path)
		}
	}
	// Unloading what is not loaded is not an error.
	wantStatus(t, e.post(t, "/api/chat", `{"model":"gemma4","messages":[],"keep_alive":0}`), 200)
	wantError(t, e.post(t, "/api/chat", `{"model":"nope","messages":[]}`), 404, "not found")
}

func TestKeepAlive(t *testing.T) {
	e := newEnv(t, setup{})
	ask := func(keep string) {
		t.Helper()
		wantStatus(t, e.post(t, "/api/chat", `{"model":"gemma4","stream":false,"keep_alive":`+keep+`,`+msgs("x")+`}`), 200)
	}
	pinned := func() bool {
		e.admin.mu.Lock()
		defer e.admin.mu.Unlock()
		return e.admin.pins[modelID]
	}
	for _, keep := range []string{`"5m"`, `300`, `"24h"`, `"10"`, `null`} {
		ask(keep)
		if pinned() || e.admin.unloads != 0 {
			t.Fatalf("keep_alive %s changed the model's residency", keep)
		}
	}
	for _, keep := range []string{`-1`, `"-1"`, `"-1m"`} {
		e.admin.mu.Lock()
		e.admin.pins[modelID] = false
		e.admin.mu.Unlock()
		ask(keep)
		if !pinned() {
			t.Errorf("keep_alive %s did not pin the model", keep)
		}
	}
	// 0 unloads the model once it has answered.
	for _, keep := range []string{`0`, `"0"`, `"0s"`} {
		ask(keep)
		waitFor(t, "the model to be unloaded after keep_alive "+keep, func() bool {
			return len(e.admin.p.Status()) == 0
		})
	}
	// A stream is unloaded after its last line.
	resp := e.post(t, "/api/generate", `{"model":"gemma4","keep_alive":0,"prompt":"x"}`)
	lines(t, resp)
	waitFor(t, "the model to be unloaded after a stream", func() bool { return len(e.admin.p.Status()) == 0 })
}

func TestOnlyAnAdminManagesTheModel(t *testing.T) {
	e := newEnv(t, setup{keys: []httpx.KeySpec{{Name: "admin", Secret: "sk-admin", Admin: true}, {Name: "user", Secret: "sk-user"}}})
	post := func(token, path, body string) *http.Response { return e.do(t, http.MethodPost, path, body, token) }

	// Anyone may chat and warm the model up, and a keep_alive on a chat is a
	// hint that costs nobody else anything.
	wantStatus(t, post("sk-user", "/api/chat", `{"model":"gemma4","stream":false,"keep_alive":-1,`+msgs("x")+`}`), 200)
	if e.admin.pins[modelID] {
		t.Error("a user key pinned the model")
	}
	wantStatus(t, post("sk-user", "/api/chat", `{"model":"gemma4","messages":[]}`), 200)
	wantStatus(t, post("sk-user", "/api/chat", `{"model":"gemma4","stream":false,"keep_alive":0,`+msgs("x")+`}`), 200)
	if len(e.admin.p.Status()) != 1 {
		t.Error("a user key unloaded the model")
	}
	// What would change the server's state is refused.
	wantError(t, post("sk-user", "/api/chat", `{"model":"gemma4","messages":[],"keep_alive":0}`), 403, "administrative key")
	wantError(t, post("sk-user", "/api/pull", `{"model":"gemma4"}`), 403, "administrative key")
	if len(e.admin.downloads) != 0 {
		t.Error("a user key started a download")
	}

	wantStatus(t, post("sk-admin", "/api/chat", `{"model":"gemma4","stream":false,"keep_alive":-1,`+msgs("x")+`}`), 200)
	if !e.admin.pins[modelID] {
		t.Error("an admin key could not pin")
	}
	wantStatus(t, post("sk-admin", "/api/pull", `{"model":"gemma4","stream":false}`), 200)
	wantStatus(t, post("sk-admin", "/api/chat", `{"model":"gemma4","messages":[],"keep_alive":0}`), 200)
}

func TestAuth(t *testing.T) {
	e := newEnv(t, setup{keys: []httpx.KeySpec{{Name: "user", Secret: "sk-user"}}})
	for _, tc := range []struct {
		method, path, body, token string
		status                    int
	}{
		{"GET", "/", "", "", 401},
		{"GET", "/api/version", "", "", 401},
		{"GET", "/api/tags", "", "sk-wrong", 401},
		{"POST", "/api/chat", `{"model":"gemma4","stream":false,"messages":[{"role":"user","content":"x"}]}`, "", 401},
		{"POST", "/api/embed", `{}`, "", 401},
		{"GET", "/", "", "sk-user", 200},
		{"GET", "/api/version", "", "sk-user", 200},
		{"GET", "/api/tags", "", "sk-user", 200},
		{"POST", "/api/chat", `{"model":"gemma4","stream":false,"messages":[{"role":"user","content":"x"}]}`, "sk-user", 200},
		{"POST", "/api/embed", `{}`, "sk-user", 501},
	} {
		resp := e.do(t, tc.method, tc.path, tc.body, tc.token)
		if resp.StatusCode != tc.status {
			t.Errorf("%s %s with %q: %d, want %d", tc.method, tc.path, tc.token, resp.StatusCode, tc.status)
		}
		if tc.status == 401 && resp.Header.Get("WWW-Authenticate") == "" {
			t.Errorf("%s %s: no WWW-Authenticate", tc.method, tc.path)
		}
	}
}

func TestNotImplemented(t *testing.T) {
	e := newEnv(t, setup{})
	for _, tc := range []struct{ method, path, err string }{
		{"POST", "/api/embed", "embeddings are not available"},
		{"POST", "/api/embeddings", "embeddings are not available"},
		{"POST", "/api/create", "creating models"},
		{"POST", "/api/copy", "copying models"},
		{"POST", "/api/push", "pushing models"},
		{"POST", "/api/blobs/sha256:abc", "blobs"},
		{"DELETE", "/api/delete", "nanollm models rm"},
	} {
		resp := e.do(t, tc.method, tc.path, `{"model":"gemma4","input":"x"}`, "")
		wantError(t, resp, 501, tc.err)
	}
	if resp := e.do(t, http.MethodHead, "/api/blobs/sha256:abc", "", ""); resp.StatusCode != 501 {
		t.Errorf("HEAD /api/blobs: %d", resp.StatusCode)
	}
	// One chat model: the message says so.
	resp := e.post(t, "/api/embed", `{}`)
	wantError(t, resp, 501, "one chat model")
}

func TestMountWithoutAService(t *testing.T) {
	mux := http.NewServeMux()
	if err := (&Adapter{}).Mount(mux, adapter.Deps{Catalog: testCatalog}); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(mux)
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/api/chat", "application/json", strings.NewReader(`{"model":"gemma4",`+msgs("x")+`}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	wantError(t, resp, 503, "no backend")
}

func TestRegistered(t *testing.T) {
	found := false
	for _, n := range adapter.Available() {
		found = found || n == "ollama"
	}
	if !found {
		t.Fatalf("the dialect is not registered: %v", adapter.Available())
	}
}

func TestParseKeepAlive(t *testing.T) {
	for _, tc := range []struct {
		in      string
		want    keepAlive
		wantErr bool
	}{
		{in: ``}, {in: `null`}, {in: `300`}, {in: `"5m"`}, {in: `"1h30m"`}, {in: `1.5`},
		{in: `-1`, want: keepAlive{Forever: true}}, {in: `"-1"`, want: keepAlive{Forever: true}}, {in: `"-5m"`, want: keepAlive{Forever: true}},
		{in: `0`, want: keepAlive{Unload: true}}, {in: `"0"`, want: keepAlive{Unload: true}}, {in: `"0s"`, want: keepAlive{Unload: true}},
		{in: `"soon"`, wantErr: true}, {in: `true`, wantErr: true}, {in: `[1]`, wantErr: true},
	} {
		got, err := parseKeepAlive(json.RawMessage(tc.in))
		if (err != nil) != tc.wantErr || got != tc.want {
			t.Errorf("%s: %+v, %v", tc.in, got, err)
		}
	}
}
