// Package ollama is Ollama API compatibility (SPEC §6, v2), so stock Ollama
// clients — the ollama Python and JavaScript packages, Open WebUI, anything that
// speaks to localhost:11434 — work against a NanoLLM server by changing the host.
//
// Routes: GET /, /api/version, /api/tags, /api/ps, POST /api/show, /api/chat,
// /api/generate and /api/pull. Streaming is newline-delimited JSON, and unlike
// OpenAI's it is the default ("stream": false opts out). Errors are
// {"error": "message"} with the status the cause deserves.
//
// The dialect is opt-in: api.dialects must name it. Clients that hardcode port
// 11434 need server.addr set to match (127.0.0.1:11434); a second listener is
// not part of the dialect.
//
// Divergences from Ollama, documented here and to callers:
//   - The server serves one chat model. Embeddings (/api/embed, /api/embeddings)
//     and the model-building routes (/api/create, /api/copy, /api/push,
//     /api/blobs, DELETE /api/delete) answer 501. Delete a model with
//     `nanollm models rm`.
//   - Image input, "raw", "template", "suffix", "context" and logprobs answer
//     400: the prompt always goes through the model's own chat template, and the
//     model has no vision input yet.
//   - Sampling that a request leaves out comes from the model card, not from
//     Ollama's defaults (temperature 0.8). options.num_ctx is accepted and
//     ignored: the context is what the placement plan leaves, and a prompt that
//     does not fit gets a 400. Options this server does not apply are listed
//     in a "warnings" extension on the final line, which clients ignore.
//   - think takes true or false; Ollama's "low", "medium" and "high" mean true.
//     The thinking text arrives in message.thinking (response.thinking for
//     /api/generate).
//   - keep_alive is advisory. A negative value pins the model in memory and 0
//     unloads it after the answer, both only for an administrative key (or in
//     open mode); any other value is accepted and ignored, as the model stays
//     resident until another needs the memory. An empty chat or generate
//     request loads the model, as Ollama's does.
//   - A model is named by its catalog id, with an optional ":latest" tag, or by
//     one of Aliases. Responses name it the way the request did; /api/tags and
//     /api/ps use "<id>:latest".
//   - quantization_level is always Q4_0 (the one model is a QAT int4 build), and
//     load_duration is 0: the load is not a stage of the request pipeline.
//     prompt_eval_count counts the whole prompt, cached tokens included.
//   - Errors that the shared HTTP layer produces before a request reaches this
//     package (401 without a valid key, 429, 413) are problem+json rather than
//     {"error": …}; the status is what clients branch on.
package ollama

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/api/adapter"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/registry"
)

func init() { adapter.Register(config.DialectOllama, &Adapter{}) }

// CompatVersion is the Ollama release whose API this dialect follows, reported
// by GET /api/version. Clients gate features on it, so it names the behaviour
// implemented here and not NanoLLM's own build.
const CompatVersion = "0.12.6"

// Quantization is what every model reports as quantization_level: the one model
// the server carries is a QAT int4 (Q4_0) build.
const Quantization = "Q4_0"

// Aliases maps the names an Ollama user would type for the model to catalog
// ids. A key is lower case without a ":latest" tag; a name whose target is not
// in the catalog is unknown. Extend it before the server mounts the dialect.
var Aliases = map[string]string{
	"gemma4":            "gemma-4-e4b-qat",
	"gemma4:e4b":        "gemma-4-e4b-qat",
	"gemma4:e4b-qat":    "gemma-4-e4b-qat",
	"gemma4:e4b-it-qat": "gemma-4-e4b-qat",
	"gemma-4-e4b":       "gemma-4-e4b-qat",
}

// Adapter is the ollama dialect.
type Adapter struct{}

func (*Adapter) Mount(mux *http.ServeMux, deps adapter.Deps) error {
	h := &handler{deps: deps, started: time.Now().UTC()}
	// "{$}" is the root and nothing under it: a bare "/" would claim every
	// path no other route does.
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = io.WriteString(w, "Ollama is running")
	})
	mux.HandleFunc("GET /api/version", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]string{"version": CompatVersion})
	})
	mux.HandleFunc("GET /api/tags", h.tags)
	mux.HandleFunc("GET /api/ps", h.ps)
	mux.HandleFunc("POST /api/show", h.show)
	mux.HandleFunc("POST /api/chat", h.chat)
	mux.HandleFunc("POST /api/generate", h.generate)
	mux.HandleFunc("POST /api/pull", h.pull)

	const single = "this server serves one chat model; "
	for route, msg := range map[string]string{
		"POST /api/embed":          single + "embeddings are not available",
		"POST /api/embeddings":     single + "embeddings are not available",
		"POST /api/create":         single + "creating models is not supported",
		"POST /api/copy":           single + "copying models is not supported",
		"POST /api/push":           single + "pushing models is not supported",
		"POST /api/blobs/{digest}": single + "blobs are not supported",
		"HEAD /api/blobs/{digest}": single + "blobs are not supported",
		"DELETE /api/delete":       "deleting models over the API is not supported; use `nanollm models rm`",
	} {
		mux.HandleFunc(route, func(w http.ResponseWriter, _ *http.Request) {
			writeError(w, core.Errorf(core.CodeNotImplemented, "%s", msg))
		})
	}
	return nil
}

type handler struct {
	deps adapter.Deps
	// started stands in for modified_at: the catalog has no per-model
	// timestamp, and a client only compares it against what it saw before.
	started time.Time
}

func (h *handler) service(w http.ResponseWriter) (core.Service, bool) {
	if h.deps.Service == nil {
		writeError(w, core.Errorf(core.CodeModelUnavailable, "no backend can serve models in this configuration"))
		return nil, false
	}
	return h.deps.Service, true
}

// mayManage says whether the caller may change what is resident or on disk:
// an administrative key, or open mode where there is no key to check.
func (h *handler) mayManage(r *http.Request) bool {
	if h.deps.Admin == nil {
		return false
	}
	k, authed := httpx.KeyFrom(r.Context())
	return !authed || k.Admin
}

// --- model names ---

// canonical is a client's model name reduced to what Aliases and the catalog
// are keyed by.
func canonical(name string) string {
	n := strings.ToLower(strings.TrimSpace(name))
	n = strings.TrimPrefix(n, "registry.ollama.ai/")
	n = strings.TrimPrefix(n, "library/")
	return strings.TrimSuffix(n, ":latest")
}

// resolve finds the catalog entry a client's model name stands for.
func (h *handler) resolve(name string) (registry.Manifest, error) {
	if strings.TrimSpace(name) == "" {
		return registry.Manifest{}, core.Errorf(core.CodeInvalidRequest, "model is required").WithParam("model")
	}
	key := canonical(name)
	if m, ok := registry.Find(h.deps.Catalog, key); ok {
		return m, nil
	}
	if id, ok := Aliases[key]; ok {
		if m, ok := registry.Find(h.deps.Catalog, id); ok {
			return m, nil
		}
	}
	return registry.Manifest{}, core.Errorf(core.CodeModelNotFound, "model '%s' not found", name).WithParam("model")
}

// tagged is the name /api/tags and /api/ps give a model.
func tagged(id string) string { return id + ":latest" }

// digestOf is the model's identity as Ollama shows it: 64 hex digits. The
// pinned sha256 of the weights is one when the catalog has it.
func digestOf(m registry.Manifest) string {
	if f, ok := m.File(registry.RoleModel); ok && len(f.SHA256) == 64 {
		return f.SHA256
	}
	sum := sha256.Sum256([]byte(m.ID))
	return hex.EncodeToString(sum[:])
}

// --- requests ---

func decode(r *http.Request, v any) error {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		var mbe *http.MaxBytesError
		switch {
		case errors.As(err, &mbe):
			return core.Errorf(core.CodeInvalidRequest, "the request body is larger than %d bytes", mbe.Limit)
		case errors.Is(err, io.EOF):
			return core.Errorf(core.CodeInvalidRequest, "the request body is empty")
		}
		return core.Errorf(core.CodeInvalidRequest, "the request body is not valid JSON: %v", err)
	}
	return nil
}

// --- errors ---

type errorBody struct {
	Error string `json:"error"`
}

func writeError(w http.ResponseWriter, err error) {
	e := core.AsError(err)
	httpx.WriteJSON(w, e.Code.HTTPStatus(), errorBody{Error: e.Message})
}
