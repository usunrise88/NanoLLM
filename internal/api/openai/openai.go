// Package openai is OpenAI Chat Completions compatibility under /v1 (SPEC §6),
// so existing clients work by changing base_url and nothing else.
//
// Divergences from the original API, documented here and to callers:
//   - n > 1 returns 400.
//   - Image and audio content parts return 400 until vision lands in v2.
//   - Responses carry extra fields OpenAI clients ignore: reasoning_content on
//     messages and deltas, stats (timings, device) and warnings.
//   - enable_thinking, top_k, min_p and repeat_penalty are accepted as
//     extensions.
//
// Errors use the OpenAI error envelope rather than problem+json: OpenAI SDKs
// parse error.message and error.type strictly, and a client that cannot read
// the error is not compatible however good the successes are.
package openai

import (
	"encoding/json"
	"net/http"

	"github.com/usunrise88/nanollm/internal/api/adapter"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
)

func init() { adapter.Register(config.DialectOpenAI, &Adapter{}) }

type Adapter struct{}

func (*Adapter) Mount(mux *http.ServeMux, _ adapter.Deps) error {
	pending := func(what string) http.HandlerFunc {
		return func(w http.ResponseWriter, _ *http.Request) {
			writeError(w, httpx.NotYet(what, "v1.0"))
		}
	}
	mux.HandleFunc("POST /v1/chat/completions", pending("chat completions"))
	mux.HandleFunc("POST /v1/completions", pending("completions"))
	mux.HandleFunc("GET /v1/models", pending("model listing"))
	mux.HandleFunc("GET /v1/models/{id}", pending("model listing"))
	mux.HandleFunc("POST /v1/embeddings", func(w http.ResponseWriter, _ *http.Request) {
		writeError(w, httpx.NotYet("embeddings", "v2"))
	})
	return nil
}

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

func writeError(w http.ResponseWriter, err error) {
	e := core.AsError(err)
	var env errorEnvelope
	env.Error.Message = e.Message
	env.Error.Type = errorType(e.Code)
	env.Error.Param = e.Param
	env.Error.Code = string(e.Code)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.Code.HTTPStatus())
	_ = json.NewEncoder(w).Encode(env)
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
