// Package native is the full service interface under /api/v1 (SPEC §6): chat,
// completion, tokenization, batch jobs, and model and device management.
//
// Native-only request fields: grammar, session_id, device, fallback,
// enable_thinking, max_reasoning_tokens, cache_prompt, webhook_url. Errors are
// application/problem+json (RFC 9457). Model, device and configuration
// management require an administrative key.
//
// Every route is mounted now so the surface is visible; the ones that need the
// v1.0 core answer 501 with the milestone they arrive in.
package native

import (
	"net/http"

	"github.com/usunrise88/nanollm/internal/api/adapter"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/httpx"
)

func init() { adapter.Register(config.DialectNative, &Adapter{}) }

type Adapter struct{}

func (*Adapter) Mount(mux *http.ServeMux, deps adapter.Deps) error {
	pending := func(what string) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			httpx.WriteProblem(w, httpx.NotYet(what, "v1.0"))
		})
	}
	admin := httpx.RequireAdmin

	mux.Handle("POST /api/v1/chat", pending("chat generation"))
	mux.Handle("POST /api/v1/complete", pending("raw completion"))
	mux.Handle("POST /api/v1/tokenize", pending("tokenization"))
	mux.Handle("POST /api/v1/detokenize", pending("detokenization"))

	mux.Handle("POST /api/v1/jobs", pending("the job queue"))
	mux.Handle("GET /api/v1/jobs", pending("the job queue"))
	mux.Handle("GET /api/v1/jobs/{id}", pending("the job queue"))
	mux.Handle("GET /api/v1/jobs/{id}/events", pending("the job queue"))
	mux.Handle("DELETE /api/v1/jobs/{id}", pending("the job queue"))

	mux.Handle("GET /api/v1/devices", pending("device listing"))
	mux.Handle("GET /api/v1/models", pending("model listing"))
	mux.HandleFunc("GET /api/v1/catalog", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]any{"models": deps.Catalog})
	})
	for _, verb := range []string{"download", "load", "unload", "pin", "reload"} {
		mux.Handle("POST /api/v1/models/{id}/"+verb, admin(pending("model "+verb)))
	}
	mux.Handle("GET /api/v1/config", admin(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, deps.ConfigSnapshot())
	})))
	return nil
}
