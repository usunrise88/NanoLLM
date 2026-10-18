// Package native is the full service interface under /api/v1 (SPEC §6): chat,
// completion, tokenization, batch jobs, and model and device management.
//
// Request bodies are the core types themselves (core.ChatRequest and the
// rest), so every native-only field — grammar, session_id, device, fallback,
// enable_thinking, max_reasoning_tokens, cache_prompt — is simply a field.
// "stream": true answers with server-sent events: "delta" events carrying a
// core.Delta, then one "done" event with the full result, or an "error"
// event with a problem. Errors are application/problem+json (RFC 9457).
// Model, device and configuration management require an administrative key.
package native

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/usunrise88/nanollm/internal/api/adapter"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
)

func init() { adapter.Register(config.DialectNative, &Adapter{}) }

type Adapter struct{}

// Mounter is implemented by the parts of the server that add native routes of
// their own (the job queue), so this package does not import them.
type Mounter interface {
	MountNative(mux *http.ServeMux)
}

func (*Adapter) Mount(mux *http.ServeMux, deps adapter.Deps) error {
	h := &handler{deps: deps}
	admin := httpx.RequireAdmin

	mux.HandleFunc("POST /api/v1/chat", h.chat)
	mux.HandleFunc("POST /api/v1/complete", h.complete)
	mux.HandleFunc("POST /api/v1/tokenize", h.tokenize)
	mux.HandleFunc("POST /api/v1/detokenize", h.detokenize)

	if m, ok := deps.Extra.(Mounter); ok {
		m.MountNative(mux)
	} else {
		// queue.enabled is false, or the queue could not open its database.
		off := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			httpx.WriteProblem(w, core.Errorf(core.CodeCapabilityUnavailable,
				"the job queue is off on this server: queue.enabled is false or its database could not be opened (see the server log)"))
		})
		for _, route := range []string{"POST /api/v1/jobs", "GET /api/v1/jobs", "GET /api/v1/jobs/{id}", "GET /api/v1/jobs/{id}/events", "DELETE /api/v1/jobs/{id}"} {
			mux.Handle(route, off)
		}
	}

	mux.HandleFunc("GET /api/v1/devices", h.devices)
	mux.HandleFunc("GET /api/v1/models", h.models)
	mux.HandleFunc("GET /api/v1/catalog", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]any{"models": deps.Catalog})
	})
	mux.Handle("POST /api/v1/models/{id}/download", admin(http.HandlerFunc(h.download)))
	mux.Handle("POST /api/v1/models/{id}/load", admin(http.HandlerFunc(h.load)))
	mux.Handle("POST /api/v1/models/{id}/unload", admin(http.HandlerFunc(h.unload)))
	mux.Handle("POST /api/v1/models/{id}/pin", admin(http.HandlerFunc(h.pin)))
	mux.Handle("POST /api/v1/models/{id}/reload", admin(http.HandlerFunc(h.reload)))
	mux.Handle("GET /api/v1/config", admin(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, deps.ConfigSnapshot())
	})))
	return nil
}

type handler struct{ deps adapter.Deps }

func (h *handler) service(w http.ResponseWriter) (core.Service, bool) {
	if h.deps.Service == nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeModelUnavailable, "no backend can serve models in this configuration"))
		return nil, false
	}
	return h.deps.Service, true
}

func (h *handler) adminSvc(w http.ResponseWriter) (core.Admin, bool) {
	if h.deps.Admin == nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeModelUnavailable, "model management is not available in this configuration"))
		return nil, false
	}
	return h.deps.Admin, true
}

// Decode reads a JSON body strictly: an unknown field in the native dialect
// is a typo, and silently ignoring session_idd would cost someone an
// afternoon.
func Decode(r *http.Request, v any) error {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		var mbe *http.MaxBytesError
		switch {
		case errors.As(err, &mbe):
			return core.Errorf(core.CodeInvalidRequest, "the request body is larger than %d bytes", mbe.Limit)
		case errors.Is(err, io.EOF):
			return core.Errorf(core.CodeInvalidRequest, "the request body is empty")
		}
		return core.Errorf(core.CodeInvalidRequest, "the request body is not valid: %v", err)
	}
	return nil
}

type chatBody struct {
	core.ChatRequest
	Stream bool `json:"stream"`
}

type completeBody struct {
	core.CompletionRequest
	Stream bool `json:"stream"`
}

func (h *handler) chat(w http.ResponseWriter, r *http.Request) {
	var body chatBody
	if err := Decode(r, &body); err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	svc, ok := h.service(w)
	if !ok {
		return
	}
	if !body.Stream {
		res, err := svc.Chat(r.Context(), body.ChatRequest, nil)
		if err != nil {
			httpx.WriteProblem(w, err)
			return
		}
		httpx.WriteJSON(w, http.StatusOK, res)
		return
	}
	stream(w, r.Context(), func(ctx context.Context, sink core.Sink) (any, error) {
		return svc.Chat(ctx, body.ChatRequest, sink)
	})
}

func (h *handler) complete(w http.ResponseWriter, r *http.Request) {
	var body completeBody
	if err := Decode(r, &body); err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	svc, ok := h.service(w)
	if !ok {
		return
	}
	if !body.Stream {
		res, err := svc.Complete(r.Context(), body.CompletionRequest, nil)
		if err != nil {
			httpx.WriteProblem(w, err)
			return
		}
		httpx.WriteJSON(w, http.StatusOK, res)
		return
	}
	stream(w, r.Context(), func(ctx context.Context, sink core.Sink) (any, error) {
		return svc.Complete(ctx, body.CompletionRequest, sink)
	})
}

// stream runs a generation as server-sent events. Nothing is written until
// the first delta (or a heartbeat during a long prefill), so an early failure
// is still an ordinary problem response with its status.
func stream(w http.ResponseWriter, ctx context.Context, gen func(context.Context, core.Sink) (any, error)) {
	type result struct {
		v   any
		err error
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	deltas := make(chan core.Delta, 64)
	done := make(chan result, 1)
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

	var st *httpx.Stream
	open := func() bool {
		if st != nil {
			return true
		}
		var err error
		if st, err = httpx.NewStream(w); err != nil {
			httpx.WriteProblem(w, err)
			return false
		}
		return true
	}
	defer func() {
		if st != nil {
			st.Close()
		}
	}()
	early := time.NewTimer(15 * time.Second)
	defer early.Stop()
	var heartbeat <-chan time.Time
	for {
		select {
		case d, ok := <-deltas:
			if !ok {
				res := <-done
				if res.err != nil {
					if st == nil {
						httpx.WriteProblem(w, res.err)
						return
					}
					if !errors.Is(res.err, context.Canceled) {
						_ = st.Send("", "error", httpx.ProblemOf(res.err))
					}
					return
				}
				if open() {
					_ = st.Send("", "done", res.v)
				}
				return
			}
			if !open() {
				cancel()
				<-done
				return
			}
			heartbeat = st.Heartbeat()
			if err := st.Send("", "delta", d); err != nil {
				cancel()
				<-done
				return
			}
		case <-early.C:
			if !open() {
				cancel()
				<-done
				return
			}
			heartbeat = st.Heartbeat()
		case <-heartbeat:
			if err := st.Comment("keep-alive"); err != nil {
				cancel()
				<-done
				return
			}
		}
	}
}

func (h *handler) tokenize(w http.ResponseWriter, r *http.Request) {
	var req core.TokenizeRequest
	if err := Decode(r, &req); err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	svc, ok := h.service(w)
	if !ok {
		return
	}
	res, err := svc.Tokenize(r.Context(), req)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, res)
}

func (h *handler) detokenize(w http.ResponseWriter, r *http.Request) {
	var req core.DetokenizeRequest
	if err := Decode(r, &req); err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	svc, ok := h.service(w)
	if !ok {
		return
	}
	res, err := svc.Detokenize(r.Context(), req)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, res)
}

func (h *handler) devices(w http.ResponseWriter, r *http.Request) {
	a, ok := h.adminSvc(w)
	if !ok {
		return
	}
	devs, err := a.Devices(r.Context())
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]any{"devices": devs})
}

func (h *handler) models(w http.ResponseWriter, r *http.Request) {
	a, ok := h.adminSvc(w)
	if !ok {
		return
	}
	ms, err := a.Models(r.Context())
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]any{"models": ms})
}

// download streams progress as server-sent events: "progress" while bytes
// arrive, then "done" or "error". A second call while the download runs
// joins it; a model already on disk answers "done" at once.
func (h *handler) download(w http.ResponseWriter, r *http.Request) {
	a, ok := h.adminSvc(w)
	if !ok {
		return
	}
	st, err := httpx.NewStream(w)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	defer st.Close()
	id := r.PathValue("id")
	progress := make(chan core.DownloadProgress, 16)
	done := make(chan error, 1)
	go func() {
		done <- a.Download(r.Context(), id, func(p core.DownloadProgress) {
			select {
			case progress <- p:
			default: // a slow reader misses a percentage, nothing more
			}
		})
	}()
	for {
		select {
		case p := <-progress:
			if st.Send("", "progress", p) != nil {
				return
			}
		case err := <-done:
			if err != nil {
				_ = st.Send("", "error", httpx.ProblemOf(err))
				return
			}
			_ = st.Send("", "done", map[string]string{"id": id})
			return
		case <-st.Heartbeat():
			if st.Comment("keep-alive") != nil {
				return
			}
		}
	}
}

func (h *handler) load(w http.ResponseWriter, r *http.Request) {
	a, ok := h.adminSvc(w)
	if !ok {
		return
	}
	info, err := a.Load(r.Context(), r.PathValue("id"))
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, info)
}

func (h *handler) unload(w http.ResponseWriter, r *http.Request) {
	a, ok := h.adminSvc(w)
	if !ok {
		return
	}
	if err := a.Unload(r.Context(), r.PathValue("id")); err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *handler) pin(w http.ResponseWriter, r *http.Request) {
	a, ok := h.adminSvc(w)
	if !ok {
		return
	}
	var body struct {
		Pin *bool `json:"pin"`
	}
	if r.ContentLength != 0 {
		if err := Decode(r, &body); err != nil {
			httpx.WriteProblem(w, err)
			return
		}
	}
	pin := body.Pin == nil || *body.Pin
	if err := a.Pin(r.Context(), r.PathValue("id"), pin); err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]any{"id": r.PathValue("id"), "pinned": pin})
}

func (h *handler) reload(w http.ResponseWriter, r *http.Request) {
	a, ok := h.adminSvc(w)
	if !ok {
		return
	}
	info, err := a.Reload(r.Context(), r.PathValue("id"))
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, info)
}
