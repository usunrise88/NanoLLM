// Package admin is /admin: the moderation queue, hide and show, the alias
// table, the vendor-verified badge and reindex, behind a bearer token. The
// page itself is static; its script asks for the token, keeps it in the
// tab's sessionStorage and sends it with every API call.
package admin

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/usunrise88/nanollm/services/bench/internal/ingest"
	"github.com/usunrise88/nanollm/services/bench/internal/normalize"
	"github.com/usunrise88/nanollm/services/bench/internal/store"
	"github.com/usunrise88/nanollm/services/bench/templates"
)

// Admin serves the admin routes.
type Admin struct {
	token string
	st    *store.Store
	ing   *ingest.Service
	log   *slog.Logger
}

// New makes the admin. An empty token disables /admin entirely.
func New(token string, st *store.Store, ing *ingest.Service, log *slog.Logger) *Admin {
	if log == nil {
		log = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	return &Admin{token: token, st: st, ing: ing, log: log}
}

// Routes registers the admin routes.
func (a *Admin) Routes(mux *http.ServeMux) {
	if a.token == "" {
		return
	}
	mux.HandleFunc("GET /admin", a.page)
	mux.HandleFunc("GET /admin/{$}", a.page)
	mux.HandleFunc("GET /admin/api/queue", a.auth(a.queue))
	mux.HandleFunc("GET /admin/api/submissions/{id}", a.auth(a.submission))
	mux.HandleFunc("POST /admin/api/submissions/{id}/state", a.auth(a.setState))
	mux.HandleFunc("GET /admin/api/aliases", a.auth(a.aliases))
	mux.HandleFunc("POST /admin/api/aliases", a.auth(a.setAlias))
	mux.HandleFunc("DELETE /admin/api/aliases", a.auth(a.deleteAlias))
	mux.HandleFunc("GET /admin/api/verified", a.auth(a.verified))
	mux.HandleFunc("POST /admin/api/verified", a.auth(a.setVerified))
	mux.HandleFunc("DELETE /admin/api/verified", a.auth(a.deleteVerified))
	mux.HandleFunc("POST /admin/api/reindex", a.auth(a.reindex))
}

func (a *Admin) auth(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		got, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || subtle.ConstantTimeCompare([]byte(got), []byte(a.token)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="benchd admin"`)
			writeErr(w, http.StatusUnauthorized, "a bearer token is required")
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		h(w, r)
	}
}

func (a *Admin) page(w http.ResponseWriter, r *http.Request) {
	b, err := templates.FS.ReadFile("admin.html")
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "no admin page")
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(b)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func (a *Admin) fail(w http.ResponseWriter, err error) {
	var re *ingest.Error
	switch {
	case errors.As(err, &re):
		writeErr(w, re.Status, re.Msg)
	case errors.Is(err, store.ErrNotFound):
		writeErr(w, http.StatusNotFound, "not found")
	default:
		a.log.Error("admin request failed", "err", err)
		writeErr(w, http.StatusInternalServerError, err.Error())
	}
}

func decode(r *http.Request, v any) error {
	return json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(v)
}

func (a *Admin) queue(w http.ResponseWriter, r *http.Request) {
	state := r.URL.Query().Get("state")
	if state == "" {
		state = store.StateQueued
	}
	items, err := store.ListModeration(r.Context(), a.st.DB, state, 500)
	if err != nil {
		a.fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"state": state, "items": items})
}

// submission is the full record, install key included: the admin needs it
// to grant the vendor-verified badge.
func (a *Admin) submission(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	id := r.PathValue("id")
	sub, mod, err := store.GetSubmission(ctx, a.st.DB, id)
	if err != nil {
		a.fail(w, err)
		return
	}
	view, err := store.GetView(ctx, a.st.DB, id)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		a.fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "received_on": sub.ReceivedOn, "install_id": sub.InstallID, "moderation": mod,
		"view": view, "raw": json.RawMessage(sub.Raw),
	})
}

func (a *Admin) setState(w http.ResponseWriter, r *http.Request) {
	var req struct {
		State  string `json:"state"`
		Reason string `json:"reason"`
	}
	if err := decode(r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "want {\"state\", \"reason\"}")
		return
	}
	if err := a.ing.Moderate(r.Context(), r.PathValue("id"), req.State, req.Reason); err != nil {
		a.fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"id": r.PathValue("id"), "state": req.State})
}

func (a *Admin) aliases(w http.ResponseWriter, r *http.Request) {
	as, err := store.ListAliases(r.Context(), a.st.DB)
	if err != nil {
		a.fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"aliases": as})
}

// reindexed applies an input change of the normalizer: the derived tables
// are rebuilt, never patched.
func (a *Admin) reindexed(ctx context.Context, w http.ResponseWriter, change func() error) {
	if err := change(); err != nil {
		a.fail(w, err)
		return
	}
	n, err := a.ing.Reindex(ctx)
	if err != nil {
		a.fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"reindexed": n})
}

func validKind(k string) bool {
	switch k {
	case normalize.KindCPU, normalize.KindGPU, normalize.KindVendor, normalize.KindSystem:
		return true
	}
	return false
}

func (a *Admin) setAlias(w http.ResponseWriter, r *http.Request) {
	var al store.Alias
	if err := decode(r, &al); err != nil || !validKind(al.Kind) || strings.TrimSpace(al.Pattern) == "" || strings.TrimSpace(al.Name) == "" {
		writeErr(w, http.StatusBadRequest, "want {\"kind\": cpu|gpu|vendor|system, \"pattern\", \"name\"}")
		return
	}
	a.reindexed(r.Context(), w, func() error { return store.SetAlias(r.Context(), a.st.DB, al) })
}

func (a *Admin) deleteAlias(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	a.reindexed(r.Context(), w, func() error { return store.DeleteAlias(r.Context(), a.st.DB, q.Get("kind"), q.Get("pattern")) })
}

func (a *Admin) verified(w http.ResponseWriter, r *http.Request) {
	vs, err := store.ListVerified(r.Context(), a.st.DB)
	if err != nil {
		a.fail(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"verified": vs})
}

// setVerified grants the badge to an install, named by its key or by one
// of its submissions.
func (a *Admin) setVerified(w http.ResponseWriter, r *http.Request) {
	var req struct {
		InstallID    string `json:"install_id"`
		SubmissionID string `json:"submission_id"`
		Label        string `json:"label"`
	}
	if err := decode(r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "want {\"install_id\" or \"submission_id\", \"label\"}")
		return
	}
	if req.InstallID == "" && req.SubmissionID != "" {
		sub, _, err := store.GetSubmission(r.Context(), a.st.DB, req.SubmissionID)
		if err != nil {
			a.fail(w, err)
			return
		}
		req.InstallID = sub.InstallID
	}
	if req.InstallID == "" {
		writeErr(w, http.StatusBadRequest, "install_id or submission_id is required")
		return
	}
	a.reindexed(r.Context(), w, func() error {
		return store.SetVerified(r.Context(), a.st.DB, store.Verified{InstallID: req.InstallID, Label: req.Label})
	})
}

func (a *Admin) deleteVerified(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("install_id")
	a.reindexed(r.Context(), w, func() error { return store.DeleteVerified(r.Context(), a.st.DB, id) })
}

func (a *Admin) reindex(w http.ResponseWriter, r *http.Request) {
	a.reindexed(r.Context(), w, func() error { return nil })
}
