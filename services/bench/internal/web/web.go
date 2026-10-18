// Package web is the public face of the service: the JSON API the client
// and the rest of the site use, the CSV/JSON export, and the /benchmarks
// pages (html/template, RU/EN, a little JS, no build step).
package web

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"

	"github.com/usunrise88/nanollm/services/bench/internal/ingest"
	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

// Config is what the pages and the API need to know.
type Config struct {
	PublicURL string
	// DataLicense is the SPDX id of the open data licence (CC-BY-4.0 or
	// CC0-1.0: the owner decides) and DataLicenseURL its text.
	DataLicense    string
	DataLicenseURL string
	// DefaultSuite is the suite shown when none is asked for: results of
	// different suites are never mixed.
	DefaultSuite string
	// TrustProxy takes the client address from X-Forwarded-For (the
	// reverse proxy's last hop) instead of the connection.
	TrustProxy bool
	// BackupRetentionDays is stated in the privacy policy.
	BackupRetentionDays int
}

// LicenseURL is the text of a known licence.
func LicenseURL(spdx string) string {
	switch strings.ToUpper(spdx) {
	case "CC-BY-4.0":
		return "https://creativecommons.org/licenses/by/4.0/"
	case "CC0-1.0":
		return "https://creativecommons.org/publicdomain/zero/1.0/"
	case "ODBL-1.0":
		return "https://opendatacommons.org/licenses/odbl/1-0/"
	}
	return ""
}

// Server serves the public routes.
type Server struct {
	cfg   Config
	st    *store.Store
	ing   *ingest.Service
	perIP *ingest.Limiter
	pages *pages
	log   *slog.Logger
}

// New makes the server. perIP hashes addresses for the rate limit.
func New(cfg Config, st *store.Store, ing *ingest.Service, perIP *ingest.Limiter, log *slog.Logger) (*Server, error) {
	if cfg.DefaultSuite == "" {
		cfg.DefaultSuite = "standard/v1"
	}
	if cfg.DataLicense == "" {
		cfg.DataLicense = "CC-BY-4.0"
	}
	if cfg.DataLicenseURL == "" {
		cfg.DataLicenseURL = LicenseURL(cfg.DataLicense)
	}
	cfg.PublicURL = strings.TrimSuffix(cfg.PublicURL, "/")
	p, err := loadPages()
	if err != nil {
		return nil, err
	}
	if log == nil {
		log = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	return &Server{cfg: cfg, st: st, ing: ing, perIP: perIP, pages: p, log: log}, nil
}

// Routes registers the public routes.
func (s *Server) Routes(mux *http.ServeMux) {
	mux.HandleFunc("POST /api/v1/submissions", s.submit)
	mux.HandleFunc("DELETE /api/v1/submissions/{id}", s.withdraw)
	mux.HandleFunc("GET /api/v1/submissions/{id}", s.getSubmission)
	mux.HandleFunc("GET /api/v1/results", s.results)
	mux.HandleFunc("GET /api/v1/facets", s.facets)
	mux.HandleFunc("GET /api/v1/export", s.export)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if err := s.st.DB.PingContext(r.Context()); err != nil {
			http.Error(w, "db", http.StatusServiceUnavailable)
			return
		}
		_, _ = io.WriteString(w, "ok\n")
	})
	s.pageRoutes(mux)
}

// Headers wraps a handler with the security headers every response gets.
func Headers(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hd := w.Header()
		hd.Set("X-Content-Type-Options", "nosniff")
		hd.Set("Referrer-Policy", "same-origin")
		hd.Set("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
		h.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// fail answers an error: a refusal with its status, anything else as a
// 500 whose cause goes to the log (never with the client's address).
func (s *Server) fail(w http.ResponseWriter, r *http.Request, err error) {
	var re *ingest.Error
	if errors.As(err, &re) {
		writeErr(w, re.Status, re.Msg)
		return
	}
	if errors.Is(err, context.Canceled) {
		return
	}
	s.log.Error("request failed", "method", r.Method, "path", r.URL.Path, "err", err)
	writeErr(w, http.StatusInternalServerError, "internal error")
}

// clientIP is the sender's address, used only to key the in-memory rate
// limit through a salted hash; it is never stored or logged.
func (s *Server) clientIP(r *http.Request) string {
	if s.cfg.TrustProxy {
		if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
			parts := strings.Split(xff, ",")
			return strings.TrimSpace(parts[len(parts)-1])
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func (s *Server) submit(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, ingest.MaxBody))
	if err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeErr(w, http.StatusRequestEntityTooLarge, "a submission is at most 256 KB")
			return
		}
		writeErr(w, http.StatusBadRequest, "could not read the body")
		return
	}
	key := ""
	if s.perIP != nil {
		key = s.perIP.Key(s.clientIP(r))
	}
	rc, err := s.ing.Submit(r.Context(), body, key)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	status := http.StatusCreated
	if rc.Existing {
		status = http.StatusOK
	}
	writeJSON(w, status, rc)
}

func (s *Server) withdraw(w http.ResponseWriter, r *http.Request) {
	var req struct {
		InstallID string `json:"install_id"`
		Signature string `json:"signature"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "want {\"install_id\", \"signature\"}")
		return
	}
	id := r.PathValue("id")
	if err := s.ing.Withdraw(r.Context(), id, req.InstallID, req.Signature); err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"id": id, "withdrawn": true})
}

// publicRaw is the submitted document without install_id and signature:
// the key links an install's submissions, which the pages do not publish.
func publicRaw(raw []byte) (map[string]any, error) {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	delete(m, "install_id")
	delete(m, "signature")
	return m, nil
}

func (s *Server) getSubmission(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	ctx := r.Context()
	sub, mod, err := store.GetSubmission(ctx, s.st.DB, id)
	if errors.Is(err, store.ErrNotFound) {
		if gone, _ := store.Withdrawn(ctx, s.st.DB, id); gone {
			writeErr(w, http.StatusGone, "withdrawn")
			return
		}
		writeErr(w, http.StatusNotFound, "no such submission")
		return
	}
	if err != nil {
		s.fail(w, r, err)
		return
	}
	switch mod.State {
	case store.StateHidden:
		writeErr(w, http.StatusNotFound, "no such submission")
		return
	case store.StateQueued:
		writeJSON(w, http.StatusOK, map[string]any{"id": id, "state": mod.State})
		return
	}
	view, err := store.GetView(ctx, s.st.DB, id)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	raw, err := publicRaw(sub.Raw)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"submission": view, "result": raw, "license": s.cfg.DataLicense})
}

func (s *Server) filter(r *http.Request) store.Filter {
	q := r.URL.Query()
	f := store.Filter{
		Model: q.Get("model"), Suite: q.Get("suite"), Preset: q.Get("preset"), CPU: q.Get("cpu"), GPU: q.Get("gpu"),
		System: q.Get("system"), OS: q.Get("os"), FormFactor: q.Get("form_factor"),
	}
	if f.Suite == "" {
		f.Suite = s.cfg.DefaultSuite
	}
	return f
}

func (s *Server) results(w http.ResponseWriter, r *http.Request) {
	f := s.filter(r)
	groups, err := store.Results(r.Context(), s.st.DB, f)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	if groups == nil {
		groups = []store.Group{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"suite": f.Suite, "license": s.cfg.DataLicense, "groups": groups})
}

func (s *Server) facets(w http.ResponseWriter, r *http.Request) {
	suite := r.URL.Query().Get("suite")
	if suite == "" {
		suite = s.cfg.DefaultSuite
	}
	fc, err := store.ReadFacets(r.Context(), s.st.DB, suite)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"suite": suite, "facets": fc})
}
