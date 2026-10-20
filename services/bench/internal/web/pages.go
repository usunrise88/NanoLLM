package web

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io/fs"
	"net/http"
	"net/url"

	"github.com/usunrise88/nanollm/services/bench/internal/store"
	"github.com/usunrise88/nanollm/services/bench/templates"
)

type pages struct {
	t map[string]*template.Template
}

var pageNames = []string{"benchmarks", "entity", "submission", "howto", "privacy", "message"}

func funcs() template.FuncMap {
	return template.FuncMap{
		"t": T,
		"num": func(v float64) string {
			switch {
			case v == 0:
				return "—"
			case v >= 100:
				return fmt.Sprintf("%.0f", v)
			default:
				return fmt.Sprintf("%.1f", v)
			}
		},
		"stat": func(m map[string]store.Stat, key string) *store.Stat {
			if s, ok := m[key]; ok {
				return &s
			}
			return nil
		},
		"ff":     func(lang, f string) string { return T(lang, "ff."+f) },
		"mul100": func(v float64) float64 { return v * 100 },
		"status": func(lang, s string) string { return T(lang, "status."+s) },
	}
}

func loadPages() (*pages, error) {
	p := &pages{t: map[string]*template.Template{}}
	for _, name := range pageNames {
		t, err := template.New("layout.html").Funcs(funcs()).ParseFS(templates.FS, "layout.html", name+".html")
		if err != nil {
			return nil, fmt.Errorf("template %s: %w", name, err)
		}
		p.t[name] = t
	}
	return p, nil
}

// base is what every page's layout needs.
type base struct {
	Lang        string
	Section     string // results, howto or privacy: the tab that is current
	OtherLang   string
	OtherURL    string
	Title       string
	License     string
	LicenseURL  string
	PublicURL   string
	ExportQuery string
}

func (s *Server) base(w http.ResponseWriter, r *http.Request, titleKey string) base {
	l := lang(w, r)
	other := LangEN
	if l == LangEN {
		other = LangRU
	}
	q := r.URL.Query()
	q.Set("lang", other)
	section := "results"
	switch titleKey {
	case "title.howto":
		section = "howto"
	case "title.privacy":
		section = "privacy"
	}
	return base{
		Lang: l, Section: section, OtherLang: other, OtherURL: r.URL.Path + "?" + q.Encode(), Title: T(l, titleKey),
		License: s.cfg.DataLicense, LicenseURL: s.cfg.DataLicenseURL, PublicURL: s.cfg.PublicURL,
	}
}

func (s *Server) render(w http.ResponseWriter, r *http.Request, status int, name string, data any) {
	var buf bytes.Buffer
	if err := s.pages.t[name].Execute(&buf, data); err != nil {
		s.fail(w, r, err)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(status)
	_, _ = buf.WriteTo(w)
}

func (s *Server) message(w http.ResponseWriter, r *http.Request, status int, key string) {
	b := s.base(w, r, "site")
	s.render(w, r, status, "message", struct {
		base
		Message string
	}{b, T(b.Lang, key)})
}

func (s *Server) pageRoutes(mux *http.ServeMux) {
	static, _ := fs.Sub(templates.FS, "static")
	mux.Handle("GET /benchmarks/static/", http.StripPrefix("/benchmarks/static/", cacheFor(http.FileServerFS(static))))
	mux.HandleFunc("GET /benchmarks", s.benchmarks)
	mux.HandleFunc("GET /benchmarks/{$}", s.benchmarks)
	mux.HandleFunc("GET /benchmarks/systems/{slug}", s.entity("system"))
	mux.HandleFunc("GET /benchmarks/cpu/{slug}", s.entity("cpu"))
	mux.HandleFunc("GET /benchmarks/gpu/{slug}", s.entity("gpu"))
	mux.HandleFunc("GET /benchmarks/s/{id}", s.submissionPage)
	mux.HandleFunc("GET /benchmarks/how-to-run", s.static("howto", "title.howto"))
	mux.HandleFunc("GET /benchmarks/privacy", s.static("privacy", "title.privacy"))
}

func cacheFor(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "public, max-age=3600")
		h.ServeHTTP(w, r)
	})
}

type benchmarksPage struct {
	base
	Filter store.Filter
	Facets store.Facets
	Groups []store.Group
	Recent []store.SubmissionView
}

func (s *Server) benchmarks(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	f := s.filter(r)
	fc, err := store.ReadFacets(ctx, s.st.DB, f.Suite)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	groups, err := store.Results(ctx, s.st.DB, f)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	recent, err := store.PublishedViews(ctx, s.st.DB, f.Suite, 10)
	if err != nil {
		s.fail(w, r, err)
		return
	}
	b := s.base(w, r, "title.benchmarks")
	q := r.URL.Query()
	q.Del("lang")
	b.ExportQuery = q.Encode()
	s.render(w, r, http.StatusOK, "benchmarks", benchmarksPage{b, f, fc, groups, recent})
}

type entityPage struct {
	base
	Kind   string
	Name   string
	Groups []store.Group
}

func (s *Server) entity(kind string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		slug := r.PathValue("slug")
		f := s.filter(r)
		switch kind {
		case "system":
			f.System = slug
		case "cpu":
			f.CPU = slug
		case "gpu":
			f.GPU = slug
		}
		groups, err := store.Results(r.Context(), s.st.DB, f)
		if err != nil {
			s.fail(w, r, err)
			return
		}
		if len(groups) == 0 {
			s.message(w, r, http.StatusNotFound, "notfound")
			return
		}
		name := map[string]string{"system": groups[0].System, "cpu": groups[0].CPU, "gpu": groups[0].GPU}[kind]
		b := s.base(w, r, "kind."+kind)
		b.Title = name
		q := url.Values{}
		q.Set(map[string]string{"system": "system", "cpu": "cpu", "gpu": "gpu"}[kind], slug)
		q.Set("suite", f.Suite)
		b.ExportQuery = q.Encode()
		s.render(w, r, http.StatusOK, "entity", entityPage{b, kind, name, groups})
	}
}

type submissionPage struct {
	base
	ID     string
	State  string
	View   store.SubmissionView
	Raw    string
	Gone   bool
	Queued bool
}

func (s *Server) submissionPage(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	id := r.PathValue("id")
	b := s.base(w, r, "s.title")
	b.Title = T(b.Lang, "s.title") + " " + id
	sub, mod, err := store.GetSubmission(ctx, s.st.DB, id)
	if errors.Is(err, store.ErrNotFound) {
		if gone, _ := store.Withdrawn(ctx, s.st.DB, id); gone {
			s.render(w, r, http.StatusGone, "submission", submissionPage{base: b, ID: id, Gone: true})
			return
		}
		s.message(w, r, http.StatusNotFound, "notfound")
		return
	}
	if err != nil {
		s.fail(w, r, err)
		return
	}
	switch mod.State {
	case store.StateHidden:
		s.message(w, r, http.StatusNotFound, "notfound")
		return
	case store.StateQueued:
		s.render(w, r, http.StatusOK, "submission", submissionPage{base: b, ID: id, Queued: true})
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
	pretty, _ := json.MarshalIndent(raw, "", "  ")
	s.render(w, r, http.StatusOK, "submission", submissionPage{base: b, ID: id, State: mod.State, View: view, Raw: string(pretty)})
}

type staticPage struct {
	base
	RetentionDays int
}

func (s *Server) static(name, titleKey string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		b := s.base(w, r, titleKey)
		s.render(w, r, http.StatusOK, name, staticPage{b, s.cfg.BackupRetentionDays})
	}
}
