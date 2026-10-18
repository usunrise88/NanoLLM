//go:build !noui

// Package ui serves the built SPA from inside the binary at /ui/ (SPEC §11).
// There is no Node server in production: `make web` writes the Vite output of
// web/ into dist/ and it is embedded at compile time.
//
// Building with -tags noui removes the SPA entirely, for headless deployments.
package ui

import (
	"embed"
	"errors"
	"io/fs"
	"net/http"
	"strings"
)

//go:embed all:dist
var assets embed.FS

// Path is where the SPA is mounted. Fixed rather than configurable: it is an
// unauthenticated prefix, and a setting for it is one edit away from exempting
// the API.
const Path = "/ui/"

// Enabled reports whether this build contains a usable UI.
//
// Compiled-in is not the same as present. dist/ always holds a committed
// placeholder page so that `go build ./cmd/nanollm` works in a fresh checkout,
// and serving that placeholder would mount an unauthenticated path that
// answers 200 with an apology. So Enabled asks what is actually embedded.
var Enabled = hasSPA(assets)

var errNoSPA = errors.New(
	"the web UI was not built into this binary: run `make web` before building, " +
		"or build with -tags noui to drop it on purpose")

// hasSPA reports whether dist holds a real Vite build rather than the
// placeholder. The build writes hashed bundles into dist/assets; the
// placeholder is one index.html and nothing else.
func hasSPA(fsys fs.FS) bool {
	entries, err := fs.ReadDir(fsys, "dist/assets")
	return err == nil && len(entries) > 0
}

// Handler serves the SPA under Path. Unknown paths fall back to index.html so
// client-side routing works on a hard refresh; asset paths do not, so a missing
// asset is still a 404 rather than an HTML page with a JavaScript content type.
func Handler() (http.Handler, error) {
	if !Enabled {
		return nil, errNoSPA
	}
	sub, err := fs.Sub(assets, "dist")
	if err != nil {
		return nil, err
	}
	files := http.FileServer(http.FS(sub))

	return http.StripPrefix(strings.TrimSuffix(Path, "/"),
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			p := strings.TrimPrefix(r.URL.Path, "/")
			if p == "" {
				p = "index.html"
			}
			if _, err := fs.Stat(sub, p); err != nil {
				if strings.Contains(p, ".") {
					http.NotFound(w, r)
					return
				}
				r = r.Clone(r.Context())
				r.URL.Path = "/"
			}
			w.Header().Set("Content-Security-Policy",
				"default-src 'self'; img-src 'self' data: blob:; "+
					"style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'")
			files.ServeHTTP(w, r)
		})), nil
}
