//go:build !noui

package ui

import (
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strings"
	"testing"
	"testing/fstest"
)

// TestHasSPA: a real Vite build has hashed bundles under dist/assets; the
// committed placeholder is one index.html and nothing else, and must not
// count as a UI.
func TestHasSPA(t *testing.T) {
	placeholder := fstest.MapFS{"dist/index.html": {Data: []byte("<p>not built</p>")}}
	if hasSPA(placeholder) {
		t.Fatal("the placeholder alone counts as a UI")
	}
	emptyAssets := fstest.MapFS{
		"dist/index.html": {Data: []byte("<p>not built</p>")},
		"dist/assets":     {Mode: fs.ModeDir},
	}
	if hasSPA(emptyAssets) {
		t.Fatal("an empty assets directory counts as a UI")
	}
	built := fstest.MapFS{
		"dist/index.html":          {Data: []byte(`<script src="/ui/assets/index-abc.js"></script>`)},
		"dist/assets/index-abc.js": {Data: []byte("console.log(1)")},
	}
	if !hasSPA(built) {
		t.Fatal("a build with assets does not count as a UI")
	}
}

var assetRef = regexp.MustCompile(`(?:src|href)="/ui/(assets/[^"]+)"`)

// TestEmbeddedSPA checks what this binary actually embeds, when it embeds a
// build: index.html exists, every asset it references is there, and the
// handler serves the page, falls back to it for client routes and refuses a
// missing asset. A build without `make web` skips, unless
// NANOLLM_REQUIRE_UI is set, as CI's e2e job does after building the SPA.
func TestEmbeddedSPA(t *testing.T) {
	if !Enabled {
		if os.Getenv("NANOLLM_REQUIRE_UI") != "" {
			t.Fatal("NANOLLM_REQUIRE_UI is set but the binary embeds no SPA: run `make web` first")
		}
		t.Skip("the SPA is not built into this binary; run `make web` to test it")
	}

	index, err := fs.ReadFile(assets, "dist/index.html")
	if err != nil {
		t.Fatalf("no index.html in the embedded build: %v", err)
	}
	refs := assetRef.FindAllStringSubmatch(string(index), -1)
	if len(refs) == 0 {
		t.Fatalf("index.html references no /ui/assets/ bundle; was it built with base /ui/?\n%s", index)
	}
	for _, m := range refs {
		if _, err := fs.Stat(assets, "dist/"+m[1]); err != nil {
			t.Errorf("index.html references %s, which is not embedded", m[1])
		}
	}

	h, err := Handler()
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.Handle(Path, h)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	get := func(path string) (*http.Response, string) {
		t.Helper()
		res, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		body, _ := io.ReadAll(res.Body)
		return res, string(body)
	}

	for _, path := range []string{"/ui/", "/ui/models", "/ui/settings"} {
		res, body := get(path)
		if res.StatusCode != http.StatusOK || !strings.Contains(body, `<div id="root">`) {
			t.Errorf("GET %s: %d, want the SPA page", path, res.StatusCode)
		}
		if csp := res.Header.Get("Content-Security-Policy"); !strings.Contains(csp, "script-src 'self'") {
			t.Errorf("GET %s: CSP %q", path, csp)
		}
	}
	if res, _ := get("/ui/" + refs[0][1]); res.StatusCode != http.StatusOK {
		t.Errorf("GET %s: %d", refs[0][1], res.StatusCode)
	}
	if res, _ := get("/ui/assets/missing-0000.js"); res.StatusCode != http.StatusNotFound {
		t.Errorf("a missing asset answers %d, want 404 rather than the page", res.StatusCode)
	}
}
