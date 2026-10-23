package httpx

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
)

func newStore(t *testing.T) *KeyStore {
	t.Helper()
	digest := sha256.Sum256([]byte("sk-hashed"))
	s, err := NewKeyStore([]KeySpec{
		{Name: "admin", Secret: "sk-admin", Admin: true},
		{Name: "user", Secret: "sha256:" + hex.EncodeToString(digest[:])},
	})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func serve(h http.Handler, path, token string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodGet, path, nil)
	if token != "" {
		r.Header.Set("Authorization", "Bearer "+token)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}

func TestAuth(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {})
	h := Chain(ok, Auth(newStore(t), "/healthz", "/ui/"))

	for _, c := range []struct {
		path, token string
		want        int
	}{
		{"/healthz", "", 200},
		{"/ui/assets/app.js", "", 200},
		{"/v1/models", "", 401},
		{"/v1/models", "sk-wrong", 401},
		{"/v1/models", "sk-admin", 200},
		{"/v1/models", "sk-hashed", 200}, // configured as a digest
		{"/healthz/x", "", 401},          // exact entries are not prefixes
	} {
		if got := serve(h, c.path, c.token).Code; got != c.want {
			t.Errorf("%s with %q: got %d, want %d", c.path, c.token, got, c.want)
		}
	}
}

func TestUnauthorizedIsAProblem(t *testing.T) {
	h := Chain(http.NotFoundHandler(), Auth(newStore(t)))
	w := serve(h, "/api/v1/chat", "")
	if ct := w.Header().Get("Content-Type"); ct != "application/problem+json" {
		t.Errorf("content type %q", ct)
	}
	var p Problem
	if err := json.NewDecoder(w.Body).Decode(&p); err != nil {
		t.Fatal(err)
	}
	if p.Status != 401 || p.Code != "unauthorized" {
		t.Errorf("problem: %+v", p)
	}
}

func TestRequireAdmin(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {})
	h := Chain(RequireAdmin(ok), Auth(newStore(t)))
	if got := serve(h, "/api/v1/config", "sk-hashed").Code; got != http.StatusForbidden {
		t.Errorf("user key: got %d", got)
	}
	if got := serve(h, "/api/v1/config", "sk-admin").Code; got != http.StatusOK {
		t.Errorf("admin key: got %d", got)
	}
	// Open mode: no key in the context, nothing to refuse.
	if got := serve(RequireAdmin(ok), "/api/v1/config", "").Code; got != http.StatusOK {
		t.Errorf("open mode: got %d", got)
	}
}

func TestBadDigestIsRefused(t *testing.T) {
	if _, err := NewKeyStore([]KeySpec{{Secret: "sha256:abc"}}); err == nil {
		t.Fatal("short digest accepted")
	}
}

func TestWriteProblemSetsRetryAfter(t *testing.T) {
	rec := httptest.NewRecorder()
	WriteProblem(rec, core.Errorf(core.CodeModelUnavailable, "being measured").WithRetryAfter(7*time.Second+200*time.Millisecond))
	if rec.Code != http.StatusServiceUnavailable || rec.Header().Get("Retry-After") != "8" {
		t.Fatalf("%d, Retry-After %q", rec.Code, rec.Header().Get("Retry-After"))
	}
	rec = httptest.NewRecorder()
	WriteProblem(rec, core.Errorf(core.CodeModelUnavailable, "loading"))
	if rec.Header().Get("Retry-After") != "" {
		t.Fatalf("Retry-After %q without a time to wait", rec.Header().Get("Retry-After"))
	}
	rec = httptest.NewRecorder()
	WriteProblem(rec, core.Errorf(core.CodeConflict, "a run is in progress"))
	if rec.Code != http.StatusConflict {
		t.Fatalf("conflict is %d", rec.Code)
	}
}
