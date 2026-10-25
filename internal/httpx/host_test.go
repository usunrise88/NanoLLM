package httpx

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRequireHostRefusesAForeignName(t *testing.T) {
	h := Chain(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}), RequireHost(LoopbackHosts("4242")...))

	for host, want := range map[string]int{
		"127.0.0.1:4242":    http.StatusNoContent,
		"LOCALHOST:4242":    http.StatusNoContent,
		"[::1]:4242":        http.StatusNoContent,
		"127.0.0.1:80":      http.StatusMisdirectedRequest,
		"evil.example:4242": http.StatusMisdirectedRequest,
		"":                  http.StatusMisdirectedRequest,
	} {
		r := httptest.NewRequest(http.MethodGet, "/healthz", nil)
		r.Host = host
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != want {
			t.Errorf("Host %q: status %d, want %d", host, w.Code, want)
		}
		if want != http.StatusMisdirectedRequest {
			continue
		}
		var p Problem
		if err := json.NewDecoder(w.Body).Decode(&p); err != nil || p.Status != want || w.Header().Get("Content-Type") != "application/problem+json" {
			t.Errorf("Host %q: the refusal is not a problem: %+v %v", host, p, err)
		}
	}
}
