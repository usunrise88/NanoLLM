package httpx

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRateLimiter(t *testing.T) {
	l := NewRateLimiter()
	now := time.Unix(0, 0)
	l.now = func() time.Time { return now }
	for i := range 4 {
		if ok, _ := l.Allow("k", 2); !ok {
			t.Fatalf("burst request %d refused", i)
		}
	}
	ok, wait := l.Allow("k", 2)
	if ok || wait <= 0 || wait > time.Second {
		t.Fatalf("over the burst: %v %v", ok, wait)
	}
	now = now.Add(wait)
	if ok, _ := l.Allow("k", 2); !ok {
		t.Fatal("refused after the advised wait")
	}
	if ok, _ := l.Allow("other", 0); !ok {
		t.Fatal("zero rate is unlimited")
	}
}

func TestLimitBodyAndRequestID(t *testing.T) {
	h := Chain(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, 100)
		_, err := r.Body.Read(buf)
		if err != nil && strings.Contains(err.Error(), "too large") {
			w.WriteHeader(http.StatusRequestEntityTooLarge)
			return
		}
		if RequestID(r.Context()) == "" {
			t.Error("no request id in context")
		}
	}), WithRequestID(), LimitBody(10), SecurityHeaders())
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/", strings.NewReader(strings.Repeat("x", 50))))
	if rec.Code != http.StatusRequestEntityTooLarge || rec.Header().Get("X-Request-Id") == "" || rec.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Fatalf("%d %v", rec.Code, rec.Header())
	}
}
