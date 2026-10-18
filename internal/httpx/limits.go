package httpx

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
)

type requestIDKey struct{}

// RequestID returns the id assigned to this request.
func RequestID(ctx context.Context) string {
	s, _ := ctx.Value(requestIDKey{}).(string)
	return s
}

// WithRequestID assigns an id, echoes it and puts it in the context so logs
// and error responses can be correlated. A client's own X-Request-Id is kept
// if it is short enough to be an id rather than a payload.
func WithRequestID() Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			id := r.Header.Get("X-Request-Id")
			if id == "" || len(id) > 64 {
				var b [8]byte
				_, _ = rand.Read(b[:])
				id = hex.EncodeToString(b[:])
			}
			w.Header().Set("X-Request-Id", id)
			next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), requestIDKey{}, id)))
		})
	}
}

// LimitBody caps the request body before any handler reads it, so an
// oversized body is refused without being buffered first.
func LimitBody(maxBytes int64) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Body != nil {
				r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
			}
			next.ServeHTTP(w, r)
		})
	}
}

// SecurityHeaders sets the headers that cost nothing and prevent a class of
// mistakes.
func SecurityHeaders() Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			h := w.Header()
			h.Set("X-Content-Type-Options", "nosniff")
			h.Set("Referrer-Policy", "no-referrer")
			next.ServeHTTP(w, r)
		})
	}
}

// burstFactor is how much of a burst a key may spend at once, as a multiple of
// its per-second rate: a UI opens several requests on page load, and a limit
// with no burst refuses that ordinary case.
const burstFactor = 2

// RateLimiter throttles per API key with a token bucket. One process and a
// handful of keys, so the buckets live in memory.
type RateLimiter struct {
	mu      sync.Mutex
	buckets map[string]*bucket
	now     func() time.Time
}

type bucket struct {
	tokens float64
	last   time.Time
}

func NewRateLimiter() *RateLimiter {
	return &RateLimiter{buckets: map[string]*bucket{}, now: time.Now}
}

// Allow reports whether the key may make a request now at rate per second,
// and how long to wait if not. A rate of zero is unlimited.
func (l *RateLimiter) Allow(id string, rate float64) (bool, time.Duration) {
	if rate <= 0 {
		return true, 0
	}
	capacity := math.Max(1, rate*burstFactor)
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	b, ok := l.buckets[id]
	if !ok {
		b = &bucket{tokens: capacity, last: now}
		l.buckets[id] = b
	}
	if elapsed := now.Sub(b.last).Seconds(); elapsed > 0 {
		b.tokens = math.Min(capacity, b.tokens+elapsed*rate)
		b.last = now
	}
	if b.tokens < 1 {
		// Round up: advising a wait a hair too short guarantees the retry is
		// refused too.
		return false, time.Duration(math.Ceil((1-b.tokens)/rate*1000)) * time.Millisecond
	}
	b.tokens--
	return true, 0
}

// RateLimit refuses requests from a key over its rps. It goes after Auth:
// until the key is known there is nothing to throttle, and open mode has no
// keys.
func RateLimit(l *RateLimiter) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			k, ok := KeyFrom(r.Context())
			if !ok {
				next.ServeHTTP(w, r)
				return
			}
			if allowed, wait := l.Allow(k.Name, k.RPS); !allowed {
				w.Header().Set("Retry-After", strconv.Itoa(max(1, int(math.Ceil(wait.Seconds())))))
				WriteProblem(w, core.Errorf(core.CodeRateLimited, "this API key is over its request rate"))
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}
