package ingest

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

// Limiter is a fixed-window rate limit kept only in memory. Addresses are
// never stored, not even here: the key is a hash of the address with a salt
// drawn at random each UTC day and never written anywhere, so yesterday's
// keys cannot be linked to today's or reversed, and a restart forgets
// everything.
type Limiter struct {
	mu     sync.Mutex
	now    func() time.Time
	day    string
	salt   [32]byte
	window time.Duration
	limit  int
	hits   map[string]*counter
}

type counter struct {
	start time.Time
	n     int
}

// maxKeys bounds the memory a flood of distinct addresses can take: past
// it, the window's table starts over.
const maxKeys = 200000

// NewLimiter allows limit events per window per key; limit ≤ 0 disables it.
func NewLimiter(limit int, window time.Duration) *Limiter {
	return &Limiter{now: time.Now, window: window, limit: limit, hits: map[string]*counter{}}
}

// Key is the salted daily hash of a value (an IP address).
func (l *Limiter) Key(v string) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.rotate()
	h := sha256.New()
	h.Write(l.salt[:])
	h.Write([]byte(v))
	return hex.EncodeToString(h.Sum(nil)[:12])
}

func (l *Limiter) rotate() {
	day := l.now().UTC().Format("2006-01-02")
	if day != l.day {
		l.day = day
		if _, err := rand.Read(l.salt[:]); err != nil {
			panic(err) // crypto/rand does not fail on supported platforms
		}
		l.hits = map[string]*counter{}
	}
}

// Allow counts one event for key and reports whether it is within the
// limit.
func (l *Limiter) Allow(key string) bool {
	if l == nil || l.limit <= 0 {
		return true
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.rotate()
	now := l.now()
	c, ok := l.hits[key]
	if !ok || now.Sub(c.start) >= l.window {
		if !ok && len(l.hits) >= maxKeys {
			l.hits = map[string]*counter{}
		}
		c = &counter{start: now}
		l.hits[key] = c
	}
	if c.n >= l.limit {
		return false
	}
	c.n++
	return true
}
