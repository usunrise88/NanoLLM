package httpx

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"

	"github.com/usunrise88/nanollm/internal/core"
)

// Key is one credential as the server holds it: a digest, never the secret.
type Key struct {
	Name        string
	Admin       bool
	Interactive bool
	RPS         float64
	digest      [sha256.Size]byte
}

// KeySpec is a configured key. It is separate from config.APIKey so this
// package does not depend on the configuration schema.
type KeySpec struct {
	Name        string
	Secret      string // plaintext, or "sha256:<hex>"
	Admin       bool
	Interactive bool
	RPS         float64
}

// KeyStore checks bearer tokens against the configured keys.
type KeyStore struct {
	keys []Key
}

// NewKeyStore hashes every secret up front, so the plaintext can be dropped
// from memory the server keeps.
func NewKeyStore(specs []KeySpec) (*KeyStore, error) {
	s := &KeyStore{}
	for i, spec := range specs {
		k := Key{Name: spec.Name, Admin: spec.Admin, Interactive: spec.Interactive, RPS: spec.RPS}
		if k.Name == "" {
			k.Name = fmt.Sprintf("key-%d", i+1)
		}
		if hexDigest, ok := strings.CutPrefix(spec.Secret, "sha256:"); ok {
			b, err := hex.DecodeString(hexDigest)
			if err != nil || len(b) != sha256.Size {
				return nil, fmt.Errorf("auth.keys[%d] (%q): sha256: must be followed by 64 hex digits", i, spec.Name)
			}
			copy(k.digest[:], b)
		} else {
			k.digest = sha256.Sum256([]byte(spec.Secret))
		}
		s.keys = append(s.keys, k)
	}
	return s, nil
}

// Lookup finds the key a token belongs to. Every key is compared, in constant
// time, so the response time says nothing about which key came close.
func (s *KeyStore) Lookup(token string) (Key, bool) {
	d := sha256.Sum256([]byte(token))
	var found Key
	ok := false
	for _, k := range s.keys {
		if subtle.ConstantTimeCompare(d[:], k.digest[:]) == 1 {
			found, ok = k, true
		}
	}
	return found, ok
}

// Names lists the key names, for the startup log.
func (s *KeyStore) Names() []string {
	out := make([]string, len(s.keys))
	for i, k := range s.keys {
		out[i] = k.Name
	}
	return out
}

type keyCtx struct{}

// KeyFrom returns the key that authenticated the request. ok is false in open
// mode, where nobody authenticated.
func KeyFrom(ctx context.Context) (Key, bool) {
	k, ok := ctx.Value(keyCtx{}).(Key)
	return k, ok
}

// Auth requires a valid key on every path except the public ones. A public
// entry ending in "/" is a prefix; any other is matched exactly, so "/" can
// be public without making the whole server so.
func Auth(store *KeyStore, public ...string) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if isPublic(r.URL.Path, public) {
				next.ServeHTTP(w, r)
				return
			}
			token := bearer(r)
			k, ok := store.Lookup(token)
			if token == "" || !ok {
				w.Header().Set("WWW-Authenticate", `Bearer realm="nanollm"`)
				WriteProblem(w, core.Errorf(core.CodeUnauthorized, "a valid API key is required"))
				return
			}
			next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), keyCtx{}, k)))
		})
	}
}

// RequireAdmin guards model, device and configuration management. In open
// mode there is no key to check and the loopback-only rule is the guard.
func RequireAdmin(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if k, ok := KeyFrom(r.Context()); ok && !k.Admin {
			WriteProblem(w, core.Errorf(core.CodeForbidden, "this endpoint needs an administrative key"))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func bearer(r *http.Request) string {
	t, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	if !ok {
		return ""
	}
	return strings.TrimSpace(t)
}

func isPublic(path string, public []string) bool {
	for _, p := range public {
		if path == p || (strings.HasSuffix(p, "/") && strings.HasPrefix(path, p)) {
			return true
		}
	}
	return false
}
