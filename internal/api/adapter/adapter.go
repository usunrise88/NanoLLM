// Package adapter is how NanoLLM speaks more than one API dialect (SPEC §6).
//
// A dialect is one Go package that registers itself from init() and mounts its
// routes onto the shared mux. Core logic lives behind core.Service and never
// learns which dialect a request arrived through, so adding a dialect is adding
// a package, not threading a new shape through the pipeline.
package adapter

import (
	"fmt"
	"net/http"
	"sort"
	"sync"

	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/registry"
)

// Deps is what a dialect is allowed to reach. Note the absence of the model
// pool, the scheduler and the queue: a dialect that needs them is a sign the
// capability belongs in core.Service instead.
type Deps struct {
	// Service is nil until the v1.0 core lands; dialects answer 501 for every
	// route that needs it.
	Service core.Service
	// Catalog is the built-in model catalog.
	Catalog []registry.Manifest
	// ConfigSnapshot returns the effective configuration with secrets already
	// redacted. A function rather than a value so a dialect cannot be handed
	// the live struct and reach into it.
	ConfigSnapshot func() any
}

// Adapter is one dialect.
type Adapter interface {
	Mount(mux *http.ServeMux, deps Deps) error
}

var (
	mu       sync.RWMutex
	adapters = map[string]Adapter{}
)

// Register makes a dialect available under name. Called from init(); a
// duplicate is a programming error.
func Register(name string, a Adapter) {
	mu.Lock()
	defer mu.Unlock()
	if _, dup := adapters[name]; dup {
		panic(fmt.Sprintf("dialect %q registered twice", name))
	}
	adapters[name] = a
}

// MountAll mounts the named dialects. A name the configuration accepts but
// this build does not carry (ollama before v2) is an error, not a silent skip:
// a server that starts without the API it was told to serve looks healthy to
// everything except its clients.
func MountAll(mux *http.ServeMux, names []string, deps Deps) error {
	mu.RLock()
	defer mu.RUnlock()
	for _, n := range names {
		a, ok := adapters[n]
		if !ok {
			return fmt.Errorf("api.dialects names %q, which this build does not include (have %v)",
				n, available())
		}
		if err := a.Mount(mux, deps); err != nil {
			return fmt.Errorf("mount %s: %w", n, err)
		}
	}
	return nil
}

// Available lists the dialects built into this binary, sorted.
func Available() []string {
	mu.RLock()
	defer mu.RUnlock()
	return available()
}

func available() []string {
	out := make([]string, 0, len(adapters))
	for n := range adapters {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
