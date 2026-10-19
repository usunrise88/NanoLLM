package server

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/promptcache"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/sched"
)

// The prompt cache on disk as the runtime wires it (OPT-13): off by default and
// then absent altogether, on by cache.prompt_disk, restored across a restart.

func complete(t *testing.T, rt *Runtime, prompt string) *core.CompletionResult {
	t.Helper()
	res, err := rt.Engine.Complete(context.Background(), core.CompletionRequest{
		Generation: core.Generation{Model: "alpha"}, Prompt: prompt,
	}, nil)
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	return res
}

func diskStats(rt *Runtime) *sched.PromptDiskStats {
	return resident(rt)["alpha"].Loop.PromptDisk
}

const sharedPrefix = "A long prefix that a few requests have in common, as a system prompt is. "

func TestPromptDiskIsOffByDefault(t *testing.T) {
	data := t.TempDir()
	rt := newRuntime(t, 1, fake.Options{}, func(c *config.Config) { c.Server.DataDir = data })
	if c := config.Default().Cache; c.PromptDisk || c.PromptMaxBytes != 2<<30 || c.PromptMinTokens != 256 || c.PromptDir != "" {
		t.Fatalf("defaults %+v", c)
	}
	chat(t, rt, "alpha", strings.Repeat(sharedPrefix, 10))
	complete(t, rt, strings.Repeat(sharedPrefix, 10)+"[[fake:echo]]")
	if diskStats(rt) != nil {
		t.Fatalf("the loop reports a disk cache: %+v", diskStats(rt))
	}
	if ents, _ := os.ReadDir(data); len(ents) != 0 {
		t.Fatalf("a runtime with the option off wrote to its data directory: %v", ents)
	}

	mux := http.NewServeMux()
	rt.MountStats(mux)
	for _, path := range []string{"/api/v1/stats", "/metrics"} {
		body := serve(mux, http.MethodGet, path, "").Body.String()
		if strings.Contains(body, "prompt_disk") {
			t.Errorf("GET %s mentions the disk cache with the option off", path)
		}
	}
}

func TestPromptDiskThroughTheRuntime(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "states")
	on := func(c *config.Config) {
		c.Cache.PromptDisk = true
		c.Cache.PromptDir = dir
		c.Cache.PromptMinTokens = 16
	}
	long := strings.Repeat(sharedPrefix, 6)
	follow := long + "and the question [[fake:echo]]"

	// What the answer is with nothing on disk.
	ref := newRuntime(t, 1, fake.Options{})
	want := complete(t, ref, follow)
	if want.Stats.CachedTokens != 0 || want.Text == "" {
		t.Fatalf("reference %+v", want)
	}

	first := newRuntime(t, 1, fake.Options{}, on)
	complete(t, first, long)
	eventually(t, "the state to be handed to the store", func() bool {
		d := diskStats(first)
		return d != nil && d.Saves == 1
	})
	mux := http.NewServeMux()
	first.MountStats(mux)
	if body := serve(mux, http.MethodGet, "/api/v1/stats", "").Body.String(); !strings.Contains(body, `"prompt_disk":{"saves":1`) {
		t.Errorf("stats JSON: %s", body)
	}
	if body := serve(mux, http.MethodGet, "/metrics", "").Body.String(); !strings.Contains(body, `nanollm_prompt_disk_saves_total{device="cpu",model="alpha"} 1`) ||
		!strings.Contains(body, "nanollm_prompt_disk_restored_tokens_total") {
		t.Errorf("metrics: %s", body)
	}
	first.Close() // writes what is queued
	if ents, err := filepath.Glob(filepath.Join(dir, "*"+promptcache.Ext)); err != nil || len(ents) != 1 {
		t.Fatalf("files %v, %v", ents, err)
	}

	// A new process: nothing is in memory but the files.
	second := newRuntime(t, 1, fake.Options{}, on)
	got := complete(t, second, follow)
	if got.Text != want.Text {
		t.Fatalf("restored answer %q, uncached %q", got.Text, want.Text)
	}
	if got.Stats.CachedTokens < len(long) {
		t.Fatalf("cached %d tokens of a %d-byte shared prefix", got.Stats.CachedTokens, len(long))
	}
	if d := diskStats(second); d == nil || d.Restores != 1 || d.RestoredTokens != int64(got.Stats.CachedTokens) || d.RestoreErrors != 0 {
		t.Fatalf("stats %+v", d)
	}
}

func TestPromptDiskDefaultDirectory(t *testing.T) {
	data := t.TempDir()
	rt := newRuntime(t, 1, fake.Options{}, func(c *config.Config) {
		c.Server.DataDir = data
		c.Cache.PromptDisk = true
		c.Cache.PromptMinTokens = 16
	})
	complete(t, rt, strings.Repeat(sharedPrefix, 3))
	rt.Close()
	if ents, _ := filepath.Glob(filepath.Join(data, "promptcache", "*"+promptcache.Ext)); len(ents) != 1 {
		t.Fatalf("files %v under %s", ents, data)
	}
}

func TestPromptDiskFingerprint(t *testing.T) {
	b := fake.New(fake.Options{})
	m := registry.Manifest{ID: "alpha", Files: []registry.File{
		{Role: registry.RoleModel, SHA256: strings.Repeat("a", 64)},
		{Role: registry.RoleDrafter, SHA256: strings.Repeat("b", 64)},
	}}
	spec := backend.LoadSpec{
		Path: "/m/model.gguf", DrafterPath: "/m/drafter.gguf", KVTypeK: "f16", KVTypeV: "f16",
		FlashAttention: true, SWAFull: true, NCtx: 4096, Slots: 2,
	}
	base := promptFingerprint(m, spec, b)
	if promptFingerprint(m, spec, b) != base {
		t.Fatal("not stable")
	}
	// The path is not part of the weights; the content is.
	moved := spec
	moved.Path = "/elsewhere/model.gguf"
	if promptFingerprint(m, moved, b) != base {
		t.Error("the fingerprint follows the path of a pinned file")
	}
	for name, mutate := range map[string]func(*backend.LoadSpec){
		"kv type k":  func(s *backend.LoadSpec) { s.KVTypeK = "q8_0" },
		"kv type v":  func(s *backend.LoadSpec) { s.KVTypeV = "q8_0" },
		"flash":      func(s *backend.LoadSpec) { s.FlashAttention = false },
		"swa_full":   func(s *backend.LoadSpec) { s.SWAFull = false },
		"kv unified": func(s *backend.LoadSpec) { s.KVUnified = true },
		"n_ctx":      func(s *backend.LoadSpec) { s.NCtx = 8192 },
		"slots":      func(s *backend.LoadSpec) { s.Slots = 3 },
		"no drafter": func(s *backend.LoadSpec) { s.DrafterPath = "" },
	} {
		s := spec
		mutate(&s)
		if promptFingerprint(m, s, b) == base {
			t.Errorf("the fingerprint ignores %s", name)
		}
	}
	other := m
	other.Files = []registry.File{{Role: registry.RoleModel, SHA256: strings.Repeat("c", 64)}, m.Files[1]}
	if promptFingerprint(other, spec, b) == base {
		t.Error("the fingerprint ignores the model file's sha256")
	}
	if promptFingerprint(m, spec, otherBackend{b}) == base {
		t.Error("the fingerprint ignores the engine build")
	}
}

// otherBackend is the same engine at another version.
type otherBackend struct{ *fake.Backend }

func (otherBackend) Version() string { return "another build" }
