package server

import (
	"fmt"
	"os"
	"strconv"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/pool"
	"github.com/usunrise88/nanollm/internal/promptcache"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/sched"
)

// setupLoop is the pool's SetupLoop: it gives a model's loop the prompt cache
// on disk when cache.prompt_disk asks for one. With the option off it returns
// before touching anything, so no directory is created and the loop is what
// it always was. A store that cannot be opened is logged and the model runs
// without it: a full disk or a read-only directory must not keep a model from
// answering.
func (r *Runtime) setupLoop(k pool.Key, l *sched.Loop) {
	c := r.cfg.Cache
	if !c.PromptDisk {
		return
	}
	r.mu.Lock()
	spec, ok := r.specs[k]
	b := r.byName[k.Backend]
	r.mu.Unlock()
	if !ok || b == nil {
		return
	}
	m, err := r.manifest(k.Model)
	if err != nil {
		r.log.Warn("prompt cache on disk is off", "model", k.Model, "err", err)
		return
	}
	dir := r.cfg.PromptCachePath()
	store, err := promptcache.Open(promptcache.Options{
		Dir: dir, Fingerprint: promptFingerprint(m, spec, b), MaxBytes: c.PromptMaxBytes, Log: r.log,
	})
	if err != nil {
		r.log.Warn("prompt cache on disk is off", "model", k.Model, "dir", dir, "err", err)
		return
	}
	l.SetPromptStore(store, c.PromptMinTokens)
	st := store.Stats()
	r.log.Info("prompt cache on disk", "model", k.Model, "dir", dir, "entries", st.Entries,
		"mib", st.Bytes>>20, "max_mib", c.PromptMaxBytes>>20, "min_tokens", c.PromptMinTokens)
}

// promptFingerprint is everything a saved state depends on: the weights (the
// model file's sha256 from the catalog, and the drafter's), how the context
// was built (KV cache types, flash attention, swa_full, unified KV, tokens
// per slot and slot count) and the engine build, whose state layout may
// change between releases. A state saved under another fingerprint is
// never read.
func promptFingerprint(m registry.Manifest, spec backend.LoadSpec, b backend.Backend) promptcache.Fingerprint {
	return promptcache.NewFingerprint(
		"engine="+b.Name()+" "+b.Version(),
		"model="+fileIdentity(m, registry.RoleModel, spec.Path),
		"drafter="+fileIdentity(m, registry.RoleDrafter, spec.DrafterPath),
		"kv_type_k="+spec.KVTypeK,
		"kv_type_v="+spec.KVTypeV,
		"flash_attention="+strconv.FormatBool(spec.FlashAttention),
		"swa_full="+strconv.FormatBool(spec.SWAFull),
		"kv_unified="+strconv.FormatBool(spec.KVUnified),
		"n_ctx="+strconv.Itoa(spec.NCtx),
		"slots="+strconv.Itoa(spec.Slots),
	)
}

// fileIdentity names the bytes of a model file: its sha256 from the catalog,
// or for a file the catalog does not pin, its path, size and modification
// time. A role the load has no file for (no drafter, or a backend that needs
// none) is its catalog entry's name, so that the fingerprint is still
// specific to the model.
func fileIdentity(m registry.Manifest, role, path string) string {
	if path == "" {
		if role == registry.RoleDrafter {
			return "none"
		}
		return "id:" + m.ID
	}
	if f, ok := m.File(role); ok && f.SHA256 != "" {
		return "sha256:" + f.SHA256
	}
	if fi, err := os.Stat(path); err == nil {
		return fmt.Sprintf("file:%s:%d:%d", path, fi.Size(), fi.ModTime().UnixNano())
	}
	return "path:" + path
}
