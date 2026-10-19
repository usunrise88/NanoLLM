package sched

import (
	"io"
	"sync/atomic"

	"github.com/usunrise88/nanollm/internal/backend"
)

// PromptStore is where the loop keeps the states of prompts it has prefilled,
// beyond the slots themselves (OPT-13, internal/promptcache). It is optional:
// a loop without one behaves exactly as before.
//
// The loop calls it from its own goroutine, between model steps, so Lookup and
// Has must not block on anything slow, and Save must hand the write to
// another goroutine. Load reads a file and does block; a restore is worth it
// only because reading a state is far cheaper than computing it.
type PromptStore interface {
	// Lookup returns the token count of the longest stored state whose tokens
	// are a proper prefix of prompt (shorter than prompt).
	Lookup(prompt []backend.Token) (n int, ok bool)
	// Load returns the state stored for prompt[:n]. An entry that turns out
	// to be gone or damaged is an error, and is not offered again.
	Load(prompt []backend.Token, n int) ([]byte, error)
	// Has reports whether this exact token list has a state.
	Has(tokens []backend.Token) bool
	// Save takes ownership of state and stores it for tokens, later. It
	// returns an error for a state it dropped.
	Save(tokens []backend.Token, state []byte) error
}

// DefaultPromptMinTokens is the shortest prompt saved when SetPromptStore is
// given no minimum.
const DefaultPromptMinTokens = 256

// promptDisk is the store with the loop's settings for it.
type promptDisk struct {
	store     PromptStore
	minTokens int
}

// PromptDiskStats counts what the prompt cache on disk did for this loop.
type PromptDiskStats struct {
	// Saves are states handed to the store after a prefill.
	Saves int64 `json:"saves"`
	// Restores are requests that started from a state read from disk, and
	// RestoredTokens the prompt tokens they did not have to evaluate.
	Restores       int64 `json:"restores"`
	RestoredTokens int64 `json:"restored_tokens"`
	// SaveErrors are states that could not be taken from the model or were
	// refused by the store; RestoreErrors are states that could not be read
	// or put into a slot. A request goes on without the state in both cases.
	SaveErrors    int64 `json:"save_errors"`
	RestoreErrors int64 `json:"restore_errors"`
}

type promptDiskCounters struct {
	saves, restores, restoredTokens, saveErrors, restoreErrors atomic.Int64
}

// SetPromptStore attaches the disk cache. Call it before the first Submit;
// nil detaches it. minTokens is the shortest prompt saved (zero is
// DefaultPromptMinTokens), and a restore must beat the slot's own cache by
// at least half of it. The loop takes the store over: when the loop stops, it
// closes the store if it is an io.Closer, which writes what is still queued.
func (l *Loop) SetPromptStore(store PromptStore, minTokens int) {
	if store == nil {
		l.disk.Store(nil)
		return
	}
	if minTokens <= 0 {
		minTokens = DefaultPromptMinTokens
	}
	l.disk.Store(&promptDisk{store: store, minTokens: minTokens})
}

// closePromptStore runs when the loop stops.
func (l *Loop) closePromptStore() {
	if d := l.disk.Load(); d != nil {
		if c, ok := d.store.(io.Closer); ok {
			_ = c.Close()
		}
	}
}

// restorePrompt tries to give slot seq the longest stored prefix of the
// request's prompt, if that beats what the slot holds (reuse tokens) by
// enough. On success the slot holds exactly prompt[:n] and n is returned. On
// failure the slot is as the caller left it, or emptied if the model may have
// been touched, and 0 is returned: the request goes on as if nothing had been
// found. The caller has checked that the request may use the cache.
func (l *Loop) restorePrompt(seq int, prompt []backend.Token, reuse int) int {
	d := l.disk.Load()
	if d == nil {
		return 0
	}
	n, ok := d.store.Lookup(prompt)
	if !ok || n >= len(prompt) || n-reuse < max(1, d.minTokens/2) {
		return 0
	}
	state, err := d.store.Load(prompt, n)
	if err != nil {
		l.diskStats.restoreErrors.Add(1)
		return 0
	}
	s := &l.slots[seq]
	l.model.ClearSeq(seq)
	s.cached = s.cached[:0]
	if err := l.model.RestoreSeq(seq, state); err != nil {
		// A half-restored state is worse than none.
		l.model.ClearSeq(seq)
		s.cached = nil
		l.diskStats.restoreErrors.Add(1)
		return 0
	}
	s.cached = append(s.cached, prompt[:n]...)
	l.diskStats.restores.Add(1)
	l.diskStats.restoredTokens.Add(int64(n))
	return n
}

// savePrompt stores the state of a slot whose prompt has just been
// prefilled, if it is worth it: long enough, not opted out, and not already
// there. It is called right after the request's first token was delivered,
// when the slot holds exactly the prompt. The model call is made here, on
// the loop's goroutine, as every model call is; the file is written by the
// store's own goroutine.
func (l *Loop) savePrompt(seq int, h *Handle) {
	d := l.disk.Load()
	if d == nil || h.req.NoCache {
		return
	}
	prompt := h.req.Prompt
	s := &l.slots[seq]
	if len(prompt) < d.minTokens || len(s.cached) != len(prompt) || d.store.Has(prompt) {
		return
	}
	state, err := l.model.SaveSeq(seq)
	if err != nil {
		l.diskStats.saveErrors.Add(1)
		return
	}
	if err := d.store.Save(prompt, state); err != nil {
		l.diskStats.saveErrors.Add(1)
		return
	}
	l.diskStats.saves.Add(1)
}
