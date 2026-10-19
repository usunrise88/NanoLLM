package sched

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/promptcache"
)

// The prompt cache on disk through the loop, on the fake backend. A "restart"
// is a second loop over a second fake model and a second store on the same
// directory: nothing but the files carries over.

var diskFP = promptcache.NewFingerprint("test=sched", "n_ctx=4096")

const diskMin = 64

// diskLoop is a loop with a store on dir. Closing the loop closes the store,
// which writes what is queued; tests that look at the directory close the loop
// first.
type diskLoop struct {
	*Loop
	m     backend.Model
	store *promptcache.Store
}

func newDiskLoop(t *testing.T, dir string, fp promptcache.Fingerprint, max int64, wrap func(backend.Model) backend.Model, spec backend.LoadSpec) *diskLoop {
	t.Helper()
	m, err := fake.New(fake.Options{}).Load(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	store, err := promptcache.Open(promptcache.Options{Dir: dir, Fingerprint: fp, MaxBytes: max})
	if err != nil {
		t.Fatal(err)
	}
	lm := m
	if wrap != nil {
		lm = wrap(m)
	}
	l := New(lm)
	l.SetPromptStore(store, diskMin)
	t.Cleanup(func() { l.Close(); m.Close() })
	return &diskLoop{Loop: l, m: m, store: store}
}

// longPrompt is a conversation turn of about 330 tokens that differs from
// every other tag from its first words on, so no two share a prefix worth
// restoring.
func longPrompt(m backend.Model, tag string) []backend.Token {
	return prompt(m, tag+strings.Repeat(" the quick brown fox", 15)+" [[fake:echo]]")
}

// extend is the prompt with a follow-up turn after it.
func extend(m backend.Model, p []backend.Token, user string) []backend.Token {
	return append(append([]backend.Token(nil), p...), prompt(m, user+" [[fake:echo]]")...)
}

func nlpcFiles(t *testing.T, dir string) []string {
	t.Helper()
	names, err := filepath.Glob(filepath.Join(dir, "*"+promptcache.Ext))
	if err != nil {
		t.Fatal(err)
	}
	return names
}

func dirSize(t *testing.T, dir string) (n int64) {
	t.Helper()
	for _, f := range nlpcFiles(t, dir) {
		fi, err := os.Stat(f)
		if err != nil {
			t.Fatal(err)
		}
		n += fi.Size()
	}
	return n
}

func disk(t *testing.T, l *Loop) PromptDiskStats {
	t.Helper()
	d := l.Snapshot().PromptDisk
	if d == nil {
		t.Fatal("the loop reports no prompt cache on disk")
	}
	return *d
}

// waitSaved waits until n states were handed over and written.
func (d *diskLoop) waitSaved(t *testing.T, n int64) {
	t.Helper()
	eventually(t, "the state to be saved", func() bool { return disk(t, d.Loop).Saves >= n })
	d.store.Flush()
}

// uncached is what the model answers to a prompt with no disk cache anywhere.
func uncached(t *testing.T, p []backend.Token) string {
	t.Helper()
	l, m := newLoop(t, fake.Options{}, backend.LoadSpec{Slots: 1})
	text, ev := run(t, l, m, Request{Prompt: p, Logprobs: -1})
	if ev.Err != nil || ev.Stats.CachedTokens != 0 {
		t.Fatalf("reference run: %+v", ev)
	}
	return text
}

func TestPromptDiskSavesAfterPrefill(t *testing.T) {
	dir := t.TempDir()
	d := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})

	// Too short to be worth a file.
	short := prompt(d.m, "hi [[fake:echo]]")
	if len(short) >= diskMin {
		t.Fatalf("setup: the short prompt has %d tokens", len(short))
	}
	run(t, d.Loop, d.m, Request{Prompt: short, Logprobs: -1})

	long := longPrompt(d.m, "alpha")
	if len(long) < 4*diskMin {
		t.Fatalf("setup: the long prompt has %d tokens", len(long))
	}
	run(t, d.Loop, d.m, Request{Prompt: long, Logprobs: -1})
	d.waitSaved(t, 1)
	if got := nlpcFiles(t, dir); len(got) != 1 {
		t.Fatalf("files after one long prompt: %v", got)
	}
	if !d.store.Has(long) {
		t.Fatal("the store has no entry for exactly the prompt")
	}

	// The same prompt again is already stored, and the follow-up turn is a
	// new prompt with its own entry.
	run(t, d.Loop, d.m, Request{Prompt: long, Logprobs: -1})
	run(t, d.Loop, d.m, Request{Prompt: extend(d.m, long, "and then"), Logprobs: -1})
	d.waitSaved(t, 2)
	d.Close()
	if got := nlpcFiles(t, dir); len(got) != 2 {
		t.Fatalf("files: %v", got)
	}
	if st := disk(t, d.Loop); st.Saves != 2 || st.SaveErrors != 0 || st.Restores != 0 {
		t.Fatalf("stats %+v", st)
	}
}

func TestPromptDiskRestoresAfterARestart(t *testing.T) {
	dir := t.TempDir()
	first := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	p := longPrompt(first.m, "bravo")
	text, ev := run(t, first.Loop, first.m, Request{Prompt: p, Logprobs: -1})
	if ev.Err != nil || !strings.HasPrefix(text, "bravo") {
		t.Fatalf("first run: %q %+v", text, ev)
	}
	first.Close()

	follow := extend(first.m, p, "now the follow-up")
	want := uncached(t, follow)
	if want != "now the follow-up" {
		t.Fatalf("reference answer %q", want)
	}

	second := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	got, ev := run(t, second.Loop, second.m, Request{Prompt: follow, Logprobs: -1})
	if ev.Err != nil || got != want {
		t.Fatalf("after a restart: %q, %v; want %q", got, ev.Err, want)
	}
	if ev.Stats.CachedTokens != len(p) {
		t.Fatalf("cached %d tokens, restored prefix is %d", ev.Stats.CachedTokens, len(p))
	}
	if st := disk(t, second.Loop); st.Restores != 1 || st.RestoredTokens != int64(len(p)) || st.RestoreErrors != 0 {
		t.Fatalf("stats %+v", st)
	}
	// The slot's record is what the model holds.
	if cached := mirrored(t, second.Loop, second.m, 0); len(cached) < len(follow) {
		t.Fatalf("slot holds %d tokens of a %d-token prompt", len(cached), len(follow))
	}

	// A restored state serves the next turn from the slot, not the disk.
	more := extend(second.m, follow, "and once more")
	got, ev = run(t, second.Loop, second.m, Request{Prompt: more, Logprobs: -1})
	if got != "and once more" || ev.Stats.CachedTokens != len(follow) {
		t.Fatalf("third turn: %q, cached %d of %d", got, ev.Stats.CachedTokens, len(follow))
	}
	if st := disk(t, second.Loop); st.Restores != 1 {
		t.Fatalf("the slot's own cache was beaten by the disk: %+v", st)
	}
}

func TestPromptDiskRestoresIntoAFreeSlotOfSeveral(t *testing.T) {
	dir := t.TempDir()
	first := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 3})
	a, b := longPrompt(first.m, "charlie"), longPrompt(first.m, "delta")
	run(t, first.Loop, first.m, Request{Prompt: a, Logprobs: -1})
	run(t, first.Loop, first.m, Request{Prompt: b, Logprobs: -1})
	first.Close()

	second := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 3})
	for _, p := range [][]backend.Token{b, a} {
		follow := extend(second.m, p, "follow-up")
		got, ev := run(t, second.Loop, second.m, Request{Prompt: follow, Logprobs: -1})
		if got != "follow-up" || ev.Stats.CachedTokens != len(p) {
			t.Fatalf("%q, cached %d of %d", got, ev.Stats.CachedTokens, len(p))
		}
	}
	if st := disk(t, second.Loop); st.Restores != 2 {
		t.Fatalf("stats %+v", st)
	}
}

func TestPromptDiskKeepsTheSlotWhenItHoldsAsMuch(t *testing.T) {
	dir := t.TempDir()
	d := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	p := longPrompt(d.m, "echo")
	run(t, d.Loop, d.m, Request{Prompt: p, Logprobs: -1})
	d.waitSaved(t, 1)
	// The slot holds the prompt the entry was made from: restoring would
	// gain nothing.
	_, ev := run(t, d.Loop, d.m, Request{Prompt: extend(d.m, p, "next"), Logprobs: -1})
	if ev.Stats.CachedTokens != len(p) {
		t.Fatalf("cached %d of %d", ev.Stats.CachedTokens, len(p))
	}
	if st := disk(t, d.Loop); st.Restores != 0 || st.RestoreErrors != 0 {
		t.Fatalf("stats %+v", st)
	}
}

func TestPromptDiskUsesTheLongestEntry(t *testing.T) {
	dir := t.TempDir()
	first := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	short := longPrompt(first.m, "foxtrot")
	long := extend(first.m, short, "the second turn")
	run(t, first.Loop, first.m, Request{Prompt: short, Logprobs: -1})
	run(t, first.Loop, first.m, Request{Prompt: long, Logprobs: -1})
	first.Close()
	if got := nlpcFiles(t, dir); len(got) != 2 {
		t.Fatalf("files %v", got)
	}

	second := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	_, ev := run(t, second.Loop, second.m, Request{Prompt: extend(second.m, long, "the third"), Logprobs: -1})
	if ev.Stats.CachedTokens != len(long) {
		t.Fatalf("cached %d tokens, the longest entry has %d (the shorter %d)", ev.Stats.CachedTokens, len(long), len(short))
	}
}

func TestPromptDiskIgnoresAnotherFingerprint(t *testing.T) {
	dir := t.TempDir()
	first := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	p := longPrompt(first.m, "golf")
	run(t, first.Loop, first.m, Request{Prompt: p, Logprobs: -1})
	first.Close()
	saved := nlpcFiles(t, dir)
	if len(saved) != 1 {
		t.Fatalf("files %v", saved)
	}

	other := promptcache.NewFingerprint("test=sched", "n_ctx=8192")
	second := newDiskLoop(t, dir, other, 0, nil, backend.LoadSpec{Slots: 1})
	follow := extend(second.m, p, "follow-up")
	got, ev := run(t, second.Loop, second.m, Request{Prompt: follow, Logprobs: -1})
	if got != "follow-up" || ev.Err != nil || ev.Stats.CachedTokens != 0 {
		t.Fatalf("%q, cached %d, %v", got, ev.Stats.CachedTokens, ev.Err)
	}
	if st := disk(t, second.Loop); st.Restores != 0 || st.RestoreErrors != 0 {
		t.Fatalf("stats %+v", st)
	}
	second.Close()
	// The first model's file is still there, beside the new one.
	if got := nlpcFiles(t, dir); len(got) != 2 {
		t.Fatalf("files %v", got)
	}
	if _, err := os.Stat(saved[0]); err != nil {
		t.Fatalf("the other fingerprint's file is gone: %v", err)
	}
}

func TestPromptDiskDamagedFilesDoNotBreakTheRequest(t *testing.T) {
	for _, tc := range []struct {
		name string
		f    func([]byte) []byte
		// scanned: the damage is found when the directory is read, the
		// state is never offered; otherwise it is found when loaded.
		scanned bool
	}{
		{"truncated", func(b []byte) []byte { return b[:len(b)/2] }, true},
		{"a flipped token", func(b []byte) []byte { b[100] ^= 0xff; return b }, true},
		{"a flipped byte of the state", func(b []byte) []byte { b[len(b)-9] ^= 0xff; return b }, false},
		{"garbage", func(b []byte) []byte { return []byte("not a state") }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			first := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
			p := longPrompt(first.m, "hotel")
			run(t, first.Loop, first.m, Request{Prompt: p, Logprobs: -1})
			first.Close()
			files := nlpcFiles(t, dir)
			if len(files) != 1 {
				t.Fatalf("files %v", files)
			}
			b, err := os.ReadFile(files[0])
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(files[0], tc.f(b), 0o600); err != nil {
				t.Fatal(err)
			}

			follow := extend(first.m, p, "follow-up")
			second := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
			got, ev := run(t, second.Loop, second.m, Request{Prompt: follow, Logprobs: -1})
			if ev.Err != nil || got != "follow-up" || ev.Stats.CachedTokens != 0 {
				t.Fatalf("%q, cached %d, %v", got, ev.Stats.CachedTokens, ev.Err)
			}
			st := disk(t, second.Loop)
			wantErrors := int64(1)
			if tc.scanned {
				wantErrors = 0
			}
			if st.Restores != 0 || st.RestoreErrors != wantErrors {
				t.Fatalf("stats %+v, want %d restore errors", st, wantErrors)
			}
			mirrored(t, second.Loop, second.m, 0)
		})
	}
}

func TestPromptDiskEvictsTheLeastRecentlyUsed(t *testing.T) {
	dir := t.TempDir()
	probe := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	// Every prompt has the same length, so every state has the same size.
	prompts := make([][]backend.Token, 5)
	for i := range prompts {
		prompts[i] = longPrompt(probe.m, string(rune('a'+i))+"-turn")
	}
	run(t, probe.Loop, probe.m, Request{Prompt: prompts[0], Logprobs: -1})
	probe.Close()
	one := dirSize(t, dir)
	if one == 0 {
		t.Fatal("setup: nothing saved")
	}
	if err := os.Remove(nlpcFiles(t, dir)[0]); err != nil {
		t.Fatal(err)
	}

	limit := 2*one + one/2 // two states fit
	d := newDiskLoop(t, dir, diskFP, limit, nil, backend.LoadSpec{Slots: 1})
	for i := 0; i < 2; i++ {
		run(t, d.Loop, d.m, Request{Prompt: prompts[i], Logprobs: -1})
		d.waitSaved(t, int64(i+1))
	}
	// Using the first entry makes the second the one to go.
	if _, ok := d.store.Lookup(extend(d.m, prompts[0], "x")); !ok {
		t.Fatal("setup: the first state is not stored")
	}
	if _, err := d.store.Load(extend(d.m, prompts[0], "x"), len(prompts[0])); err != nil {
		t.Fatal(err)
	}
	run(t, d.Loop, d.m, Request{Prompt: prompts[2], Logprobs: -1})
	d.waitSaved(t, 3)
	if got := dirSize(t, dir); got > limit {
		t.Fatalf("the directory holds %d bytes, the cap is %d", got, limit)
	}
	for i, want := range []bool{true, false, true} {
		if d.store.Has(prompts[i]) != want {
			t.Errorf("prompt %d stored: %v, want %v", i, !want, want)
		}
	}
	for i := 3; i < 5; i++ {
		run(t, d.Loop, d.m, Request{Prompt: prompts[i], Logprobs: -1})
		d.waitSaved(t, int64(i+1))
		if got := dirSize(t, dir); got > limit {
			t.Fatalf("after %d states the directory holds %d bytes, the cap is %d", i+1, got, limit)
		}
	}
	if len(nlpcFiles(t, dir)) != 2 || !d.store.Has(prompts[4]) {
		t.Fatalf("files %v", nlpcFiles(t, dir))
	}
}

func TestPromptDiskLeavesNoCacheRequestsAlone(t *testing.T) {
	dir := t.TempDir()
	first := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	kept := longPrompt(first.m, "india")
	// An opted-out request is not written down...
	run(t, first.Loop, first.m, Request{Prompt: longPrompt(first.m, "juliet"), NoCache: true, Logprobs: -1})
	// ...and an ordinary one is.
	run(t, first.Loop, first.m, Request{Prompt: kept, Logprobs: -1})
	first.Close()
	if got := nlpcFiles(t, dir); len(got) != 1 {
		t.Fatalf("files %v", got)
	}
	if st := disk(t, first.Loop); st.Saves != 1 {
		t.Fatalf("stats %+v", st)
	}

	follow := extend(first.m, kept, "follow-up")
	second := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	got, ev := run(t, second.Loop, second.m, Request{Prompt: follow, NoCache: true, Logprobs: -1})
	if got != "follow-up" || ev.Stats.CachedTokens != 0 {
		t.Fatalf("NoCache: %q, cached %d", got, ev.Stats.CachedTokens)
	}
	if st := disk(t, second.Loop); st.Restores != 0 || st.RestoreErrors != 0 {
		t.Fatalf("NoCache touched the disk cache: %+v", st)
	}
	second.Close()
	if got := nlpcFiles(t, dir); len(got) != 1 {
		t.Fatalf("a NoCache request left files: %v", got)
	}

	// The entry is untouched for the next start, which may use it.
	third := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
	got, ev = run(t, third.Loop, third.m, Request{Prompt: follow, Logprobs: -1})
	if got != "follow-up" || ev.Stats.CachedTokens != len(kept) {
		t.Fatalf("after NoCache: %q, cached %d of %d", got, ev.Stats.CachedTokens, len(kept))
	}
}

// failingRestore is a model whose RestoreSeq fails. With dirty set it first
// does the restore, as a backend that fails halfway would.
type failingRestore struct {
	backend.Model
	dirty    bool
	restores atomic.Int64
	clears   atomic.Int64
}

func (f *failingRestore) RestoreSeq(seq int, state []byte) error {
	f.restores.Add(1)
	if f.dirty {
		_ = f.Model.RestoreSeq(seq, state)
	}
	return errors.New("restore failed on purpose")
}

func (f *failingRestore) ClearSeq(seq int) {
	f.clears.Add(1)
	f.Model.ClearSeq(seq)
}

func TestPromptDiskRestoreFailureLeavesACleanSlot(t *testing.T) {
	for _, dirty := range []bool{false, true} {
		name := map[bool]string{false: "refused", true: "half done"}[dirty]
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			first := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 1})
			p := longPrompt(first.m, "kilo")
			run(t, first.Loop, first.m, Request{Prompt: p, Logprobs: -1})
			first.Close()

			var fr *failingRestore
			second := newDiskLoop(t, dir, diskFP, 0, func(m backend.Model) backend.Model {
				fr = &failingRestore{Model: m, dirty: dirty}
				return fr
			}, backend.LoadSpec{Slots: 1})
			follow := extend(second.m, p, "follow-up")
			got, ev := run(t, second.Loop, second.m, Request{Prompt: follow, Logprobs: -1})
			if ev.Err != nil || got != "follow-up" {
				t.Fatalf("%q, %v", got, ev.Err)
			}
			if fr.restores.Load() != 1 || fr.clears.Load() == 0 {
				t.Fatalf("restores %d, clears %d", fr.restores.Load(), fr.clears.Load())
			}
			// Nothing was reused, and what the slot holds is the request's own
			// prompt and answer, not leftovers of the failed restore.
			if ev.Stats.CachedTokens != 0 {
				t.Fatalf("cached %d tokens after a failed restore", ev.Stats.CachedTokens)
			}
			if st := disk(t, second.Loop); st.Restores != 0 || st.RestoreErrors != 1 || st.RestoredTokens != 0 {
				t.Fatalf("stats %+v", st)
			}
			if cached := mirrored(t, second.Loop, second.m, 0); len(cached) < len(follow) {
				t.Fatalf("slot holds %d tokens of a %d-token prompt", len(cached), len(follow))
			}
			// And the loop carries on.
			got, ev = run(t, second.Loop, second.m, Request{Prompt: longPrompt(second.m, "lima"), Logprobs: -1})
			if ev.Err != nil || !strings.HasPrefix(got, "lima") {
				t.Fatalf("next request: %q, %v", got, ev.Err)
			}
		})
	}
}

func TestPromptDiskIsOffByDefault(t *testing.T) {
	l, m := newLoop(t, fake.Options{}, backend.LoadSpec{Slots: 1})
	dir := t.TempDir()
	run(t, l, m, Request{Prompt: longPrompt(m, "mike"), Logprobs: -1})
	l.Close()
	if st := l.Snapshot(); st.PromptDisk != nil {
		t.Fatalf("a loop with no store reports %+v", st.PromptDisk)
	}
	if got, _ := os.ReadDir(dir); len(got) != 0 {
		t.Fatalf("files %v", got)
	}
	// Detaching is the same as never attaching.
	l2, m2 := newLoop(t, fake.Options{}, backend.LoadSpec{Slots: 1})
	store, err := promptcache.Open(promptcache.Options{Dir: dir, Fingerprint: diskFP})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	l2.SetPromptStore(store, diskMin)
	l2.SetPromptStore(nil, 0)
	run(t, l2, m2, Request{Prompt: longPrompt(m2, "november"), Logprobs: -1})
	l2.Close()
	store.Flush()
	if l2.Snapshot().PromptDisk != nil || len(nlpcFiles(t, dir)) != 0 {
		t.Fatal("a detached store was used")
	}
}

func TestPromptDiskUnderConcurrentRequests(t *testing.T) {
	dir := t.TempDir()
	first := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 4, NBatch: 128})
	const n = 8
	prompts := make([][]backend.Token, n)
	for i := range prompts {
		prompts[i] = longPrompt(first.m, string(rune('a'+i))+"-session")
	}
	var wg sync.WaitGroup
	for i := range prompts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			text, ev := run(t, first.Loop, first.m, Request{Prompt: prompts[i], Logprobs: -1})
			if ev.Err != nil || !strings.HasPrefix(text, string(rune('a'+i))+"-session") {
				t.Errorf("request %d: %q, %v", i, text, ev.Err)
			}
		}()
	}
	wg.Wait()
	first.Close()
	// A writer that is behind drops a state rather than queue it, so the
	// prompts that were kept are whatever the disk managed.
	if got := nlpcFiles(t, dir); len(got) == 0 {
		t.Fatal("nothing was saved")
	}

	second := newDiskLoop(t, dir, diskFP, 0, nil, backend.LoadSpec{Slots: 4, NBatch: 128})
	stored := 0
	for _, p := range prompts {
		if second.store.Has(p) {
			stored++
		}
	}
	for i := range prompts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			text, ev := run(t, second.Loop, second.m, Request{Prompt: extend(second.m, prompts[i], "follow-up"), Logprobs: -1})
			if ev.Err != nil || text != "follow-up" {
				t.Errorf("request %d: %q, %v", i, text, ev.Err)
			}
		}()
	}
	wg.Wait()
	// Four slots for eight prompts that share nothing worth keeping: every
	// request that has a state on disk finds it, whichever slot it lands in.
	if st := disk(t, second.Loop); st.Restores != int64(stored) || st.RestoreErrors != 0 {
		t.Fatalf("stats %+v, %d states on disk", st, stored)
	}
}
