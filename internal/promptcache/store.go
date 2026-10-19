// Package promptcache keeps the states of prompts that have been prefilled on
// disk, so that a request starting with one of them gets the state back
// instead of computing it again (OPT-13): after a restart, or after the slot
// that held the prompt was reused. It is a store of opaque byte strings keyed
// by the token ids they were computed from; what the bytes are (a backend's
// SaveSeq) is the caller's business, which keeps the package independent of
// any engine.
//
// Every entry is one file (format.go) written under a temporary name and
// renamed into place, so a crash leaves a whole file or none. The index is
// in memory, built by scanning the directory when the store opens: a file
// that is not ours, truncated, corrupted or saved under another fingerprint is
// never served. The ones that carry our magic still count against the size
// cap and age out with the rest, so a directory used by an earlier build does
// not stay full forever. The cap is applied to the whole directory, least
// recently used first.
//
// A directory is meant to belong to one process. Several stores on one
// directory (two resident models) work, but each evicts by its own view of
// the sizes, and a state removed under another store's feet is noticed only
// when that store tries to load it.
package promptcache

import (
	"bufio"
	"crypto/sha256"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
)

// Errors Save and Load return.
var (
	// ErrExists: the tokens already have an entry (stored or being written).
	ErrExists = errors.New("promptcache: this prompt is already stored")
	// ErrTooLarge: the entry alone would not fit under the size cap.
	ErrTooLarge = errors.New("promptcache: the state is larger than the whole cache")
	// ErrBusy: the writer has as many states queued as it holds; the save is
	// dropped rather than letting gigabytes pile up in memory.
	ErrBusy = errors.New("promptcache: too many states are waiting to be written")
	// ErrClosed: the store is closed.
	ErrClosed = errors.New("promptcache: closed")
	// ErrNotFound: no stored state for these tokens.
	ErrNotFound = errors.New("promptcache: no stored state for this prompt")
)

const (
	// queueDepth is how many states may wait for the writer besides the one
	// it is writing: each is the whole slot state, hundreds of megabytes for
	// a long prompt, so at most two are in memory at once.
	queueDepth = 1
	// staleTmp is how old an unfinished temporary file must be to be removed
	// when the store opens.
	staleTmp = 10 * time.Minute
)

// Options open a Store.
type Options struct {
	// Dir holds the files; it is created if missing.
	Dir string
	// Fingerprint says what the states depend on (NewFingerprint).
	Fingerprint Fingerprint
	// MaxBytes caps the directory's stored states; zero or less is no cap.
	MaxBytes int64
	// Log receives what was ignored and what failed; nil discards.
	Log *slog.Logger
}

// Stats is the store's state, for tests and the operator.
type Stats struct {
	// Entries are the states of this store's fingerprint that can be served.
	Entries int
	// Bytes is every tracked file: Entries' and the ones that cannot be
	// served.
	Bytes int64
	// Unusable are tracked files that cannot be served: another fingerprint,
	// a truncated or corrupted file.
	Unusable int
	// WriteErrors counts states that could not be written.
	WriteErrors int64
}

type entry struct {
	sum  [32]byte // sha256 of the token ids
	n    int
	size int64
	name string
	// own: sound and saved under this fingerprint, so it may be served.
	own bool
	// ready: on disk. An entry is pending while its write is queued.
	ready bool
	used  uint64 // the LRU clock at the last save or use
}

type job struct {
	e      *entry
	tokens []backend.Token
	state  []byte
}

// Store is the index of one directory and the goroutine writing to it. It is
// safe for concurrent use.
type Store struct {
	dir string
	fp  Fingerprint
	max int64
	log *slog.Logger

	jobs chan job
	done chan struct{}

	mu     sync.Mutex
	cond   *sync.Cond // signalled when a write finishes
	own    map[[32]byte]*entry
	other  map[string]*entry // tracked files that cannot be served, by name
	lens   []int             // distinct token counts of the ready own entries, ascending; replaced, never edited
	lenCnt map[int]int
	total  int64
	clock  uint64
	// pending counts queued and running writes.
	pending     int
	writeErrors int64
	closed      bool
}

// Open builds the index of dir and starts the writer.
func Open(o Options) (*Store, error) {
	if o.Dir == "" {
		return nil, errors.New("promptcache: no directory")
	}
	if err := os.MkdirAll(o.Dir, 0o700); err != nil {
		return nil, fmt.Errorf("promptcache: %w", err)
	}
	log := o.Log
	if log == nil {
		log = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	s := &Store{
		dir: o.Dir, fp: o.Fingerprint, max: o.MaxBytes, log: log,
		jobs: make(chan job, queueDepth), done: make(chan struct{}),
		own: map[[32]byte]*entry{}, other: map[string]*entry{}, lenCnt: map[int]int{},
	}
	s.cond = sync.NewCond(&s.mu)
	s.scan()
	s.mu.Lock()
	s.evictLocked()
	s.mu.Unlock()
	go s.writer()
	return s, nil
}

// fileName is the name an entry is stored under. Only the first bytes of each
// hash are in it: it is a label for a person looking at the directory, and
// the header is what is trusted.
func (s *Store) fileName(sum [32]byte) string {
	return fmt.Sprintf("%x-%x%s", s.fp[:6], sum[:12], Ext)
}

// scan reads the directory's headers into the index.
func (s *Store) scan() {
	des, err := os.ReadDir(s.dir)
	if err != nil {
		s.log.Warn("prompt cache: cannot read the directory", "dir", s.dir, "err", err)
		return
	}
	type found struct {
		e     *entry
		mtime time.Time
	}
	var all []found
	for _, de := range des {
		name := de.Name()
		if !de.Type().IsRegular() {
			continue
		}
		info, err := de.Info()
		if err != nil {
			continue
		}
		if strings.HasPrefix(name, tmpPrefix) && strings.HasSuffix(name, tmpSuffix) {
			if time.Since(info.ModTime()) > staleTmp {
				_ = os.Remove(filepath.Join(s.dir, name))
			}
			continue
		}
		if !strings.HasSuffix(name, Ext) {
			continue
		}
		e, why := s.inspect(filepath.Join(s.dir, name), info)
		if errors.Is(why, errNotOurs) {
			s.log.Debug("prompt cache: not one of ours, left alone", "file", name)
			continue
		}
		e.name = name
		if why != nil {
			s.log.Debug("prompt cache: file cannot be served", "file", name, "why", why)
		}
		all = append(all, found{e, info.ModTime()})
	}
	// The oldest file is the least recently used.
	sort.SliceStable(all, func(i, j int) bool { return all[i].mtime.Before(all[j].mtime) })
	for _, f := range all {
		e := f.e
		s.clock++
		e.used = s.clock
		if e.own {
			if _, dup := s.own[e.sum]; dup {
				e.own = false
				s.log.Debug("prompt cache: a second file for the same prompt", "file", e.name)
			}
		}
		if e.own {
			s.own[e.sum] = e
			s.lenCnt[e.n]++
		} else {
			s.other[e.name] = e
		}
		s.total += e.size
	}
	s.rebuildLensLocked()
	s.log.Debug("prompt cache: scanned", "dir", s.dir, "usable", len(s.own), "unusable", len(s.other), "bytes", s.total)
}

// inspect validates one file by its header and its token ids; the state is
// not read. The entry is own when the file is sound and ours; otherwise why
// says what is wrong, and errNotOurs means the file is not to be tracked.
func (s *Store) inspect(path string, info fs.FileInfo) (*entry, error) {
	e := &entry{size: info.Size(), ready: true}
	f, err := os.Open(path)
	if err != nil {
		return e, errNotOurs
	}
	defer f.Close()
	var raw [headerSize]byte
	got, _ := io.ReadFull(f, raw[:])
	h, err := parseHeader(raw[:got])
	if err != nil {
		if errors.Is(err, errNotOurs) && got >= len(magic) && string(raw[:len(magic)]) == magic {
			return e, errors.New("truncated header")
		}
		return e, err
	}
	e.n, e.sum = h.n, h.tokens
	switch {
	case info.Size() != h.size():
		return e, fmt.Errorf("truncated or extended: %d bytes, the header says %d", info.Size(), h.size())
	case h.fp != s.fp:
		return e, errors.New("saved under another fingerprint")
	case h.n == 0:
		return e, errors.New("no tokens")
	}
	sum := sha256.New()
	if _, err := io.CopyN(sum, f, 4*int64(h.n)); err != nil {
		return e, fmt.Errorf("reading the tokens: %w", err)
	}
	var tokensSum [32]byte
	sum.Sum(tokensSum[:0])
	if tokensSum != h.tokens {
		return e, errors.New("the token ids do not match their hash")
	}
	e.own = true
	return e, nil
}

// rebuildLensLocked replaces the sorted list of token counts. Lookup reads the
// old slice without the lock, so the list is built afresh, never edited.
func (s *Store) rebuildLensLocked() {
	lens := make([]int, 0, len(s.lenCnt))
	for n := range s.lenCnt {
		lens = append(lens, n)
	}
	sort.Ints(lens)
	s.lens = lens
}

// Has reports whether the exact token list has an entry, stored or being
// written.
func (s *Store) Has(tokens []backend.Token) bool {
	if len(tokens) == 0 {
		return false
	}
	sum := tokensSum(tokens)
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.own[sum]
	return ok
}

// Lookup returns the token count of the longest stored entry whose tokens are
// a proper prefix of prompt (shorter than prompt: a request always evaluates
// at least its last token itself). It hashes the prompt once, front to back,
// stopping to look up only at the token counts that some entry has.
func (s *Store) Lookup(prompt []backend.Token) (int, bool) {
	s.mu.Lock()
	lens := s.lens
	s.mu.Unlock()
	if len(lens) == 0 {
		return 0, false
	}
	type cand struct {
		n   int
		sum [32]byte
	}
	var cands []cand
	h := sha256.New()
	pos := 0
	for _, n := range lens {
		if n >= len(prompt) {
			break
		}
		_ = tokenBytes(h, prompt[pos:n])
		pos = n
		c := cand{n: n}
		h.Sum(c.sum[:0])
		cands = append(cands, c)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(cands) - 1; i >= 0; i-- {
		if e := s.own[cands[i].sum]; e != nil && e.ready && e.n == cands[i].n {
			return e.n, true
		}
	}
	return 0, false
}

// Load reads the state stored for prompt[:n], verifying it end to end. An
// entry whose file is gone or does not verify is dropped from the index (and
// a corrupted file deleted), so that the next lookup does not offer it again.
func (s *Store) Load(prompt []backend.Token, n int) ([]byte, error) {
	if n <= 0 || n > len(prompt) {
		return nil, ErrNotFound
	}
	sum := tokensSum(prompt[:n])
	s.mu.Lock()
	e := s.own[sum]
	if e == nil || !e.ready || e.n != n {
		s.mu.Unlock()
		return nil, ErrNotFound
	}
	path := filepath.Join(s.dir, e.name)
	s.mu.Unlock()

	state, err := s.read(path, sum, n)
	if err != nil {
		missing := errors.Is(err, fs.ErrNotExist)
		var bad *corruptError
		if missing || errors.As(err, &bad) {
			s.mu.Lock()
			if s.own[sum] == e {
				s.dropLocked(e, !missing)
			}
			s.mu.Unlock()
			if !missing {
				s.log.Warn("prompt cache: a stored state is corrupted and was removed", "file", e.name, "err", err)
			}
		}
		return nil, err
	}
	s.mu.Lock()
	if s.own[sum] == e {
		s.clock++
		e.used = s.clock
	}
	s.mu.Unlock()
	now := time.Now()
	_ = os.Chtimes(path, now, now)
	return state, nil
}

type corruptError struct{ why string }

func (e *corruptError) Error() string { return "promptcache: " + e.why }

// read verifies and returns the state in path.
func (s *Store) read(path string, sum [32]byte, n int) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	br := bufio.NewReaderSize(f, 64<<10)
	var raw [headerSize]byte
	if _, err := io.ReadFull(br, raw[:]); err != nil {
		return nil, &corruptError{"truncated header"}
	}
	h, err := parseHeader(raw[:])
	if err != nil {
		return nil, &corruptError{err.Error()}
	}
	if h.fp != s.fp || h.n != n || h.tokens != sum {
		return nil, &corruptError{"the header is not the entry's"}
	}
	if info.Size() != h.size() {
		return nil, &corruptError{"truncated or extended"}
	}
	hash := sha256.New()
	if _, err := io.CopyN(hash, br, 4*int64(n)); err != nil {
		return nil, &corruptError{"truncated token ids"}
	}
	var got [32]byte
	hash.Sum(got[:0])
	if got != sum {
		return nil, &corruptError{"the token ids do not match their hash"}
	}
	state := make([]byte, h.stateLen)
	if _, err := io.ReadFull(br, state); err != nil {
		return nil, &corruptError{"truncated state"}
	}
	if crc32.Checksum(state, crcTable) != h.crc {
		return nil, &corruptError{"the state does not match its checksum"}
	}
	return state, nil
}

// Save stores state as the entry for tokens. The state is handed over (the
// caller must not change it) and written by the store's goroutine, so the
// call itself costs one hash of the tokens; ErrBusy says the writer is behind
// and the state was dropped. The entry is visible to Has at once and to
// Lookup when the file is in place.
func (s *Store) Save(tokens []backend.Token, state []byte) error {
	if len(tokens) == 0 || len(state) == 0 {
		return errors.New("promptcache: nothing to save")
	}
	sum := tokensSum(tokens)
	size := headerSize + 4*int64(len(tokens)) + int64(len(state))
	if s.max > 0 && size > s.max {
		return ErrTooLarge
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	if _, ok := s.own[sum]; ok {
		return ErrExists
	}
	e := &entry{sum: sum, n: len(tokens), size: size, name: s.fileName(sum), own: true}
	j := job{e: e, tokens: append([]backend.Token(nil), tokens...), state: state}
	select {
	case s.jobs <- j:
	default:
		return ErrBusy
	}
	s.own[sum] = e
	s.pending++
	return nil
}

// Flush blocks until every state handed to Save is on disk (or failed).
func (s *Store) Flush() {
	s.mu.Lock()
	for s.pending > 0 {
		s.cond.Wait()
	}
	s.mu.Unlock()
}

// Close writes what is queued and stops the writer. Save fails afterwards;
// Lookup, Has and Load keep working on what is indexed.
func (s *Store) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	close(s.jobs)
	s.mu.Unlock()
	<-s.done
	return nil
}

// Stats reads the index.
func (s *Store) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := Stats{Bytes: s.total, Unusable: len(s.other), WriteErrors: s.writeErrors}
	for _, e := range s.own {
		if e.ready {
			st.Entries++
		}
	}
	return st
}

func (s *Store) writer() {
	defer close(s.done)
	for j := range s.jobs {
		err := s.write(j)
		s.mu.Lock()
		s.pending--
		if err != nil {
			if s.own[j.e.sum] == j.e {
				delete(s.own, j.e.sum)
			}
			s.writeErrors++
		} else {
			j.e.ready = true
			s.clock++
			j.e.used = s.clock
			s.total += j.e.size
			s.lenCnt[j.e.n]++
			if s.lenCnt[j.e.n] == 1 {
				s.rebuildLensLocked()
			}
			s.evictLocked()
		}
		s.cond.Broadcast()
		s.mu.Unlock()
		if err != nil {
			s.log.Warn("prompt cache: cannot write a state", "file", j.e.name, "err", err)
		}
	}
}

// write puts the entry on disk: a temporary file, synced, then renamed.
func (s *Store) write(j job) error {
	h := header{
		fp: s.fp, n: len(j.tokens), tokens: j.e.sum,
		stateLen: uint64(len(j.state)), crc: crc32.Checksum(j.state, crcTable),
	}
	f, err := os.CreateTemp(s.dir, tmpPrefix+"*"+tmpSuffix)
	if err != nil {
		return err
	}
	tmp := f.Name()
	fail := func(err error) error {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	bw := bufio.NewWriterSize(f, 64<<10)
	if _, err := bw.Write(h.marshal()); err != nil {
		return fail(err)
	}
	if err := tokenBytes(bw, j.tokens); err != nil {
		return fail(err)
	}
	if err := bw.Flush(); err != nil {
		return fail(err)
	}
	if _, err := f.Write(j.state); err != nil {
		return fail(err)
	}
	if err := f.Sync(); err != nil {
		return fail(err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, filepath.Join(s.dir, j.e.name)); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	// Make the rename durable too; where a directory cannot be synced the
	// file is still whole, only possibly absent after a power cut.
	if d, err := os.Open(s.dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
	return nil
}

// evictLocked deletes the least recently used files until the directory is
// under the cap. Pending entries are not on disk yet and are not candidates.
func (s *Store) evictLocked() {
	for s.max > 0 && s.total > s.max {
		var victim *entry
		for _, e := range s.own {
			if e.ready && (victim == nil || e.used < victim.used) {
				victim = e
			}
		}
		for _, e := range s.other {
			if victim == nil || e.used < victim.used {
				victim = e
			}
		}
		if victim == nil {
			return
		}
		s.log.Debug("prompt cache: evicting", "file", victim.name, "bytes", victim.size)
		s.dropLocked(victim, true)
	}
}

// dropLocked removes an entry from the index and, if asked, its file.
func (s *Store) dropLocked(e *entry, remove bool) {
	if e.own {
		delete(s.own, e.sum)
		if s.lenCnt[e.n]--; s.lenCnt[e.n] <= 0 {
			delete(s.lenCnt, e.n)
			s.rebuildLensLocked()
		}
	} else {
		delete(s.other, e.name)
	}
	s.total -= e.size
	if remove {
		if err := os.Remove(filepath.Join(s.dir, e.name)); err != nil && !errors.Is(err, fs.ErrNotExist) {
			s.log.Debug("prompt cache: cannot remove a file", "file", e.name, "err", err)
		}
	}
}
