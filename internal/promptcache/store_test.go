package promptcache

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
)

var fpA = NewFingerprint("model=a", "n_ctx=4096")

// toks makes n token ids from a seed; the same seed gives lists that share
// a prefix, and different seeds differ in the first token.
func toks(seed, n int) []backend.Token {
	out := make([]backend.Token, n)
	for i := range out {
		out[i] = backend.Token((seed*7919 + i*31) % 5000)
	}
	out[0] = backend.Token(10000 + seed)
	return out
}

func stateOf(seed, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(seed*13 + i*7)
	}
	return b
}

func open(t *testing.T, dir string, fp Fingerprint, max int64) *Store {
	t.Helper()
	s, err := Open(Options{Dir: dir, Fingerprint: fp, MaxBytes: max})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func save(t *testing.T, s *Store, tokens []backend.Token, state []byte) {
	t.Helper()
	if err := s.Save(tokens, state); err != nil {
		t.Fatal(err)
	}
	s.Flush()
}

func files(t *testing.T, dir string) []string {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, e := range ents {
		out = append(out, e.Name())
	}
	return out
}

func dirBytes(t *testing.T, dir string) int64 {
	t.Helper()
	var n int64
	for _, name := range files(t, dir) {
		fi, err := os.Stat(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		n += fi.Size()
	}
	return n
}

func TestLookupFindsTheLongestPrefix(t *testing.T) {
	s := open(t, t.TempDir(), fpA, 0)
	base := toks(1, 300)
	save(t, s, base[:100], stateOf(1, 64))
	save(t, s, base[:200], stateOf(2, 64))
	save(t, s, toks(2, 150), stateOf(3, 64))

	for _, tc := range []struct {
		name   string
		prompt []backend.Token
		want   int
	}{
		{"longest of two prefixes", base, 200},
		{"just past the longer", base[:201], 200},
		{"only the shorter fits", base[:150], 100},
		// A state of the whole prompt is no use: one token must still be
		// evaluated, so the next best is offered.
		{"the prompt itself", base[:200], 100},
		{"one token of the shorter", base[:101], 100},
		{"the shorter itself", base[:100], 0},
		{"unrelated", toks(9, 400), 0},
		{"diverges inside the prefix", append(slices.Clone(base[:199]), 4999, 4998, 4997), 100},
	} {
		n, ok := s.Lookup(tc.prompt)
		if n != tc.want || ok != (tc.want > 0) {
			t.Errorf("%s: got %d, %v; want %d", tc.name, n, ok, tc.want)
		}
	}
	if !s.Has(base[:100]) || s.Has(base[:101]) || s.Has(nil) {
		t.Error("Has is wrong")
	}

	state, err := s.Load(base, 200)
	if err != nil || !bytes.Equal(state, stateOf(2, 64)) {
		t.Fatalf("load: %v, %d bytes", err, len(state))
	}
	if _, err := s.Load(base, 150); !errors.Is(err, ErrNotFound) {
		t.Fatalf("a length with no entry: %v", err)
	}
	if _, err := s.Load(base, 0); !errors.Is(err, ErrNotFound) {
		t.Fatalf("zero tokens: %v", err)
	}
}

func TestSurvivesARestart(t *testing.T) {
	dir := t.TempDir()
	first := open(t, dir, fpA, 0)
	prompt := toks(3, 500)
	save(t, first, prompt[:400], stateOf(5, 1000))
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	second := open(t, dir, fpA, 0)
	if n, ok := second.Lookup(prompt); !ok || n != 400 {
		t.Fatalf("after a restart: %d, %v", n, ok)
	}
	got, err := second.Load(prompt, 400)
	if err != nil || !bytes.Equal(got, stateOf(5, 1000)) {
		t.Fatalf("load after a restart: %v", err)
	}
	if st := second.Stats(); st.Entries != 1 || st.Unusable != 0 || st.Bytes != dirBytes(t, dir) {
		t.Fatalf("stats %+v, directory holds %d bytes", st, dirBytes(t, dir))
	}
}

func TestCloseWritesWhatIsQueued(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir, fpA, 0)
	if err := s.Save(toks(1, 50), stateOf(1, 10<<10)); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if len(files(t, dir)) != 1 {
		t.Fatalf("files %v", files(t, dir))
	}
	if err := s.Save(toks(2, 50), stateOf(1, 10)); !errors.Is(err, ErrClosed) {
		t.Fatalf("save after close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}

func TestSaveRefusals(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir, fpA, 1000)
	if err := s.Save(nil, stateOf(1, 10)); err == nil {
		t.Error("no tokens")
	}
	if err := s.Save(toks(1, 5), nil); err == nil {
		t.Error("no state")
	}
	if err := s.Save(toks(1, 5), stateOf(1, 2000)); !errors.Is(err, ErrTooLarge) {
		t.Errorf("a state over the whole cap: %v", err)
	}
	save(t, s, toks(1, 5), stateOf(1, 100))
	if err := s.Save(toks(1, 5), stateOf(1, 100)); !errors.Is(err, ErrExists) {
		t.Errorf("a second save: %v", err)
	}
	if len(files(t, dir)) != 1 {
		t.Errorf("files %v", files(t, dir))
	}
}

func TestAnotherFingerprintIsNeverServed(t *testing.T) {
	dir := t.TempDir()
	a := open(t, dir, fpA, 0)
	prompt := toks(4, 200)
	save(t, a, prompt[:150], stateOf(1, 100))
	a.Close()

	b := open(t, dir, NewFingerprint("model=b", "n_ctx=4096"), 0)
	if n, ok := b.Lookup(prompt); ok {
		t.Fatalf("a state of another fingerprint was offered (%d tokens)", n)
	}
	if b.Has(prompt[:150]) {
		t.Fatal("Has sees another fingerprint's entry")
	}
	// Its own save of the same prompt sits beside the other's.
	save(t, b, prompt[:150], stateOf(2, 100))
	if got, err := b.Load(prompt, 150); err != nil || !bytes.Equal(got, stateOf(2, 100)) {
		t.Fatalf("own state: %v", err)
	}
	if len(files(t, dir)) != 2 || b.Stats().Unusable != 1 {
		t.Fatalf("files %v, stats %+v", files(t, dir), b.Stats())
	}
	// The first fingerprint still finds its own.
	a2 := open(t, dir, fpA, 0)
	if got, err := a2.Load(prompt, 150); err != nil || !bytes.Equal(got, stateOf(1, 100)) {
		t.Fatalf("first fingerprint's state: %v", err)
	}
}

// damage writes one stored state, applies f to its bytes and puts it back.
func damage(t *testing.T, dir string, f func([]byte) []byte) {
	t.Helper()
	names := files(t, dir)
	if len(names) != 1 {
		t.Fatalf("want one file, have %v", names)
	}
	path := filepath.Join(dir, names[0])
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, f(b), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestDamagedFilesAreIgnored(t *testing.T) {
	const nTok = 100
	prompt := toks(6, nTok+50)
	for _, tc := range []struct {
		name string
		f    func([]byte) []byte
		// served: the scan accepts the file, and Load must refuse it.
		served bool
	}{
		{"truncated state", func(b []byte) []byte { return b[:len(b)-10] }, false},
		{"truncated tokens", func(b []byte) []byte { return b[:headerSize+40] }, false},
		{"truncated header", func(b []byte) []byte { return b[:50] }, false},
		{"extended", func(b []byte) []byte { return append(b, 1, 2, 3) }, false},
		{"a flipped token", func(b []byte) []byte { b[headerSize+8] ^= 0xff; return b }, false},
		{"another version", func(b []byte) []byte { b[4] = 99; return b }, false},
		{"a flipped state byte", func(b []byte) []byte { b[len(b)-3] ^= 0xff; return b }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			s := open(t, dir, fpA, 0)
			save(t, s, prompt[:nTok], stateOf(7, 4096))
			s.Close()
			damage(t, dir, tc.f)

			s = open(t, dir, fpA, 0)
			n, ok := s.Lookup(prompt)
			if ok != tc.served {
				t.Fatalf("lookup: %d, %v; the scan should serve it: %v", n, ok, tc.served)
			}
			if tc.served {
				if _, err := s.Load(prompt, n); err == nil {
					t.Fatal("a corrupted state was loaded")
				}
				if _, ok := s.Lookup(prompt); ok {
					t.Fatal("a corrupted state was offered again")
				}
				if len(files(t, dir)) != 0 {
					t.Fatalf("the corrupted file was left: %v", files(t, dir))
				}
				return
			}
			// Unusable files stay in place, count against the cap, and a new
			// save of the same prompt works beside them.
			if st := s.Stats(); st.Unusable != 1 || st.Entries != 0 {
				t.Fatalf("stats %+v", st)
			}
			save(t, s, prompt[:nTok], stateOf(7, 4096))
			if got, err := s.Load(prompt, nTok); err != nil || !bytes.Equal(got, stateOf(7, 4096)) {
				t.Fatalf("a new save beside the damaged file: %v", err)
			}
		})
	}
}

func TestFilesThatAreNotOursAreLeftAlone(t *testing.T) {
	dir := t.TempDir()
	junk := map[string][]byte{
		"notes.txt":                 []byte("hello"),
		"empty" + Ext:               nil,
		"short" + Ext:               []byte("NL"),
		"other-magic" + Ext:         bytes.Repeat([]byte("x"), 200),
		tmpPrefix + "1" + tmpSuffix: []byte("a write in progress"),
	}
	for name, b := range junk {
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// A temporary file from a crash long ago is cleaned up.
	old := filepath.Join(dir, tmpPrefix+"old"+tmpSuffix)
	if err := os.WriteFile(old, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	past := time.Now().Add(-time.Hour)
	if err := os.Chtimes(old, past, past); err != nil {
		t.Fatal(err)
	}

	// A cap of one byte would evict anything tracked; these are not.
	s := open(t, dir, fpA, 1)
	if st := s.Stats(); st.Bytes != 0 || st.Unusable != 0 {
		t.Fatalf("stats %+v", st)
	}
	for name := range junk {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Errorf("%s was touched: %v", name, err)
		}
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Errorf("a stale temporary file was kept: %v", err)
	}
}

func TestEvictionKeepsTheDirectoryUnderTheCap(t *testing.T) {
	dir := t.TempDir()
	const stateLen = 1000
	one := int64(headerSize + 4*50 + stateLen)
	s := open(t, dir, fpA, 3*one+one/2) // room for three
	prompts := make([][]backend.Token, 6)
	for i := range prompts {
		prompts[i] = toks(100+i, 50)
	}
	for i := 0; i < 3; i++ {
		save(t, s, prompts[i], stateOf(i, stateLen))
	}
	if st := s.Stats(); st.Entries != 3 {
		t.Fatalf("stats %+v", st)
	}
	// Use the oldest: the second becomes the least recently used.
	if _, err := s.Load(append(slices.Clone(prompts[0]), 1), 50); err != nil {
		t.Fatal(err)
	}
	save(t, s, prompts[3], stateOf(3, stateLen))
	if got := dirBytes(t, dir); got > 3*one+one/2 {
		t.Fatalf("directory holds %d bytes, cap %d", got, 3*one+one/2)
	}
	for i, want := range []bool{true, false, true, true} {
		if s.Has(prompts[i]) != want {
			t.Errorf("prompt %d kept: %v, want %v", i, !want, want)
		}
	}
	// An evicted entry is neither offered nor loadable.
	if n, ok := s.Lookup(append(slices.Clone(prompts[1]), 1)); ok {
		t.Errorf("the evicted prompt was offered (%d)", n)
	}
	if len(files(t, dir)) != 3 {
		t.Errorf("files %v", files(t, dir))
	}

	// A burst never leaves more than the cap behind.
	for i := 4; i < 6; i++ {
		save(t, s, prompts[i], stateOf(i, stateLen))
	}
	if got := dirBytes(t, dir); got > 3*one+one/2 {
		t.Fatalf("after a burst the directory holds %d bytes, cap %d", got, 3*one+one/2)
	}
	if !s.Has(prompts[5]) {
		t.Error("the newest entry was evicted")
	}
}

func TestEvictionOrderSurvivesARestart(t *testing.T) {
	dir := t.TempDir()
	const stateLen = 500
	one := int64(headerSize + 4*20 + stateLen)
	s := open(t, dir, fpA, 0)
	var prompts [][]backend.Token
	for i := 0; i < 3; i++ {
		prompts = append(prompts, toks(200+i, 20))
		save(t, s, prompts[i], stateOf(i, stateLen))
	}
	s.Close()
	// The files' modification times are the only memory of use: make the
	// middle one the oldest.
	base := time.Now().Add(-time.Hour)
	for i, age := range map[int]time.Duration{0: 20 * time.Minute, 1: 0, 2: 40 * time.Minute} {
		path := filepath.Join(dir, fpA.fileNameFor(prompts[i]))
		if err := os.Chtimes(path, base.Add(age), base.Add(age)); err != nil {
			t.Fatal(err)
		}
	}
	s2 := open(t, dir, fpA, 2*one+one/2) // opening evicts down to the cap
	if s2.Has(prompts[1]) || !s2.Has(prompts[0]) || !s2.Has(prompts[2]) {
		t.Fatalf("kept: %v %v %v; want the oldest-by-mtime (1) gone", s2.Has(prompts[0]), s2.Has(prompts[1]), s2.Has(prompts[2]))
	}
	if len(files(t, dir)) != 2 {
		t.Fatalf("files %v", files(t, dir))
	}
}

// fileNameFor is the name a store with this fingerprint gives a prompt.
func (f Fingerprint) fileNameFor(tokens []backend.Token) string {
	s := &Store{fp: f}
	return s.fileName(tokensSum(tokens))
}

func TestConcurrentUse(t *testing.T) {
	dir := t.TempDir()
	const stateLen = 2000
	one := int64(headerSize + 4*64 + stateLen)
	s := open(t, dir, fpA, 8*one)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 40; i++ {
				seed := g*1000 + i%12
				p := toks(seed, 64)
				switch err := s.Save(p, stateOf(seed, stateLen)); {
				case err == nil, errors.Is(err, ErrExists), errors.Is(err, ErrBusy):
				default:
					t.Errorf("save: %v", err)
				}
				longer := append(slices.Clone(p), 1, 2, 3)
				if n, ok := s.Lookup(longer); ok {
					if state, err := s.Load(longer, n); err == nil && !bytes.Equal(state, stateOf(seed, stateLen)) {
						t.Errorf("loaded the wrong state for seed %d", seed)
					} else if err != nil && !errors.Is(err, ErrNotFound) {
						t.Errorf("load: %v", err)
					}
				}
				s.Has(p)
				s.Stats()
			}
		}()
	}
	wg.Wait()
	s.Flush()
	if got := dirBytes(t, dir); got > 8*one {
		t.Fatalf("directory holds %d bytes, cap %d", got, 8*one)
	}
	if st := s.Stats(); st.Bytes != dirBytes(t, dir) || st.WriteErrors != 0 {
		t.Fatalf("index says %+v, directory %d bytes", st, dirBytes(t, dir))
	}
	for _, name := range files(t, dir) {
		if len(name) > 0 && name[0] == '.' {
			t.Errorf("a temporary file is left: %s", name)
		}
	}
}

func TestFingerprintDependsOnEveryField(t *testing.T) {
	base := NewFingerprint("a=1", "b=2")
	for _, other := range []Fingerprint{
		NewFingerprint("a=1", "b=3"),
		NewFingerprint("a=1"),
		NewFingerprint("a=1", "b=2", ""),
		NewFingerprint("a=1b=2"),
		NewFingerprint("b=2", "a=1"),
	} {
		if other == base {
			t.Fatal("two different field lists share a fingerprint")
		}
	}
	if NewFingerprint("a=1", "b=2") != base {
		t.Fatal("a fingerprint is not stable")
	}
	if len(base.String()) != 64 {
		t.Fatalf("hex %q", base.String())
	}
}

func TestOpenCreatesTheDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "a", "b")
	open(t, dir, fpA, 0)
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		t.Fatalf("directory: %v", err)
	}
	if _, err := Open(Options{Fingerprint: fpA}); err == nil {
		t.Error("no directory accepted")
	}
}

func TestAFailedWriteIsCountedAndForgotten(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "states")
	s := open(t, dir, fpA, 0)
	if err := os.RemoveAll(dir); err != nil {
		t.Fatal(err)
	}
	p := toks(1, 40)
	if err := s.Save(p, stateOf(1, 100)); err != nil {
		t.Fatal(err)
	}
	s.Flush()
	if st := s.Stats(); st.WriteErrors != 1 || st.Entries != 0 || st.Bytes != 0 {
		t.Fatalf("stats %+v", st)
	}
	if s.Has(p) {
		t.Fatal("an entry that was never written is still indexed")
	}
	if _, ok := s.Lookup(append(slices.Clone(p), 1)); ok {
		t.Fatal("an entry that was never written was offered")
	}
	// The directory comes back and the same prompt can be saved again.
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	save(t, s, p, stateOf(1, 100))
	if !s.Has(p) {
		t.Fatal("the retry was not stored")
	}
}
