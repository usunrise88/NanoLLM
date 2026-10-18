package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// Store is the on-disk model directory:
//
//	<root>/<id>/<revision>/<file>     the weights, by basename
//	<root>/<id>/<revision>/manifest.yaml  the catalog entry they came from
//	<root>/<id>/<revision>/.verified  sha256, size and mtime of each file
//
// The revision level lets a catalog update pin a new commit without
// destroying the weights a running server has mapped.
//
// A file is hashed once, when it is downloaded or by `models pull` on an
// existing copy. A load checks size and the stamp's mtime only: rehashing
// 4 GB on every start would make the server slow to come up for no gain,
// while a file changed in place would still be caught by its mtime.
type Store struct {
	Root          string
	Downloader    *Downloader
	StrictLicense bool
	// MaxConcurrent bounds parallel downloads across models; zero means one.
	MaxConcurrent int

	once    sync.Once
	sem     chan struct{}
	mu      sync.Mutex
	flights map[string]*flight
}

// Install states.
const (
	StateAbsent   = "absent"
	StatePartial  = "partial"
	StateComplete = "complete"
	StateCorrupt  = "corrupt"
)

// ManifestFile and stampFile sit beside the weights.
const (
	ManifestFile = "manifest.yaml"
	stampFile    = ".verified"
)

// Status is what is on disk for one entry.
type Status struct {
	ID    string       `json:"id"`
	State string       `json:"state"`
	Dir   string       `json:"dir"`
	Files []FileStatus `json:"files"`
}

type FileStatus struct {
	Role  string `json:"role"`
	Path  string `json:"path"`
	State string `json:"state"`
	Size  int64  `json:"size"`
	Have  int64  `json:"have"`
}

type stamp struct {
	SHA256 string    `json:"sha256"`
	Size   int64     `json:"size"`
	MTime  time.Time `json:"mtime"`
}

func (s *Store) init() {
	s.once.Do(func() {
		n := s.MaxConcurrent
		if n <= 0 {
			n = 1
		}
		s.sem = make(chan struct{}, n)
		s.flights = map[string]*flight{}
		if s.Downloader == nil {
			s.Downloader = &Downloader{}
		}
	})
}

// Dir is where an entry's files live. The revision of the model file names
// the directory: entries are pinned per commit.
func (s *Store) Dir(m Manifest) (string, error) {
	f, ok := m.File(RoleModel)
	if !ok {
		return "", fmt.Errorf("model %s has no model file", m.ID)
	}
	if !safeName(m.ID) || !safeName(f.Revision) {
		return "", fmt.Errorf("model %s: id or revision is not a plain directory name", m.ID)
	}
	return filepath.Join(s.Root, m.ID, f.Revision), nil
}

// FilePath is the local path of an entry's file with the given role.
func (s *Store) FilePath(m Manifest, role string) (string, error) {
	dir, err := s.Dir(m)
	if err != nil {
		return "", err
	}
	f, ok := m.File(role)
	if !ok {
		return "", fmt.Errorf("model %s has no %s file", m.ID, role)
	}
	name := path.Base(f.Path)
	if !safeName(name) {
		return "", fmt.Errorf("model %s: file name %q is not a plain name", m.ID, f.Path)
	}
	return filepath.Join(dir, name), nil
}

// safeName keeps catalog strings from walking out of the store.
func safeName(s string) bool {
	return s != "" && s != "." && s != ".." && filepath.Base(s) == s && !filepath.IsAbs(s)
}

// Status reports what is on disk without hashing anything.
func (s *Store) Status(m Manifest) (Status, error) {
	dir, err := s.Dir(m)
	if err != nil {
		return Status{}, err
	}
	st := Status{ID: m.ID, Dir: dir, State: StateComplete}
	stamps := readStamps(dir)
	found := false
	for _, f := range m.Files {
		p, err := s.FilePath(m, f.Role)
		if err != nil {
			return Status{}, err
		}
		fs := FileStatus{Role: f.Role, Path: p, Size: f.Size, State: StateAbsent}
		if info, err := os.Stat(p); err == nil {
			found = true
			fs.Have = info.Size()
			stp, ok := stamps[path.Base(p)]
			switch {
			case info.Size() != f.Size:
				fs.State = StateCorrupt
			case !ok || stp.SHA256 != f.SHA256 || !stp.MTime.Equal(info.ModTime()):
				// Present, but never verified as this pin, or touched since.
				fs.State = StateCorrupt
			default:
				fs.State = StateComplete
			}
		} else if info, err := os.Stat(p + ".part"); err == nil {
			found = true
			fs.Have = info.Size()
			fs.State = StatePartial
		}
		st.Files = append(st.Files, fs)
	}
	for _, fs := range st.Files {
		switch {
		case fs.State == StateCorrupt:
			st.State = StateCorrupt
		case fs.State != StateComplete && st.State != StateCorrupt:
			st.State = StatePartial
		}
	}
	if !found {
		st.State = StateAbsent
	}
	return st, nil
}

// Ready returns the model and drafter paths of a complete entry, or an error
// saying what is missing.
func (s *Store) Ready(m Manifest) (model, drafter string, err error) {
	st, err := s.Status(m)
	if err != nil {
		return "", "", err
	}
	if st.State != StateComplete {
		return "", "", fmt.Errorf("model %s is %s on disk; run `nanollm models pull %s`", m.ID, st.State, m.ID)
	}
	for _, f := range st.Files {
		switch f.Role {
		case RoleModel:
			model = f.Path
		case RoleDrafter:
			drafter = f.Path
		}
	}
	return model, drafter, nil
}

// Pull makes the entry complete: files already present are rehashed, missing
// or partial ones are downloaded. Concurrent pulls of one entry share the
// work; each caller gets its own progress callbacks, and the download stops
// only when every caller has gone.
func (s *Store) Pull(ctx context.Context, m Manifest, onProgress func(Progress)) error {
	s.init()
	if s.StrictLicense && m.Commercial != CommercialPermitted {
		return fmt.Errorf("model %s: commercial use is %s and registry.strict_license is on", m.ID, m.Commercial)
	}
	if !m.Pinned() {
		return fmt.Errorf("model %s is not pinned yet and cannot be downloaded", m.ID)
	}

	s.mu.Lock()
	fl, joined := s.flights[m.ID]
	if !joined {
		fctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
		fl = &flight{done: make(chan struct{}), cancel: cancel, subs: map[int]func(Progress){}}
		s.flights[m.ID] = fl
		go func() {
			err := s.pull(fctx, m, fl.publish)
			s.mu.Lock()
			delete(s.flights, m.ID)
			s.mu.Unlock()
			fl.finish(err)
		}()
	}
	id := fl.subscribe(onProgress)
	s.mu.Unlock()

	select {
	case <-fl.done:
		return fl.err
	case <-ctx.Done():
		fl.unsubscribe(id)
		return ctx.Err()
	}
}

func (s *Store) pull(ctx context.Context, m Manifest, onProgress func(Progress)) error {
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		return ctx.Err()
	}
	dir, err := s.Dir(m)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	stamps := readStamps(dir)
	for _, f := range m.Files {
		p, err := s.FilePath(m, f.Role)
		if err != nil {
			return err
		}
		if info, err := os.Stat(p); err == nil {
			// An existing copy is trusted only after hashing it.
			if info.Size() == f.Size {
				if sum, err := hashFile(ctx, p); err == nil && sum == f.SHA256 {
					stamps[path.Base(p)] = stamp{SHA256: sum, Size: f.Size, MTime: info.ModTime()}
					continue
				} else if ctx.Err() != nil {
					return ctx.Err()
				}
			}
			if err := os.Remove(p); err != nil {
				return err
			}
		}
		if err := s.Downloader.Fetch(ctx, f, p, onProgress); err != nil {
			return fmt.Errorf("model %s: %w", m.ID, err)
		}
		info, err := os.Stat(p)
		if err != nil {
			return err
		}
		stamps[path.Base(p)] = stamp{SHA256: f.SHA256, Size: f.Size, MTime: info.ModTime()}
	}
	if err := writeStamps(dir, stamps); err != nil {
		return err
	}
	return writeManifest(filepath.Join(dir, ManifestFile), m)
}

// Remove deletes every revision of an entry.
func (s *Store) Remove(id string) error {
	if !safeName(id) {
		return fmt.Errorf("%q is not a model id", id)
	}
	s.init()
	s.mu.Lock()
	_, busy := s.flights[id]
	s.mu.Unlock()
	if busy {
		return fmt.Errorf("model %s is downloading", id)
	}
	dir := filepath.Join(s.Root, id)
	if _, err := os.Stat(dir); errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("model %s is not installed", id)
	}
	return os.RemoveAll(dir)
}

// Installed lists entry ids with anything on disk, sorted.
func (s *Store) Installed() ([]string, error) {
	entries, err := os.ReadDir(s.Root)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() && safeName(e.Name()) {
			out = append(out, e.Name())
		}
	}
	sort.Strings(out)
	return out, nil
}

func readStamps(dir string) map[string]stamp {
	out := map[string]stamp{}
	b, err := os.ReadFile(filepath.Join(dir, stampFile))
	if err == nil {
		_ = json.Unmarshal(b, &out)
	}
	return out
}

func writeStamps(dir string, stamps map[string]stamp) error {
	b, err := json.MarshalIndent(stamps, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(dir, stampFile+".tmp")
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, stampFile))
}

func writeManifest(p string, m Manifest) error {
	m.Tensors = nil // the inventory ships in the binary; the copy here is for people
	b, err := yaml.Marshal(m)
	if err != nil {
		return err
	}
	header := "# Written by nanollm when this model was downloaded.\n# Edits are lost if the model is pulled again.\n"
	return os.WriteFile(p, append([]byte(header), b...), 0o644)
}

func hashFile(ctx context.Context, p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, ctxReader{ctx, f}); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// ctxReader lets a multi-gigabyte hash stop when its caller does.
type ctxReader struct {
	ctx context.Context
	r   io.Reader
}

func (c ctxReader) Read(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.Read(p)
}

// flight is one download shared by every caller pulling the same entry.
type flight struct {
	done   chan struct{}
	err    error
	cancel context.CancelFunc

	mu     sync.Mutex
	subs   map[int]func(Progress)
	nextID int
}

func (f *flight) subscribe(fn func(Progress)) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	id := f.nextID
	f.nextID++
	f.subs[id] = fn
	return id
}

// unsubscribe drops a caller; the last one leaving stops the download.
func (f *flight) unsubscribe(id int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.subs, id)
	if len(f.subs) == 0 {
		f.cancel()
	}
}

func (f *flight) publish(p Progress) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, fn := range f.subs {
		if fn != nil {
			fn(p)
		}
	}
}

func (f *flight) finish(err error) {
	f.err = err
	f.cancel()
	close(f.done)
}
