package registry

import (
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/usunrise88/nanollm/internal/gguf"
)

// The layouts of a model file (OPT-7, docs/AUTOTUNE.md): the file as
// published, and the copy `nanollm models relayout` writes beside it.
const (
	LayoutOriginal = gguf.LayoutOriginal
	LayoutNano     = gguf.LayoutNano
)

// Relaid is the nano-layout copy of an entry's model file, checked: its path,
// its sidecar, and the inventory of its own header. The catalog's inventory
// is bound to the published file's offsets; this one is what the planner, the
// range-lock and the streamer must use when the copy is what loads.
type Relaid struct {
	Path      string
	Sidecar   gguf.Sidecar
	Inventory gguf.Inventory
}

type relaidKey struct {
	path   string
	size   int64
	mtime  time.Time
	scTime time.Time
}

// NanoPath is where the relaid copy of the entry's model file lives: beside
// the original, <file>.nano.gguf, with <file>.nano.json.
func (s *Store) NanoPath(m Manifest) (string, error) {
	p, err := s.FilePath(m, RoleModel)
	if err != nil {
		return "", err
	}
	return gguf.NanoPath(p), nil
}

// Relaid returns the checked relaid copy of the entry's model file. The check
// is the cheap one: the sidecar must be of this layout version and of the
// pinned file (sha256 and size), the copy the size and modification time the
// sidecar recorded, and its tensors the catalog's. The bytes are not rehashed
// (`models relayout --verify` does that). Every refusal says what to run.
func (s *Store) Relaid(m Manifest) (Relaid, error) {
	nano, err := s.NanoPath(m)
	if err != nil {
		return Relaid{}, err
	}
	sidecar := gguf.SidecarPath(nano)
	fix := func(why error, flag string) error {
		return fmt.Errorf("model %s: layout nano is not usable: %w; run `nanollm models relayout %s%s`", m.ID, why, m.ID, flag)
	}
	sc, err := gguf.ReadSidecar(sidecar)
	if errors.Is(err, os.ErrNotExist) {
		return Relaid{}, fix(fmt.Errorf("%s is not there", sidecar), "")
	}
	if err != nil {
		return Relaid{}, fix(err, " --force")
	}
	if err := sc.Verify(nano); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Relaid{}, fix(fmt.Errorf("%s is not there", nano), " --force")
		}
		return Relaid{}, fix(err, " --force")
	}
	f, _ := m.File(RoleModel)
	if sc.Source.SHA256 != f.SHA256 || sc.Source.Size != f.Size {
		return Relaid{}, fix(fmt.Errorf("it was made from a file of sha256 %.12s (%d bytes), the catalog pins %.12s (%d bytes)",
			sc.Source.SHA256, sc.Source.Size, f.SHA256, f.Size), " --force")
	}

	st, err := os.Stat(nano)
	if err != nil {
		return Relaid{}, err
	}
	sst, err := os.Stat(sidecar)
	if err != nil {
		return Relaid{}, err
	}
	key := relaidKey{nano, st.Size(), st.ModTime(), sst.ModTime()}
	s.relaidMu.Lock()
	defer s.relaidMu.Unlock()
	if r, ok := s.relaid[key]; ok {
		return r, nil
	}
	gf, err := gguf.Open(nano)
	if err != nil {
		return Relaid{}, fix(err, " --force")
	}
	inv, unknown := gguf.BuildInventory(gf)
	if unknown != nil || inv.Tensors == nil {
		return Relaid{}, fix(errors.Join(unknown, errors.New("its header cannot be inventoried")), " --force")
	}
	inv.Role, inv.SHA256, inv.Layout = RoleModel, sc.Result.SHA256, LayoutNano
	if err := inv.Check(); err != nil {
		return Relaid{}, fix(err, " --force")
	}
	if want, ok := m.FileInventory(RoleModel); ok {
		if err := inv.SameTensors(want); err != nil {
			return Relaid{}, fix(fmt.Errorf("its tensors are not the catalog's: %w", err), " --force")
		}
	}
	r := Relaid{Path: nano, Sidecar: sc, Inventory: inv}
	if s.relaid == nil {
		s.relaid = map[relaidKey]Relaid{}
	}
	s.relaid[key] = r
	return r, nil
}

// WithRelaid returns the manifest as the planner should see it when the
// relaid copy loads: the inventory of the model file replaced by the copy's.
// The manifest is not changed.
func (m Manifest) WithRelaid(r Relaid) Manifest {
	files := make([]gguf.Inventory, 0, len(m.Inventory)+1)
	replaced := false
	for _, fi := range m.Inventory {
		if fi.Role == RoleModel {
			fi, replaced = r.Inventory, true
		}
		files = append(files, fi)
	}
	if !replaced {
		files = append(files, r.Inventory)
	}
	m.Inventory = files
	m.Tensors = nil
	for _, fi := range files {
		for _, t := range fi.Tensors {
			m.Tensors = append(m.Tensors, Tensor{Role: fi.Role, Tensor: t})
		}
	}
	return m
}
