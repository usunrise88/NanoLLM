package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/gguf"
)

// relaidFixture is a store holding the toy gemma4 as a catalog entry "tiny",
// pinned by the sha256 of its bytes, with the inventory the catalog would
// carry. The relaid copy is not written yet.
func relaidFixture(t *testing.T, w *gguf.Writer) (*Store, Manifest, string) {
	t.Helper()
	b, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	m := Manifest{ID: "tiny", Arch: "gemma4", Files: []File{{
		Role: RoleModel, Repo: "x/tiny", Path: "tiny.gguf", Revision: strings.Repeat("a", 40),
		Size: int64(len(b)), SHA256: hex.EncodeToString(sum[:]),
	}}}
	s := &Store{Root: t.TempDir()}
	p, err := s.FilePath(m, RoleModel)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	inv, unknown, err := InspectFile(p, RoleModel)
	if err != nil || unknown != nil {
		t.Fatal(err, unknown)
	}
	m.Inventory = []gguf.Inventory{inv}
	return s, m, p
}

func relayoutOf(t *testing.T, m Manifest, src string, o gguf.RelayoutOptions) gguf.Sidecar {
	t.Helper()
	f, _ := m.File(RoleModel)
	if o.SourceSHA256 == "" {
		o.SourceSHA256 = f.SHA256
	}
	sc, err := gguf.Relayout(context.Background(), src, gguf.NanoPath(src), o)
	if err != nil {
		t.Fatal(err)
	}
	return sc
}

func TestRelaidNeedsTheCopy(t *testing.T) {
	s, m, _ := relaidFixture(t, gguf.TinyGemma4().FillPattern())
	_, err := s.Relaid(m)
	if err == nil || !strings.Contains(err.Error(), "`nanollm models relayout tiny`") || !strings.Contains(err.Error(), "layout nano") {
		t.Fatalf("no sidecar: %v", err)
	}
	// The file without its sidecar is no better.
	nano, _ := s.NanoPath(m)
	if err := os.WriteFile(nano, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Relaid(m); err == nil {
		t.Error("a nano file with no sidecar was accepted")
	}
}

func TestRelaid(t *testing.T) {
	s, m, src := relaidFixture(t, gguf.TinyGemma4().FillPattern())
	sc := relayoutOf(t, m, src, gguf.RelayoutOptions{Alignment: 4096})
	nano, _ := s.NanoPath(m)
	if filepath.Base(nano) != "tiny.nano.gguf" {
		t.Errorf("nano path %s", nano)
	}
	r, err := s.Relaid(m)
	if err != nil {
		t.Fatal(err)
	}
	inv := r.Inventory
	if r.Path != nano || inv.Layout != LayoutNano || inv.Role != RoleModel || inv.SHA256 != sc.Result.SHA256 ||
		inv.Alignment != 4096 || inv.Size != sc.Result.Size {
		t.Errorf("relaid %+v, inventory %s/%s %s alignment %d", r.Path, inv.Role, inv.Layout, inv.SHA256, inv.Alignment)
	}
	if err := inv.Check(); err != nil {
		t.Error(err)
	}
	if err := inv.SameTensors(m.Inventory[0]); err != nil {
		t.Error(err)
	}
	// The nano file's own offsets, not the catalog's.
	moved := 0
	for _, tn := range inv.Tensors {
		if orig, _ := m.Inventory[0].Find(tn.Name); orig.Offset != tn.Offset {
			moved++
		}
		if tn.Offset%4096 != 0 {
			t.Errorf("%s at %d", tn.Name, tn.Offset)
		}
	}
	if moved == 0 {
		t.Error("no tensor moved")
	}
	// The header is parsed once while the files are as they were.
	again, err := s.Relaid(m)
	if err != nil || len(again.Inventory.Tensors) != len(inv.Tensors) || len(s.relaid) != 1 {
		t.Errorf("second call: %v, %d cached", err, len(s.relaid))
	}

	// The manifest the planner sees carries the nano inventory; the catalog's is untouched.
	nm := m.WithRelaid(r)
	if nm.Inventory[0].Layout != LayoutNano || m.Inventory[0].Layout != "" {
		t.Errorf("layouts after WithRelaid: %q, original %q", nm.Inventory[0].Layout, m.Inventory[0].Layout)
	}
	if len(nm.Tensors) != len(inv.Tensors) || nm.Tensors[0].Role != RoleModel || nm.Tensors[0].Name != inv.Tensors[0].Name {
		t.Errorf("tensors of the manifest: %d, first %+v", len(nm.Tensors), nm.Tensors[0])
	}
	if len(m.Tensors) != 0 && m.Tensors[0].Offset == nm.Tensors[0].Offset && m.Tensors[0].Name == nm.Tensors[0].Name {
		t.Error("the original manifest's tensors were rewritten")
	}
}

func TestRelaidIsStale(t *testing.T) {
	s, m, src := relaidFixture(t, gguf.TinyGemma4().FillPattern())
	sc := relayoutOf(t, m, src, gguf.RelayoutOptions{Alignment: 4096})
	nano, _ := s.NanoPath(m)
	sidecar := gguf.SidecarPath(nano)
	want := "`nanollm models relayout tiny --force`"

	t.Run("another revision of the file", func(t *testing.T) {
		m2 := m
		m2.Files = slices.Clone(m.Files)
		m2.Files[0].SHA256 = strings.Repeat("b", 64)
		if _, err := s.Relaid(m2); err == nil || !strings.Contains(err.Error(), "made from a file of sha256") || !strings.Contains(err.Error(), want) {
			t.Errorf("%v", err)
		}
	})
	t.Run("touched", func(t *testing.T) {
		old, _ := os.Stat(nano)
		defer func() { _ = os.Chtimes(nano, old.ModTime(), old.ModTime()) }()
		later := old.ModTime().Add(time.Hour)
		if err := os.Chtimes(nano, later, later); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Relaid(m); err == nil || !strings.Contains(err.Error(), "modified") {
			t.Errorf("%v", err)
		}
	})
	t.Run("another layout version", func(t *testing.T) {
		bad := sc
		bad.LayoutVersion++
		if err := gguf.WriteSidecar(sidecar, bad); err != nil {
			t.Fatal(err)
		}
		defer func() { _ = gguf.WriteSidecar(sidecar, sc) }()
		if _, err := s.Relaid(m); err == nil || !strings.Contains(err.Error(), "version") {
			t.Errorf("%v", err)
		}
	})
	t.Run("the copy is gone", func(t *testing.T) {
		keep, _ := os.ReadFile(nano)
		st, _ := os.Stat(nano)
		if err := os.Remove(nano); err != nil {
			t.Fatal(err)
		}
		defer func() {
			_ = os.WriteFile(nano, keep, 0o644)
			_ = os.Chtimes(nano, st.ModTime(), st.ModTime())
		}()
		if _, err := s.Relaid(m); err == nil || !strings.Contains(err.Error(), "is not there") {
			t.Errorf("%v", err)
		}
	})
	if _, err := s.Relaid(m); err != nil {
		t.Errorf("after the cases the copy is as it was: %v", err)
	}
}

// A copy made from the pinned file's sha256 but holding other tensors is not
// taken: the plan would place bytes that are not the catalog's.
func TestRelaidMustHoldTheCatalogsTensors(t *testing.T) {
	s, m, src := relaidFixture(t, gguf.TinyGemma4().FillPattern())
	other := gguf.TinyGemma4().Tensor("blk.0.extra_norm.weight", gguf.TypeF32, []int64{32}, nil)
	b, _ := other.FillPattern().Bytes()
	op := filepath.Join(t.TempDir(), "other.gguf")
	if err := os.WriteFile(op, b, 0o644); err != nil {
		t.Fatal(err)
	}
	f, _ := m.File(RoleModel)
	if _, err := gguf.Relayout(context.Background(), op, gguf.NanoPath(src), gguf.RelayoutOptions{SourceSHA256: f.SHA256}); err != nil {
		t.Fatal(err)
	}
	// Size is the catalog's; make the sidecar agree so only the tensors differ.
	nano, _ := s.NanoPath(m)
	sc, _ := gguf.ReadSidecar(gguf.SidecarPath(nano))
	sc.Source.Size = f.Size
	if err := gguf.WriteSidecar(gguf.SidecarPath(nano), sc); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Relaid(m); err == nil || !strings.Contains(err.Error(), "not the catalog's") {
		t.Errorf("%v", err)
	}
}
