package registry

import (
	"bytes"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/gguf"
)

var updateInventory = flag.Bool("update-inventory", false,
	"rewrite tensors/<id>.json from the files in $NANOLLM_TEST_MODELS")

// TestInventoryMatchesFiles re-measures the pinned files when they are at
// hand ($NANOLLM_TEST_MODELS holds them under their repository file names)
// and compares the result with the committed inventory. With
// -update-inventory it writes the inventory instead; that is how
// tensors/<id>.json is produced at catalog build.
func TestInventoryMatchesFiles(t *testing.T) {
	dir := os.Getenv("NANOLLM_TEST_MODELS")
	if dir == "" {
		t.Skip("NANOLLM_TEST_MODELS is not set; the committed inventory is checked against itself only")
	}
	cat, err := parseCatalog(builtinCatalog)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range cat {
		inv := InventoryFile{Schema: InventorySchema, ID: m.ID}
		for _, f := range m.Files {
			p := filepath.Join(dir, filepath.Base(f.Path))
			if _, err := os.Stat(p); err != nil {
				t.Skipf("%s: %v", m.ID, err)
			}
			fi, unknown, err := InspectFile(p, f.Role)
			if err != nil {
				t.Fatal(err)
			}
			if unknown != nil {
				t.Fatalf("%s: %v", p, unknown)
			}
			inv.Files = append(inv.Files, fi)
		}
		var b bytes.Buffer
		if err := EncodeInventory(&b, inv); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join("tensors", m.ID+".json")
		if *updateInventory {
			if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		have, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(have, b.Bytes()) {
			t.Errorf("%s differs from the measured files; rerun with -update-inventory", path)
		}
	}
}

func TestBuiltinInventory(t *testing.T) {
	cat, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := Find(cat, "gemma-4-e4b-qat")
	if !ok {
		t.Fatal("no E4B")
	}
	model, ok := m.FileInventory(RoleModel)
	if !ok {
		t.Fatal("E4B has no model inventory")
	}
	drafter, ok := m.FileInventory(RoleDrafter)
	if !ok {
		t.Fatal("E4B has no drafter inventory")
	}
	if len(m.Tensors) != len(model.Tensors)+len(drafter.Tensors) {
		t.Errorf("%d tensors attached, %d inventoried", len(m.Tensors), len(model.Tensors)+len(drafter.Tensors))
	}
	// What SPEC §4 says about E4B, now as measured facts.
	h := model.HParams
	if h.BlockCount != 42 || h.SharedKVLayers != 18 || h.SlidingWindow != 512 || !h.TiedOutput {
		t.Errorf("hparams: %+v", h)
	}
	if model.Groups[gguf.G1] == 0 || drafter.Groups[gguf.G6] != drafter.Total() {
		t.Errorf("groups: model %v drafter %v", model.Groups, drafter.Groups)
	}
	for _, tn := range m.Tensors {
		if tn.Group == "" {
			t.Errorf("%s: unclassified", tn.Name)
		}
	}
}

func TestAttachRejects(t *testing.T) {
	b, err := gguf.TinyGemma4().Bytes()
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	fi, unknown, err := InspectFile(p, RoleModel)
	if err != nil || unknown != nil {
		t.Fatal(err, unknown)
	}
	inv := InventoryFile{Schema: InventorySchema, ID: "tiny", Files: []gguf.Inventory{fi}}
	m := Manifest{ID: "tiny", Files: []File{{Role: RoleModel, SHA256: fi.SHA256, Size: fi.Size}}}
	if err := m.Attach(inv); err != nil {
		t.Fatal(err)
	}
	if len(m.Tensors) != len(fi.Tensors) || m.Tensors[0].Role != RoleModel {
		t.Errorf("attached %d tensors", len(m.Tensors))
	}

	other := Manifest{ID: "tiny", Files: []File{{Role: RoleModel, SHA256: strings.Repeat("0", 64), Size: fi.Size}}}
	if err := other.Attach(inv); err == nil {
		t.Error("an inventory of other bytes was attached")
	}

	// The encoded form round-trips and is checked on the way in.
	var buf bytes.Buffer
	if err := EncodeInventory(&buf, inv); err != nil {
		t.Fatal(err)
	}
	back, err := ParseInventory(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(back.Files) != 1 || len(back.Files[0].Tensors) != len(fi.Tensors) {
		t.Errorf("round trip lost tensors")
	}
	edited := bytes.Replace(buf.Bytes(), []byte(`"group":"G3"`), []byte(`"group":"G2"`), 1)
	if _, err := ParseInventory(edited); err == nil {
		t.Error("a hand-edited group passed")
	}
}
