package registry

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"regexp"
	"strings"

	"github.com/usunrise88/nanollm/internal/gguf"
)

// InventorySchema names the format of tensors/<id>.json.
const InventorySchema = "nanollm.inventory/v1"

// InventoryFile is tensors/<id>.json: the measured headers of every GGUF file
// of one catalog entry. It lives beside the catalog rather than in it because
// one model is several hundred tensors, and the catalog should stay a file a
// person reads.
type InventoryFile struct {
	Schema string           `json:"schema"`
	ID     string           `json:"id"`
	Files  []gguf.Inventory `json:"files"`
}

//go:embed tensors/*.json
var builtinInventories embed.FS

// attachInventories loads tensors/<id>.json for every entry that has one and
// binds it to the entry's files by sha256 and size. An inventory for a file the
// catalog does not pin, or one whose tensors the group table no longer
// classifies the same way, fails the catalog: a plan made from it would place
// bytes that are not the bytes being loaded.
func attachInventories(cat []Manifest, fsys embed.FS) error {
	entries, err := fsys.ReadDir("tensors")
	if err != nil {
		return err
	}
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := fsys.ReadFile(path.Join("tensors", e.Name()))
		if err != nil {
			return err
		}
		inv, err := ParseInventory(b)
		if err != nil {
			return fmt.Errorf("tensors/%s: %w", e.Name(), err)
		}
		if want := strings.TrimSuffix(e.Name(), ".json"); inv.ID != want {
			return fmt.Errorf("tensors/%s describes %q", e.Name(), inv.ID)
		}
		i := indexOf(cat, inv.ID)
		if i < 0 {
			return fmt.Errorf("tensors/%s: %q is not in the catalog", e.Name(), inv.ID)
		}
		if err := cat[i].Attach(inv); err != nil {
			return fmt.Errorf("tensors/%s: %w", e.Name(), err)
		}
	}
	return nil
}

func indexOf(cat []Manifest, id string) int {
	for i, m := range cat {
		if m.ID == id {
			return i
		}
	}
	return -1
}

// ParseInventory decodes and checks an inventory file.
func ParseInventory(b []byte) (InventoryFile, error) {
	var inv InventoryFile
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&inv); err != nil {
		return inv, fmt.Errorf("parse inventory: %w", err)
	}
	if inv.Schema != InventorySchema {
		return inv, fmt.Errorf("inventory schema %q, want %q", inv.Schema, InventorySchema)
	}
	for _, f := range inv.Files {
		if err := f.Check(); err != nil {
			return inv, fmt.Errorf("%s file: %w", f.Role, err)
		}
	}
	return inv, nil
}

// Attach binds an inventory to the manifest: every inventoried file must be a
// file of the entry with the same sha256 and size, and the model file must be
// among them. Tensors is filled from it.
func (m *Manifest) Attach(inv InventoryFile) error {
	if len(m.Tensors) > 0 {
		return fmt.Errorf("%s: tensors are listed in the catalog and inventoried too", m.ID)
	}
	var tensors []Tensor
	seen := map[string]bool{}
	for _, fi := range inv.Files {
		f, ok := m.File(fi.Role)
		if !ok {
			return fmt.Errorf("%s has no %s file", m.ID, fi.Role)
		}
		if seen[fi.Role] {
			return fmt.Errorf("%s file inventoried twice", fi.Role)
		}
		seen[fi.Role] = true
		if fi.SHA256 != f.SHA256 || fi.Size != f.Size {
			return fmt.Errorf("%s file: inventory is of sha256 %s (%d bytes), the catalog pins %s (%d bytes)",
				fi.Role, fi.SHA256, fi.Size, f.SHA256, f.Size)
		}
		for _, t := range fi.Tensors {
			tensors = append(tensors, Tensor{Role: fi.Role, Tensor: t})
		}
	}
	if !seen[RoleModel] {
		return fmt.Errorf("%s: the inventory has no model file", m.ID)
	}
	m.Inventory = inv.Files
	m.Tensors = tensors
	return nil
}

// FileInventory returns the inventory of the file with the given role.
func (m Manifest) FileInventory(role string) (gguf.Inventory, bool) {
	for _, fi := range m.Inventory {
		if fi.Role == role {
			return fi, true
		}
	}
	return gguf.Inventory{}, false
}

// InspectFile parses the header of a local GGUF and inventories it. The
// sha256 is computed over the whole file, which binds the inventory to these
// exact bytes; for a 4 GB model that is the slow part. Unknown tensors are
// returned as a warning alongside a usable inventory.
func InspectFile(p, role string) (inv gguf.Inventory, unknown error, err error) {
	f, err := gguf.Open(p)
	if err != nil {
		return inv, nil, err
	}
	inv, unknown = gguf.BuildInventory(f)
	if inv.Tensors == nil && unknown != nil {
		return inv, nil, unknown // hyperparameters, not tensors, failed
	}
	inv.Role = role
	if inv.SHA256, err = fileSHA256(p); err != nil {
		return inv, nil, err
	}
	return inv, unknown, nil
}

func fileSHA256(p string) (string, error) {
	fh, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer fh.Close()
	h := sha256.New()
	if _, err := io.Copy(h, fh); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// MatchFile finds the catalog entry and role whose pinned sha256 is this one.
func MatchFile(cat []Manifest, sha string) (Manifest, string, bool) {
	for _, m := range cat {
		for _, f := range m.Files {
			if f.SHA256 == sha {
				return m, f.Role, true
			}
		}
	}
	return Manifest{}, "", false
}

var (
	scalarArray = regexp.MustCompile(`\[(\s*(-?[0-9]+|true|false),?)+\s*\]`)
	whitespace  = regexp.MustCompile(`\s+`)
)

// EncodeInventory writes an inventory file with one tensor per line, so a
// change of quantization in a new revision reads as a diff of the tensors it
// touched rather than of every field of every tensor.
func EncodeInventory(w io.Writer, inv InventoryFile) error {
	inv.Schema = InventorySchema
	var b bytes.Buffer
	fmt.Fprintf(&b, "{\n  \"schema\": %q,\n  \"id\": %q,\n  \"files\": [\n", inv.Schema, inv.ID)
	for i, fi := range inv.Files {
		tensors := fi.Tensors
		fi.Tensors = nil
		head, err := json.MarshalIndent(fi, "    ", "  ")
		if err != nil {
			return err
		}
		// Per-layer hyperparameters are one line each, not one per layer.
		head = scalarArray.ReplaceAllFunc(head, func(m []byte) []byte {
			return whitespace.ReplaceAll(m, nil)
		})
		var list bytes.Buffer
		list.WriteString(`"tensors": [` + "\n")
		for j, t := range tensors {
			line, err := json.Marshal(t)
			if err != nil {
				return err
			}
			list.WriteString("        ")
			list.Write(line)
			if j < len(tensors)-1 {
				list.WriteByte(',')
			}
			list.WriteByte('\n')
		}
		list.WriteString("      ]")
		// With the tensors taken out, the only null in the object is theirs.
		b.WriteString("    ")
		b.Write(bytes.Replace(head, []byte(`"tensors": null`), list.Bytes(), 1))
		if i < len(inv.Files)-1 {
			b.WriteByte(',')
		}
		b.WriteByte('\n')
	}
	b.WriteString("  ]\n}\n")
	_, err := w.Write(b.Bytes())
	return err
}
