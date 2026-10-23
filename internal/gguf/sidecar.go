package gguf

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// SidecarSchema names the format of the .nano.json beside a relaid file.
const SidecarSchema = "nanollm.layout/v1"

// FileRef identifies one file of a relayout by size and content.
type FileRef struct {
	File   string `json:"file"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
	// MTime is the result's modification time when the sidecar was written;
	// a loader that finds another one knows the file was touched since.
	MTime time.Time `json:"mtime,omitzero"`
}

// Sidecar is the record beside a relaid file: what it was made from, how, and
// the digest of what came out. It makes a local file trustworthy without
// hosting it: the catalog pins the source's sha256, the sidecar binds the
// result to it, and the result is reproducible from the source alone.
type Sidecar struct {
	Schema        string   `json:"schema"`
	Layout        string   `json:"layout"`
	LayoutVersion int      `json:"layout_version"`
	Alignment     int64    `json:"alignment"`
	Tensors       int      `json:"tensors"`
	Source        FileRef  `json:"source"`
	Result        FileRef  `json:"result"`
	Sections      []Extent `json:"sections"`
	Tool          string   `json:"tool,omitempty"`
}

// NanoPath is the relaid file beside a model file: x.gguf becomes
// x.nano.gguf.
func NanoPath(model string) string {
	if base, ok := strings.CutSuffix(model, ".gguf"); ok {
		return base + ".nano.gguf"
	}
	return model + ".nano.gguf"
}

// SidecarPath is the sidecar of a relaid file: x.nano.gguf has x.nano.json.
func SidecarPath(nano string) string {
	if base, ok := strings.CutSuffix(nano, ".gguf"); ok {
		return base + ".json"
	}
	return nano + ".json"
}

// ReadSidecar reads and checks the sidecar at path. A missing file is an
// error wrapping os.ErrNotExist.
func ReadSidecar(path string) (Sidecar, error) {
	var sc Sidecar
	b, err := os.ReadFile(path)
	if err != nil {
		return sc, err
	}
	if err := json.Unmarshal(b, &sc); err != nil {
		return sc, fmt.Errorf("%s: %w", path, err)
	}
	if sc.Schema != SidecarSchema {
		return sc, fmt.Errorf("%s: schema %q, want %q", path, sc.Schema, SidecarSchema)
	}
	return sc, nil
}

// WriteSidecar writes sc to path through a temporary file and a rename.
func WriteSidecar(path string, sc Sidecar) error {
	sc.Schema = SidecarSchema
	b, err := json.MarshalIndent(sc, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// Verify is the check a loader makes before it trusts a relaid file: a
// layout this build reads, a file of the size the sidecar names, untouched
// since (same modification time). It reads no tensor data; the full check,
// which rehashes the file and compares every tensor with the source, is
// VerifyRelayout.
func (sc Sidecar) Verify(result string) error {
	switch {
	case sc.Layout != LayoutNano || sc.LayoutVersion != LayoutVersion:
		return fmt.Errorf("it is layout %s version %d, this build reads %s version %d",
			sc.Layout, sc.LayoutVersion, LayoutNano, LayoutVersion)
	case sc.Source.SHA256 == "" || sc.Result.SHA256 == "":
		return errors.New("the sidecar names no digests")
	}
	if err := CheckAlignment(sc.Alignment); err != nil {
		return err
	}
	st, err := os.Stat(result)
	if err != nil {
		return err
	}
	switch {
	case st.Size() != sc.Result.Size:
		return fmt.Errorf("%s has %d bytes, the sidecar says %d", filepath.Base(result), st.Size(), sc.Result.Size)
	case !sc.Result.MTime.IsZero() && !st.ModTime().Equal(sc.Result.MTime):
		return fmt.Errorf("%s was modified after the relayout (%s, the sidecar says %s)",
			filepath.Base(result), st.ModTime().UTC().Format(time.RFC3339), sc.Result.MTime.UTC().Format(time.RFC3339))
	}
	return nil
}

// Restamp records the result's current modification time in its sidecar, after
// a full verification found the bytes right (a copied file has a new one).
func Restamp(result string) error {
	p := SidecarPath(result)
	sc, err := ReadSidecar(p)
	if err != nil {
		return err
	}
	st, err := os.Stat(result)
	if err != nil {
		return err
	}
	sc.Result.MTime = st.ModTime().UTC()
	return WriteSidecar(p, sc)
}
