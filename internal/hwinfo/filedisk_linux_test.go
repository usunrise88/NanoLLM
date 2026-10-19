package hwinfo

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFileDisk(t *testing.T) {
	if FileDisk("/does/not/exist") != nil || FileDisk("") != nil {
		t.Error("a missing file has fields")
	}
	p := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: %v", p, FileDisk(p))
	// On tmpfs or overlay there is no block device: nothing, but no failure.
	for _, f := range FileDisk(p) {
		switch f.Key {
		case "model_disk", "model_disk_rotational", "model_read_ahead_kb", "model_max_sectors_kb":
		default:
			t.Errorf("unexpected field %q", f.Key)
		}
	}
}
