package hwinfo

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestReadDiskSpeedOnATempFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "f.bin")
	buf := make([]byte, 1<<20)
	for i := range buf {
		buf[i] = byte(i * 7)
	}
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	for range 40 {
		if _, err := f.Write(buf); err != nil {
			t.Fatal(err)
		}
	}
	f.Close()
	r, err := ReadDiskSpeed(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%+v", r)
	if r.SeqMiBs <= 0 || r.RandIOPS <= 0 || r.ReadBytes < 8<<20 {
		t.Fatalf("%+v", r)
	}
	if _, err := ReadDiskSpeed(context.Background(), filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("a missing file was read")
	}
	small := filepath.Join(t.TempDir(), "s.bin")
	_ = os.WriteFile(small, []byte("x"), 0o600)
	if _, err := ReadDiskSpeed(context.Background(), small); err == nil {
		t.Fatal("a one-byte file was accepted")
	}
}

// TestReadDiskSpeedOfTheModel measures the disk NANOLLM_TEST_MODEL lives on.
func TestReadDiskSpeedOfTheModel(t *testing.T) {
	p := os.Getenv("NANOLLM_TEST_MODEL")
	if p == "" {
		t.Skip("set NANOLLM_TEST_MODEL")
	}
	r, err := ReadDiskSpeed(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	k := DiskOf(p)
	t.Logf("%s: seq %.0f MiB/s, random 64K %.0f MiB/s (%.0f IOPS), read %d MiB, direct %v; drive %+v", p, r.SeqMiBs, r.RandMiBs, r.RandIOPS, r.ReadBytes>>20, r.Direct, k)
}

func TestDiskOfThisDirectory(t *testing.T) {
	k := DiskOf(t.TempDir())
	t.Logf("%+v", k)
	switch k.Type {
	case DiskNVMe, DiskSSD, DiskHDD, DiskUnknown:
	default:
		t.Fatalf("unexpected type %q", k.Type)
	}
}
