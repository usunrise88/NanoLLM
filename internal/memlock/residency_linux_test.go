//go:build linux

package memlock

import (
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

// tmpfsMagic is statfs's f_type of tmpfs and ramfs-like memory filesystems.
const tmpfsMagic = 0x01021994

// A tmpfs file cannot be evicted (its pages are its storage), which makes the
// residency test vacuous; the test files then go under the user's cache
// directory, on a real filesystem.
func init() {
	testDir = func(t *testing.T) string {
		dir := t.TempDir()
		var st unix.Statfs_t
		if err := unix.Statfs(dir, &st); err != nil || st.Type != tmpfsMagic {
			return dir
		}
		cache, err := os.UserCacheDir()
		if err != nil {
			return dir
		}
		base := filepath.Join(cache, "nanollm-test")
		if err := os.MkdirAll(base, 0o755); err != nil {
			return dir
		}
		d, err := os.MkdirTemp(base, "memlock-")
		if err != nil {
			return dir
		}
		t.Cleanup(func() { _ = os.RemoveAll(d) })
		return d
	}
}

// TestLockCoversOtherMappings is the claim range-lock rests on: libllama's
// own mapping of the file stays resident where our mapping is locked. The
// "other" mapping stands for libllama's; evicting through it drops every
// page that is not locked anywhere.
func TestLockCoversOtherMappings(t *testing.T) {
	const size = 32 << 20
	path := tempFile(t, size)
	locked := []Range{{0, 8 << 20}, {16 << 20, 4 << 20}}
	pin, err := PinFile(path, locked)
	if err != nil {
		t.Skipf("locking is not permitted here: %v", err)
	}
	defer pin.Close()

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	other, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Munmap(other)
	// Touch everything through the other mapping, as inference would.
	var sum byte
	for i := 0; i < size; i += int(PageSize) {
		sum += other[i]
	}
	_ = sum
	if err := evict(other, f.Fd(), 0); err != nil {
		t.Fatal(err)
	}

	for _, r := range locked {
		res, err := Resident(other[r.Off : r.Off+r.Len])
		if err != nil {
			t.Fatal(err)
		}
		if res != r.Len {
			t.Errorf("locked range %+v: only %d of %d bytes resident through the other mapping", r, res, r.Len)
		}
	}
	unlocked := other[24<<20 : 32<<20]
	res, _ := Resident(unlocked)
	if res > int64(len(unlocked))/2 {
		// Most of an unlocked range leaving is what shows the eviction
		// worked and the test is not vacuous.
		t.Errorf("unlocked range stayed resident (%d of %d): the test proves nothing here", res, len(unlocked))
	}
}
