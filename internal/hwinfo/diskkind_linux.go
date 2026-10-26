//go:build linux

package hwinfo

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"
)

// DiskOf finds the block device that holds path in sysfs: the NVMe driver
// names its devices nvme*, and queue/rotational says whether the drive spins.
func DiskOf(path string) DiskKind {
	k := DiskKind{Type: DiskUnknown}
	var st unix.Stat_t
	if path == "" || unix.Stat(path, &st) != nil {
		return k
	}
	real, err := filepath.EvalSymlinks(fmt.Sprintf("/sys/dev/block/%d:%d", unix.Major(st.Dev), unix.Minor(st.Dev)))
	if err != nil {
		return k
	}
	q := filepath.Join(real, "queue")
	if _, err := os.Stat(q); err != nil {
		q = filepath.Join(filepath.Dir(real), "queue") // a partition: its disk's queue
	}
	rot, err := os.ReadFile(filepath.Join(q, "rotational"))
	haveSeek := err == nil
	name := filepath.Base(filepath.Dir(q))
	bus := ""
	switch {
	case strings.HasPrefix(name, "nvme"):
		bus = "nvme"
	case strings.HasPrefix(name, "mmcblk"):
		bus = "mmc"
	case strings.HasPrefix(name, "vd"), strings.HasPrefix(name, "xvd"):
		bus = "virtual"
	}
	if rem, err := os.ReadFile(filepath.Join(filepath.Dir(q), "removable")); err == nil {
		k.Removable = strings.TrimSpace(string(rem)) == "1"
	}
	k.Bus = bus
	k.Type = kindOf(bus, strings.TrimSpace(string(rot)) == "1", haveSeek)
	return k
}
