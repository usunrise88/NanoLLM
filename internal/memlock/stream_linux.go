//go:build linux

package memlock

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

// The streamer itself is in streamer.go; this file is what Linux does for it.

// ReadaheadLimit is the most one MADV_WILLNEED (or POSIX_FADV_WILLNEED) call
// reads from the block device that holds path: the kernel clamps the request
// to the larger of the device's readahead window and its largest request
// (force_page_cache_ra), and the rest of the range is not read. It falls back
// to 128 KiB, the usual window, when sysfs does not say.
func ReadaheadLimit(path string) int64 {
	var st unix.Stat_t
	if unix.Stat(path, &st) != nil {
		return 128 << 10
	}
	dir := fmt.Sprintf("/sys/dev/block/%d:%d", unix.Major(st.Dev), unix.Minor(st.Dev))
	q := dir + "/queue"
	if _, err := os.Stat(q); err != nil {
		q = dir + "/../queue" // a partition: the queue is its disk's
	}
	kb := int64(128)
	for _, name := range []string{"read_ahead_kb", "max_sectors_kb"} {
		if b, err := os.ReadFile(q + "/" + name); err == nil {
			if v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil {
				kb = max(kb, v)
			}
		}
	}
	return kb << 10
}

// readLimit is the size of one read call the streamer makes.
func readLimit(path string) uintptr { return uintptr(ReadaheadLimit(path)) }

// adviseSpan asks the kernel to read the span (MADV_WILLNEED) or to drop its pages (MADV_PAGEOUT).
func adviseSpan(sp span, a advice) error {
	adv := unix.MADV_WILLNEED
	if a == pageOut {
		adv = unix.MADV_PAGEOUT
	}
	if _, _, e := unix.Syscall(unix.SYS_MADVISE, sp.addr, sp.n, uintptr(adv)); e != 0 {
		return fmt.Errorf("madvise(%d): %w", adv, e)
	}
	return nil
}
