//go:build linux

package memlock

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

// EnvNoAdvise turns AdviseRandom off ("1"), for A/B measurements.
const EnvNoAdvise = "NANOLLM_NO_ADVISE_RANDOM"

// AdviseRandom tells the kernel that the given ranges of path are read at
// random (MADV_RANDOM) in every mapping of the file this process holds: no
// readahead on a fault. The PLE table is the case: a token reads one row, and
// a fault that drags in 128 KiB around it evicts the hot rows of a page cache
// that a tight budget (M7, M8) leaves very small. It returns the bytes
// advised; a file that is not mapped (yet) advises nothing, which is not an
// error.
func AdviseRandom(path string, ranges []Range) (int64, error) {
	if len(ranges) == 0 || os.Getenv(EnvNoAdvise) == "1" {
		return 0, nil
	}
	want, err := filepath.EvalSymlinks(path)
	if err != nil {
		return 0, err
	}
	f, err := os.Open("/proc/self/maps")
	if err != nil {
		return 0, err
	}
	defer f.Close()
	page := int64(os.Getpagesize())
	var done int64
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		// start-end perms offset dev inode path
		fields := strings.Fields(sc.Text())
		if len(fields) < 6 || fields[5] != want {
			continue
		}
		lo, hi, ok := strings.Cut(fields[0], "-")
		if !ok {
			continue
		}
		start, err1 := strconv.ParseUint(lo, 16, 64)
		end, err2 := strconv.ParseUint(hi, 16, 64)
		off, err3 := strconv.ParseInt(fields[2], 16, 64)
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}
		mapEnd := off + int64(end-start)
		for _, r := range ranges {
			a, b := max(r.Off, off), min(r.Off+r.Len, mapEnd)
			if b <= a {
				continue
			}
			addr := int64(start) + (a - off)
			addr -= addr % page // the page holding the first byte
			n := int64(start) + (b - off) - addr
			n = (n + page - 1) / page * page
			if _, _, e := unix.Syscall(unix.SYS_MADVISE, uintptr(addr), uintptr(n), unix.MADV_RANDOM); e != 0 {
				return done, e
			}
			done += n
		}
	}
	return done, sc.Err()
}
