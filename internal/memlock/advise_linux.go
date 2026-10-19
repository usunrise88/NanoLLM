//go:build linux

package memlock

import (
	"os"

	"golang.org/x/sys/unix"
)

// EnvAdvise turns AdviseRandom on ("1"). It is off by default: the A/B run on
// hyper (docs/AUTOTUNE.md) found no gain, but the variant is kept for disks and
// kernels where readahead hurts, and for the autotuner to try.
const EnvAdvise = "NANOLLM_ADVISE_RANDOM"

// AdviseRandom tells the kernel that the given ranges of path are read at
// random (MADV_RANDOM) in every mapping of the file this process holds: no
// readahead on a fault. The PLE table is the case: a token reads one row, and
// a fault that drags in 128 KiB around it evicts the hot rows of a page cache
// that a tight budget (M7, M8) leaves very small. Only with EnvAdvise=1. It returns the bytes
// advised; a file that is not mapped (yet) advises nothing, which is not an
// error.
func AdviseRandom(path string, ranges []Range) (int64, error) {
	if len(ranges) == 0 || os.Getenv(EnvAdvise) != "1" {
		return 0, nil
	}
	ms, err := fileMappings(path)
	if err != nil {
		return 0, err
	}
	var done int64
	for _, r := range ranges {
		for _, sp := range spanOf(ms, r, false) {
			if _, _, e := unix.Syscall(unix.SYS_MADVISE, sp.addr, sp.n, unix.MADV_RANDOM); e != 0 {
				return done, e
			}
			done += int64(sp.n)
		}
	}
	return done, nil
}
