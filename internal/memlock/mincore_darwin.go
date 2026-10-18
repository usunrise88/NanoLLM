//go:build darwin

package memlock

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

func mincore(b []byte, vec []byte) error {
	_, _, e := unix.Syscall(unix.SYS_MINCORE, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), uintptr(unsafe.Pointer(&vec[0])))
	if e != 0 {
		return e
	}
	return nil
}

// macOS cannot drop a file's cached pages on request; the residency proof
// does not run there (measure only, docs/STAGE1.md 1.6).
func evict([]byte, uintptr, int64) error { return errUnsupported }
