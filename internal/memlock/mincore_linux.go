//go:build linux

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

// evict drops b's pages from the page cache wherever nothing holds them: the
// mapping's own references go (MADV_DONTNEED), then the file's cached pages
// (FADV_DONTNEED over b's file range). A page locked through any mapping is
// not dropped, which is the property the residency test checks.
// MADV_PAGEOUT would be the direct way, but some kernels accept it and evict
// nothing.
func evict(b []byte, fd uintptr, off int64) error {
	if err := unix.Madvise(b, unix.MADV_DONTNEED); err != nil {
		return err
	}
	return unix.Fadvise(int(fd), off, int64(len(b)), unix.FADV_DONTNEED)
}
