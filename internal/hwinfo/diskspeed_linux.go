//go:build linux

package hwinfo

import (
	"os"

	"golang.org/x/sys/unix"
)

// alignedBuffer is anonymous memory from mmap: page aligned, which O_DIRECT
// requires.
func alignedBuffer(n int) []byte {
	b, err := unix.Mmap(-1, 0, n, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		return make([]byte, n)
	}
	return b
}

// openDirect opens path with O_DIRECT. A file system that does not support it
// (tmpfs, some FUSE) is read through the cache instead, and the result says so.
func openDirect(path string) (directFile, bool, error) {
	f, err := os.OpenFile(path, os.O_RDONLY|unix.O_DIRECT, 0)
	if err == nil {
		return f, true, nil
	}
	f, err = os.Open(path)
	return f, false, err
}
