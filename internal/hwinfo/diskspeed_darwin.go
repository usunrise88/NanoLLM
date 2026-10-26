//go:build darwin

package hwinfo

import (
	"os"

	"golang.org/x/sys/unix"
)

func alignedBuffer(n int) []byte {
	b, err := unix.Mmap(-1, 0, n, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		return make([]byte, n)
	}
	return b
}

// openDirect opens path and turns the cache off for it with F_NOCACHE.
func openDirect(path string) (directFile, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	_, err = unix.FcntlInt(f.Fd(), unix.F_NOCACHE, 1)
	return f, err == nil, nil
}
