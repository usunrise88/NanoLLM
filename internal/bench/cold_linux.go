//go:build linux

package bench

import (
	"os"

	"golang.org/x/sys/unix"
)

// evictFile drops a file's clean pages from the page cache (not everyone's:
// drop_caches would also take the neighbours' on a shared host). Pages a
// process still maps stay; the previous worker has exited by now.
func evictFile(path string) {
	if path == "" {
		return
	}
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	_ = unix.Fadvise(int(f.Fd()), 0, 0, unix.FADV_DONTNEED)
}
