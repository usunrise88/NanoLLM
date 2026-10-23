//go:build !windows && !linux && !darwin

package hwinfo

import "os"

func alignedBuffer(n int) []byte { return make([]byte, n) }

func openDirect(path string) (directFile, bool, error) {
	f, err := os.Open(path)
	return f, false, err
}
