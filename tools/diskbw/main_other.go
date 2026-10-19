//go:build !linux

// Command diskbw is Linux-only (O_DIRECT, madvise, mincore).
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "diskbw: Linux only")
	os.Exit(2)
}
