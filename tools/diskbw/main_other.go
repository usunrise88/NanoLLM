//go:build !linux && !windows

// Command diskbw is built for Linux (O_DIRECT, madvise, mincore) and Windows.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "diskbw: Linux and Windows only")
	os.Exit(2)
}
