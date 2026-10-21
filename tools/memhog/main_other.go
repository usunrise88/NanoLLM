//go:build !windows

// Command memhog is Windows-only (it exists to make streaming measurements honest on a machine whose
// file cache would hold the model; on Linux the cgroup of a strict preset does that).
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "memhog: Windows only")
	os.Exit(2)
}
