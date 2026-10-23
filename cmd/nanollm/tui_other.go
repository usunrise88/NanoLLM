//go:build !windows

package main

import "io"

// enableVT: a Unix terminal reads escape sequences already.
func enableVT(io.Writer) {}
