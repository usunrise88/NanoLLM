package main

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

//go:noescape
func readAVX2(p unsafe.Pointer, blocks int) uint64

func hasAVX2() bool { return cpu.X86.HasAVX2 }

// readWide sums b with 256-bit loads; the tail shorter than 256 bytes is left.
func readWide(b []uint64) uint64 {
	blocks := len(b) / 32
	if blocks == 0 {
		return 0
	}
	return readAVX2(unsafe.Pointer(unsafe.SliceData(b)), blocks)
}
