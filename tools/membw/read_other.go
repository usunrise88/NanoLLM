//go:build !amd64

package main

func hasAVX2() bool { return false }

func readWide(b []uint64) uint64 { return read(b) }
