package main

import (
	"fmt"
	"strconv"
	"strings"
	"syscall"
)

// setAffinity restricts the process to the logical processors in the hex
// mask ("0x555" is one thread of each of the six P-cores of an i7-12700H, whose
// SMT siblings are the odd ones; "0xFFF" all twelve of their threads;
// "0xFF000" the eight E-cores).
func setAffinity(hex string) error {
	mask, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(hex), "0x"), 16, 64)
	if err != nil {
		return fmt.Errorf("-mask %q: %w", hex, err)
	}
	k32 := syscall.NewLazyDLL("kernel32.dll")
	h, _, _ := k32.NewProc("GetCurrentProcess").Call()
	r, _, e := k32.NewProc("SetProcessAffinityMask").Call(h, uintptr(mask))
	if r == 0 {
		return fmt.Errorf("SetProcessAffinityMask(%#x): %w", mask, e)
	}
	return nil
}
