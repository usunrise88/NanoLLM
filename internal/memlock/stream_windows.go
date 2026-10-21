//go:build windows

package memlock

import (
	"errors"
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

// The streamer itself is in streamer.go; this file is what Windows does for it.
//
// Reading ahead is PrefetchVirtualMemory: it queues the reads of a range and returns, so the disk works while the
// compute runs; the pages arrive in the standby list and a later touch is a soft fault. Page faults alone read
// one page cluster at a time (tools/diskbw on the NV3: 1.1 GB/s from one thread, prefetch 2.5 GB/s, overlapped
// reads with a queue depth of 4 4.2 GB/s).
//
// Dropping behind is VirtualUnlock on pages that are not locked: it fails with ERROR_NOT_LOCKED, and its documented
// side effect is that the pages leave the working set. They go to the standby list, so on a machine with memory
// to spare they stay cached; under pressure the memory manager takes the oldest first, which is what a cyclic scan
// needs. Pages another view of the file has locked (the range-locked tensors of M8, a Pin) stay resident.

// readLimit is the size of one PrefetchVirtualMemory call: diskbw found 8, 32 and 64 MiB calls equal.
func readLimit(string) uintptr { return 32 << 20 }

// adviseSpan prefetches the span or trims it out of the working set.
func adviseSpan(sp span, a advice) error {
	if a == pageOut {
		err := windows.VirtualUnlock(sp.addr, sp.n)
		if err == nil || errors.Is(err, windows.ERROR_NOT_LOCKED) {
			return nil
		}
		return fmt.Errorf("VirtualUnlock: %w", err)
	}
	entry := struct {
		addr uintptr
		size uintptr
	}{sp.addr, sp.n}
	r, _, err := procPrefetchVirtualMemory.Call(uintptr(windows.CurrentProcess()), 1, uintptr(unsafe.Pointer(&entry)), 0)
	if r == 0 {
		return fmt.Errorf("PrefetchVirtualMemory: %w", err)
	}
	return nil
}
