//go:build windows

package procmem

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

var procGetProcessMemoryInfo = windows.NewLazySystemDLL("psapi.dll").NewProc("GetProcessMemoryInfo")

// counters is PROCESS_MEMORY_COUNTERS_EX2 (Windows 10 1809 and later), the
// structure that carries the private working set.
type counters struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
	privateUsage               uintptr
	privateWorkingSetSize      uintptr
	sharedCommitUsage          uint64
}

func read() (c counters, ok bool) {
	c.cb = uint32(unsafe.Sizeof(c))
	r, _, _ := procGetProcessMemoryInfo.Call(uintptr(windows.CurrentProcess()), uintptr(unsafe.Pointer(&c)), uintptr(c.cb))
	return c, r != 0
}

// Private is the private working set: the pages of the process that are
// resident, not backed by a file and not shared. That is the RAM the process
// really occupies, and what Task Manager shows as its memory. The commit
// charge (Commit) is not used for the limit: on a discrete GPU under WDDM it
// includes the video memory the process allocates (measured on an RTX 3070 Ti
// Laptop: 4.5 GiB of commit for a model that keeps 0.2 GiB in host RAM).
func Private() int64 {
	if c, ok := read(); ok {
		return int64(c.privateWorkingSetSize)
	}
	return 0
}

// Commit is the commit charge right now (PrivateUsage), and PeakCommit the
// highest it has been (PeakPagefileUsage). Both are reported for comparison.
func Commit() int64 {
	if c, ok := read(); ok {
		return int64(c.privateUsage)
	}
	return 0
}

func PeakCommit() int64 {
	if c, ok := read(); ok {
		return int64(c.peakPagefileUsage)
	}
	return 0
}

// Resident is the peak working set, file pages included, for comparison with
// what the benchmark has always recorded.
func Resident() int64 {
	if c, ok := read(); ok {
		return int64(c.peakWorkingSetSize)
	}
	return 0
}
