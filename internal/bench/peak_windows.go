//go:build windows

package bench

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

var procGetProcessMemoryInfo = windows.NewLazySystemDLL("psapi.dll").NewProc("GetProcessMemoryInfo")

// peakMemory is the process's peak working set; Windows reports no locked
// total.
func peakMemory() (rss, locked int64) {
	var c struct {
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
	}
	c.cb = uint32(unsafe.Sizeof(c))
	if r, _, _ := procGetProcessMemoryInfo.Call(uintptr(windows.CurrentProcess()), uintptr(unsafe.Pointer(&c)), uintptr(c.cb)); r != 0 {
		rss = int64(c.peakWorkingSetSize)
	}
	return rss, 0
}
