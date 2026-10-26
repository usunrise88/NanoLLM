//go:build windows

package hwinfo

import (
	"context"
	"time"
	"unsafe"
)

var procGetSystemTimes = kernel32.NewProc("GetSystemTimes")

func memoryNow() (total, available int64) {
	var ms struct {
		Length                                     uint32
		MemoryLoad                                 uint32
		TotalPhys, AvailPhys, TotalPage, AvailPage uint64
		TotalVirtual, AvailVirtual, AvailExtended  uint64
	}
	ms.Length = uint32(unsafe.Sizeof(ms))
	if r, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&ms))); r != 0 {
		return int64(ms.TotalPhys), int64(ms.AvailPhys)
	}
	return 0, 0
}

type filetime struct{ low, high uint32 }

func (f filetime) ticks() uint64 { return uint64(f.high)<<32 | uint64(f.low) }

func systemTimes() (idle, kernel, user uint64, ok bool) {
	var i, k, u filetime
	r, _, _ := procGetSystemTimes.Call(uintptr(unsafe.Pointer(&i)), uintptr(unsafe.Pointer(&k)), uintptr(unsafe.Pointer(&u)))
	return i.ticks(), k.ticks(), u.ticks(), r != 0
}

// cpuBusy is the share of CPU time spent outside the idle thread over window.
// The kernel time GetSystemTimes returns includes the idle time.
func cpuBusy(ctx context.Context, window time.Duration) float64 {
	i0, k0, u0, ok := systemTimes()
	if !ok {
		return -1
	}
	select {
	case <-time.After(window):
	case <-ctx.Done():
		return -1
	}
	i1, k1, u1, ok := systemTimes()
	if !ok {
		return -1
	}
	total := (k1 - k0) + (u1 - u0)
	if total == 0 {
		return -1
	}
	return 100 * float64(total-(i1-i0)) / float64(total)
}
