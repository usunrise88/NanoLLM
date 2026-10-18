//go:build windows

package memlock

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	kernel32                     = windows.NewLazySystemDLL("kernel32.dll")
	procSetProcessWorkingSetSize = kernel32.NewProc("SetProcessWorkingSetSizeEx")
	procGetProcessWorkingSetSize = kernel32.NewProc("GetProcessWorkingSetSizeEx")
	procPrefetchVirtualMemory    = kernel32.NewProc("PrefetchVirtualMemory")
	psapi                        = windows.NewLazySystemDLL("psapi.dll")
	procQueryWorkingSetEx        = psapi.NewProc("QueryWorkingSetEx")
	procEmptyWorkingSet          = psapi.NewProc("EmptyWorkingSet")
)

type pinSys struct {
	mapping windows.Handle
	views   []uintptr
	sizes   []int64
}

// growWorkingSet raises the process's minimum working set by need: VirtualLock
// can lock no more than the minimum working set allows.
func growWorkingSet(need int64) error {
	proc := windows.CurrentProcess()
	var minWS, maxWS uintptr
	var flags uint32
	if r, _, err := procGetProcessWorkingSetSize.Call(uintptr(proc), uintptr(unsafe.Pointer(&minWS)), uintptr(unsafe.Pointer(&maxWS)), uintptr(unsafe.Pointer(&flags))); r == 0 {
		return err
	}
	newMin := minWS + uintptr(need) + 64<<20
	newMax := max(maxWS, newMin+64<<20)
	if r, _, err := procSetProcessWorkingSetSize.Call(uintptr(proc), newMin, newMax, 0); r == 0 {
		return limitError(need, int64(minWS), "SetProcessWorkingSetSizeEx refused ("+err.Error()+"); free memory or lower the preset")
	}
	return nil
}

func (p *Pin) lock() error {
	need := Total(p.ranges)
	if need == 0 {
		return nil
	}
	if err := growWorkingSet(need); err != nil {
		return err
	}
	h, err := windows.CreateFileMapping(windows.Handle(p.f.Fd()), nil, windows.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		return fmt.Errorf("memlock: map the file: %w", err)
	}
	p.sys.mapping = h
	for _, r := range p.ranges {
		// View offsets must be multiples of the allocation granularity
		// (64 KiB), not just of the page size.
		const gran = 64 << 10
		start := r.Off / gran * gran
		size := r.Off + r.Len - start
		addr, err := windows.MapViewOfFile(h, windows.FILE_MAP_READ, uint32(start>>32), uint32(start), uintptr(size))
		if err != nil {
			return fmt.Errorf("memlock: view %d bytes at %d: %w", size, start, err)
		}
		p.sys.views = append(p.sys.views, addr)
		p.sys.sizes = append(p.sys.sizes, size)
		entry := struct {
			addr uintptr
			size uintptr
		}{addr, uintptr(size)}
		procPrefetchVirtualMemory.Call(uintptr(windows.CurrentProcess()), 1, uintptr(unsafe.Pointer(&entry)), 0)
		if err := windows.VirtualLock(addr, uintptr(size)); err != nil {
			return limitError(need, p.locked, "VirtualLock refused ("+err.Error()+")")
		}
		p.locked += r.Len
	}
	return nil
}

func (p *Pin) unlock() error {
	var first error
	for i, v := range p.sys.views {
		_ = windows.VirtualUnlock(v, uintptr(p.sys.sizes[i]))
		if err := windows.UnmapViewOfFile(v); err != nil && first == nil {
			first = err
		}
	}
	p.sys.views, p.sys.sizes = nil, nil
	if p.sys.mapping != 0 {
		windows.CloseHandle(p.sys.mapping)
		p.sys.mapping = 0
	}
	p.locked = 0
	return first
}

// Resident counts the bytes of b in this process's working set, page by page
// through QueryWorkingSetEx.
func Resident(b []byte) (int64, error) {
	if len(b) == 0 {
		return 0, nil
	}
	n := (int64(len(b)) + PageSize - 1) / PageSize
	type info struct {
		addr  uintptr
		attrs uintptr
	}
	buf := make([]info, n)
	base := uintptr(unsafe.Pointer(&b[0]))
	for i := range buf {
		buf[i].addr = base + uintptr(int64(i)*PageSize)
	}
	if r, _, err := procQueryWorkingSetEx.Call(uintptr(windows.CurrentProcess()), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf))*unsafe.Sizeof(buf[0])); r == 0 {
		return 0, err
	}
	var res int64
	for _, e := range buf {
		if e.attrs&1 != 0 { // Valid
			res += PageSize
		}
	}
	return min(res, int64(len(b))), nil
}

// evict trims the whole process working set: a locked page stays, every
// other one leaves it.
func evict([]byte, uintptr, int64) error {
	if r, _, err := procEmptyWorkingSet.Call(uintptr(windows.CurrentProcess())); r == 0 {
		return err
	}
	return nil
}

// viewBytes is the i-th locked view as a slice. The view's address comes from
// MapViewOfFile as an integer, and this is the conversion vet accepts for
// memory the Go heap does not own.
func (p *Pin) viewBytes(i int) []byte {
	addr := p.sys.views[i]
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&addr))
	return unsafe.Slice((*byte)(ptr), p.sys.sizes[i])
}
