//go:build windows

package procmem

import (
	"fmt"
	"sort"
	"strings"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	pdh                      = windows.NewLazySystemDLL("pdh.dll")
	procPdhOpenQuery         = pdh.NewProc("PdhOpenQueryW")
	procPdhAddEnglishCounter = pdh.NewProc("PdhAddEnglishCounterW")
	procPdhCollectQueryData  = pdh.NewProc("PdhCollectQueryData")
	procPdhGetFormattedArray = pdh.NewProc("PdhGetFormattedCounterArrayW")
	procPdhCloseQuery        = pdh.NewProc("PdhCloseQuery")
)

const (
	pdhFmtLarge     = 0x00000400
	pdhMoreData     = 0x800007D2
	pdhNoData       = 0x800007D5
	pdhCalcNegative = 0x800007D6
)

// pdhItem is PDH_FMT_COUNTERVALUE_ITEM_W for a large integer value.
type pdhItem struct {
	name   *uint16
	status uint32
	_      uint32
	value  int64
}

// ProcessVRAM reads the GPU Process Memory counters of the Windows performance
// data for one process: Dedicated Usage and Shared Usage, summed over the
// adapters the process uses. They are available to an ordinary user and to a
// service alike and include the driver's own context, which the engine's
// buffer sizes do not. It needs Windows 10 1709 or later with a WDDM 2.x
// driver; otherwise it returns ErrUnavailable.
func ProcessVRAM(pid int) (VRAM, error) {
	dedicated, err := gpuProcessInstances(pid, "Dedicated Usage")
	if err != nil {
		return VRAM{}, err
	}
	shared, err := gpuProcessInstances(pid, "Shared Usage")
	if err != nil {
		return VRAM{}, err
	}
	v := VRAM{Source: SourcePDH}
	names := map[string]bool{}
	for n := range dedicated {
		names[n] = true
	}
	for n := range shared {
		names[n] = true
	}
	for n := range names {
		short := strings.TrimPrefix(n, fmt.Sprintf("pid_%d_", pid))
		v.Adapters = append(v.Adapters, Adapter{Name: short, Dedicated: dedicated[n], Shared: shared[n]})
		v.Dedicated += dedicated[n]
		v.Shared += shared[n]
	}
	sort.Slice(v.Adapters, func(i, j int) bool { return v.Adapters[i].Name < v.Adapters[j].Name })
	return v, nil
}

// gpuProcessInstances is the counter per instance (pid_<pid>_luid_…_phys_<n>:
// one per adapter the process touches).
func gpuProcessInstances(pid int, name string) (map[string]int64, error) {
	var q uintptr
	if r, _, _ := procPdhOpenQuery.Call(0, 0, uintptr(unsafe.Pointer(&q))); r != 0 {
		return nil, fmt.Errorf("%w: PdhOpenQuery %#x", ErrUnavailable, r)
	}
	defer procPdhCloseQuery.Call(q)
	path, err := windows.UTF16PtrFromString(fmt.Sprintf(`\GPU Process Memory(pid_%d_*)\%s`, pid, name))
	if err != nil {
		return nil, err
	}
	var c uintptr
	if r, _, _ := procPdhAddEnglishCounter.Call(q, uintptr(unsafe.Pointer(path)), 0, uintptr(unsafe.Pointer(&c))); r != 0 {
		return nil, fmt.Errorf("%w: the GPU Process Memory counters (%#x)", ErrUnavailable, r)
	}
	// The counters are instantaneous: one collection reads them, a second one
	// settles the instance list when the process has only just started using
	// the GPU.
	for range 2 {
		if r, _, _ := procPdhCollectQueryData.Call(q); r != 0 {
			return nil, fmt.Errorf("%w: PdhCollectQueryData %#x", ErrUnavailable, r)
		}
		time.Sleep(20 * time.Millisecond)
	}
	var size, count uint32
	r, _, _ := procPdhGetFormattedArray.Call(c, pdhFmtLarge, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), 0)
	if r != pdhMoreData {
		if r == pdhNoData || r == 0 {
			return nil, nil // the process has no GPU memory
		}
		return nil, fmt.Errorf("%w: PdhGetFormattedCounterArray %#x", ErrUnavailable, r)
	}
	buf := make([]byte, size)
	if r, _, _ := procPdhGetFormattedArray.Call(c, pdhFmtLarge, uintptr(unsafe.Pointer(&size)), uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&buf[0]))); r != 0 {
		return nil, fmt.Errorf("%w: PdhGetFormattedCounterArray %#x", ErrUnavailable, r)
	}
	out := map[string]int64{}
	items := unsafe.Slice((*pdhItem)(unsafe.Pointer(&buf[0])), count)
	for _, it := range items {
		// Only this process's instances (the wildcard `pid_1_*` also matches
		// pid_1x_ on some Windows builds).
		inst := windows.UTF16PtrToString(it.name)
		if !strings.HasPrefix(inst, fmt.Sprintf("pid_%d_", pid)) {
			continue
		}
		if it.status == 0 || it.status == pdhCalcNegative {
			out[inst] += it.value
		}
	}
	return out, nil
}
