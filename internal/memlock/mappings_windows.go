//go:build windows

package memlock

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"golang.org/x/sys/windows"
)

// memMapped is MEM_MAPPED, the Type of a region that shows a file mapping (golang.org/x/sys/windows has no constant).
const memMapped = 0x40000

var (
	procGetMappedFileNameW = psapi.NewProc("GetMappedFileNameW")
	procQueryDosDeviceW    = kernel32.NewProc("QueryDosDeviceW")
)

// ntPath turns a DOS path (C:\dir\file) into the form GetMappedFileName reports
// (\Device\HarddiskVolume3\dir\file).
func ntPath(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	if real, err := filepath.EvalSymlinks(abs); err == nil {
		abs = real
	}
	if len(abs) < 3 || abs[1] != ':' {
		return "", fmt.Errorf("memlock: %s is not a drive path", abs)
	}
	drive, err := windows.UTF16PtrFromString(abs[:2])
	if err != nil {
		return "", err
	}
	buf := make([]uint16, 1024)
	n, _, e := procQueryDosDeviceW.Call(uintptr(unsafe.Pointer(drive)), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	if n == 0 {
		return "", fmt.Errorf("memlock: QueryDosDevice(%s): %w", abs[:2], e)
	}
	return windows.UTF16ToString(buf) + abs[2:], nil
}

// fileMappings lists the views of path in this process: the regions of memory that show the whole file, found by
// walking the address space and asking for the file name of every mapped region. llama.cpp maps a model with one
// view from offset 0, so a view as large as the file is that mapping; the views of a Pin are shorter and are left out
// (they are not at offset 0, and VirtualQuery does not tell the offset). A file that is not mapped (yet) has none.
func fileMappings(path string) ([]mapping, error) {
	want, err := ntPath(path)
	if err != nil {
		return nil, err
	}
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	proc := uintptr(windows.CurrentProcess())
	type view struct {
		start, end uintptr
		match      bool
	}
	views := map[uintptr]*view{}
	var order []uintptr
	var addr uintptr
	name := make([]uint16, 1024)
	for {
		var mbi windows.MemoryBasicInformation
		if err := windows.VirtualQuery(addr, &mbi, unsafe.Sizeof(mbi)); err != nil {
			break
		}
		end := mbi.BaseAddress + mbi.RegionSize
		if mbi.Type == memMapped && mbi.State == windows.MEM_COMMIT {
			v := views[mbi.AllocationBase]
			if v == nil {
				n, _, _ := procGetMappedFileNameW.Call(proc, mbi.BaseAddress, uintptr(unsafe.Pointer(&name[0])), uintptr(len(name)))
				v = &view{start: mbi.AllocationBase, match: n > 0 && strings.EqualFold(windows.UTF16ToString(name[:n]), want)}
				views[mbi.AllocationBase] = v
				order = append(order, mbi.AllocationBase)
			}
			v.end = end
		}
		if end <= addr {
			break
		}
		addr = end
	}
	var out []mapping
	for _, base := range order {
		v := views[base]
		if v.match && int64(v.end-v.start) >= st.Size() {
			out = append(out, mapping{start: uint64(v.start), end: uint64(v.end), off: 0})
		}
	}
	return out, nil
}
