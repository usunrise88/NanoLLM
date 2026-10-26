//go:build windows

package hwinfo

import (
	"errors"
	"unsafe"

	"golang.org/x/sys/windows"
)

// alignedBuffer is memory from VirtualAlloc: page aligned, which unbuffered
// reads require.
func alignedBuffer(n int) []byte {
	p, err := windows.VirtualAlloc(0, uintptr(n), windows.MEM_COMMIT|windows.MEM_RESERVE, windows.PAGE_READWRITE)
	if err != nil {
		return make([]byte, n)
	}
	// Never freed: a probe allocates a handful of buffers once. The address
	// comes from VirtualAlloc as an integer, and this is the conversion vet
	// accepts for memory the Go heap does not own.
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&p))
	return unsafe.Slice((*byte)(ptr), n)
}

type winDirect struct{ h windows.Handle }

// openDirect opens path with FILE_FLAG_NO_BUFFERING: reads bypass the cache
// manager, so a file that is cached still reads at the speed of the disk.
// This needs no privilege.
func openDirect(path string) (directFile, bool, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, false, err
	}
	const noBuffering = 0x20000000
	h, err := windows.CreateFile(p, windows.GENERIC_READ, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil,
		windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL|noBuffering, 0)
	if err != nil {
		return nil, false, err
	}
	return &winDirect{h}, true, nil
}

func (f *winDirect) ReadAt(p []byte, off int64) (int, error) {
	var ov windows.Overlapped
	ov.Offset, ov.OffsetHigh = uint32(off), uint32(off>>32)
	var n uint32
	err := windows.ReadFile(f.h, p, &n, &ov)
	if errors.Is(err, windows.ERROR_HANDLE_EOF) {
		err = nil
	}
	return int(n), err
}

func (f *winDirect) Close() error { return windows.CloseHandle(f.h) }
