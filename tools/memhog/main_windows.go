//go:build windows

// Command memhog takes physical memory away from the rest of the system, so that a streaming
// measurement (docs/AUTOTUNE.md, OPT-8) reads its weights from the disk instead of the standby
// list: on a 32 GiB machine the file cache would hold the whole 4 GiB model after the first token and
// "streaming from disk" would measure RAM.
//
//	memhog -leave-mib 3000 -hold-sec 900          # until 3000 MiB of memory are available, for 15 minutes
//	memhog -leave-mib 3000 -stop FILE             # until FILE exists
//
// It allocates and fills (with incompressible data, the memory compressor would give it back) chunks
// of 128 MiB until the available physical memory (free + standby, GlobalMemoryStatusEx) falls to the
// requested amount, then prints one line and holds. It releases everything when the time is up, the
// stop file appears or it is killed. -max-mib bounds what it takes whatever the system says.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

type memoryStatusEx struct {
	Length                uint32
	MemoryLoad            uint32
	TotalPhys, AvailPhys  uint64
	TotalPage, AvailPage  uint64
	TotalVirtual, AvailVA uint64
	AvailExtendedVirtual  uint64
}

var procGlobalMemoryStatusEx = windows.NewLazySystemDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

func avail() (availMiB, totalMiB uint64) {
	m := memoryStatusEx{Length: uint32(unsafe.Sizeof(memoryStatusEx{}))}
	r, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&m)))
	if r == 0 {
		return 0, 0
	}
	return m.AvailPhys >> 20, m.TotalPhys >> 20
}

func main() {
	leave := flag.Uint64("leave-mib", 3000, "stop taking memory when this much is available")
	maxMiB := flag.Uint64("max-mib", 28000, "never take more than this")
	hold := flag.Duration("hold-sec", 900*time.Second, "release after this long")
	stop := flag.String("stop", "", "release when this file exists")
	flag.Parse()

	const chunk = 128 << 20
	var held [][]byte
	var taken uint64
	x := uint64(88172645463325252)
	for taken < *maxMiB {
		a, _ := avail()
		if a <= *leave {
			break
		}
		addr, err := windows.VirtualAlloc(0, chunk, windows.MEM_COMMIT|windows.MEM_RESERVE, windows.PAGE_READWRITE)
		if err != nil {
			fmt.Fprintln(os.Stderr, "memhog: VirtualAlloc:", err)
			break
		}
		// The address comes from VirtualAlloc as an integer: this is the conversion vet accepts
		// for memory the Go heap does not own.
		ptr := *(*unsafe.Pointer)(unsafe.Pointer(&addr))
		b := unsafe.Slice((*byte)(ptr), chunk)
		for i := 0; i < len(b); i += 8 {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
			*(*uint64)(unsafe.Pointer(&b[i])) = x
		}
		held = append(held, b)
		taken += chunk >> 20
	}
	a, t := avail()
	fmt.Printf("memhog: holding %d MiB, %d of %d MiB available\n", taken, a, t)
	deadline := time.Now().Add(*hold)
	for time.Now().Before(deadline) {
		if *stop != "" {
			if _, err := os.Stat(*stop); err == nil {
				break
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	fmt.Println("memhog: releasing")
}
