//go:build windows

// Command diskbw (Windows) measures how fast a file can be read from the disk
// by the ways a model's weights reach memory: parallel synchronous unbuffered
// streams, one stream with several overlapped reads in flight (queue depth,
// what an IoRing or an OVERLAPPED reader of layers would do), page faults
// through a file mapping (what llama.cpp's mmap load does), and
// PrefetchVirtualMemory in calls of a given size followed by a touch pass.
// The ratio between them decides whether the range-lock path already reads at
// the device's speed or an explicit overlapped reader (OPT-9) has room to
// gain (docs/AUTOTUNE.md).
//
//	go run ./tools/diskbw -dir D:\diskbw                       # a 7 GiB test file
//	go run ./tools/diskbw -file MODEL.gguf -offset-mib 2000    # a model; see below
//
// Windows has no call that drops a file's pages from the cache without
// administrator rights, so a run cannot evict. It makes the cold regions
// instead: the test file is written with FILE_FLAG_NO_BUFFERING (its pages
// never enter the cache), unbuffered reads leave the cache alone, and every
// mapping or prefetch measurement takes a region of its own that nothing has
// read yet. With -file the regions are cold only if the model has not been
// loaded since the cache was last emptied, so the result says so.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	file     = flag.String("file", "", "file to read; empty writes a test file into -dir")
	dir      = flag.String("dir", filepath.Join(".cache", "diskbw"), "where the test file goes (the disk to measure)")
	offMiB   = flag.Int64("offset-mib", 0, "with -file: start of the first region, MiB")
	szMiB    = flag.Int64("size-mib", 1024, "size of each measured region, MiB")
	thr      = flag.String("threads", "1,4,16", "parallel streams (synchronous reads, mapping touchers)")
	qds      = flag.String("qd", "4,16,32", "overlapped reads in flight")
	blocks   = flag.String("block-kib", "1024,4096", "read size of the unbuffered runs, KiB")
	prefetch = flag.String("prefetch-mib", "8,32,64", "PrefetchVirtualMemory call sizes, MiB (0 is one call for the region)")
	keep     = flag.Bool("keep", false, "keep the test file")
)

const sector = 4096

var procPrefetch = windows.NewLazySystemDLL("kernel32.dll").NewProc("PrefetchVirtualMemory")

func main() {
	flag.Parse()
	size := *szMiB << 20
	threads, queues, blockKiB, pre := ints(*thr), ints(*qds), ints(*blocks), intsFrom(*prefetch, 0)
	// One region for the unbuffered runs (they leave no pages behind), one
	// fresh region for each mapping and prefetch run.
	regions := 1 + len(threads) + len(pre)

	path, off := *file, *offMiB<<20
	if path == "" {
		check(os.MkdirAll(*dir, 0o755))
		path = filepath.Join(*dir, "diskbw.bin")
		fmt.Printf("# writing %d MiB unbuffered to %s\n", int64(regions)*size>>20, path)
		check(writeTestFile(path, int64(regions)*size))
		if !*keep {
			defer func() { _ = os.Remove(path) }()
		}
		off = 0
	}
	st, err := os.Stat(path)
	check(err)
	if off+int64(regions)*size > st.Size() {
		fmt.Fprintf(os.Stderr, "diskbw: %s has %d MiB past the offset, the runs need %d MiB of regions (lower -size-mib)\n", path, (st.Size()-off)>>20, int64(regions)*size>>20)
		os.Exit(1)
	}
	fmt.Printf("# %s, regions of %d MiB\n", path, size>>20)

	report := func(name string, d time.Duration, extra string) {
		fmt.Printf("%-44s %7.0f MB/s  (%.2f s%s)\n", name, float64(size)/1e6/d.Seconds(), d.Seconds(), extra)
	}
	hot := off // the unbuffered region
	fresh := off + size
	nextFresh := func() int64 { r := fresh; fresh += size; return r }

	for _, b := range blockKiB {
		for _, t := range threads {
			report(fmt.Sprintf("unbuffered, %d stream(s), %d KiB reads", t, b), direct(path, hot, size, t, int64(b)<<10), "")
		}
	}
	for _, b := range blockKiB {
		for _, q := range queues {
			report(fmt.Sprintf("overlapped, queue depth %d, %d KiB reads", q, b), overlapped(path, hot, size, q, int64(b)<<10), "")
		}
	}
	for _, t := range threads {
		report(fmt.Sprintf("mapping page faults, %d thread(s)", t), faults(path, nextFresh(), size, t), "")
	}
	for _, c := range pre {
		calls, touch := prefetched(path, nextFresh(), size, int64(c)<<20)
		name := "PrefetchVirtualMemory, one call"
		if c > 0 {
			name = fmt.Sprintf("PrefetchVirtualMemory, %d MiB calls", c)
		}
		report(name+" + touch", calls+touch, fmt.Sprintf(": calls %.2f s, touch %.2f s", calls.Seconds(), touch.Seconds()))
	}
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "diskbw:", err)
		os.Exit(1)
	}
}

func ints(s string) []int { return intsFrom(s, 1) }

// intsFrom parses a comma list, keeping the numbers that are at least lo.
func intsFrom(s string, lo int) []int {
	var out []int
	for _, p := range strings.Split(s, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(p)); err == nil && n >= lo {
			out = append(out, n)
		}
	}
	return out
}

// aligned returns a sector-aligned buffer of n bytes (unbuffered I/O wants an
// aligned address, offset and length).
func aligned(n int64) []byte {
	raw := make([]byte, n+sector)
	skip := (sector - int(uintptr(unsafe.Pointer(&raw[0])))%sector) % sector
	return raw[skip : skip+int(n)]
}

func open(path string, flags uint32, write bool) windows.Handle {
	name, err := windows.UTF16PtrFromString(path)
	check(err)
	access, disp, share := uint32(windows.GENERIC_READ), uint32(windows.OPEN_EXISTING), uint32(windows.FILE_SHARE_READ)
	if write {
		access, disp, share = windows.GENERIC_WRITE, windows.CREATE_ALWAYS, 0
	}
	h, err := windows.CreateFile(name, access, share, nil, disp, flags, 0)
	check(err)
	return h
}

// writeTestFile fills path with size bytes of pseudo-random data, written
// without the cache (and through, so the data is on the device).
func writeTestFile(path string, size int64) error {
	h := open(path, windows.FILE_FLAG_NO_BUFFERING|windows.FILE_FLAG_WRITE_THROUGH, true)
	defer func() { _ = windows.CloseHandle(h) }()
	buf := aligned(4 << 20)
	x := uint64(0x9E3779B97F4A7C15)
	for i := 0; i < len(buf); i += 8 {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		*(*uint64)(unsafe.Pointer(&buf[i])) = x
	}
	for done := int64(0); done < size; {
		n := min(int64(len(buf)), size-done)
		n = (n + sector - 1) &^ (sector - 1)
		var w uint32
		if err := windows.WriteFile(h, buf[:n], &w, nil); err != nil {
			return err
		}
		done += int64(w)
	}
	return nil
}

// direct reads the region with t unbuffered synchronous streams, each over its
// own contiguous part and its own handle.
func direct(path string, off, size int64, t int, chunk int64) time.Duration {
	var wg sync.WaitGroup
	start := time.Now()
	for k := 0; k < t; k++ {
		lo := off + size*int64(k)/int64(t)&^(sector-1)
		hi := off + size*int64(k+1)/int64(t)&^(sector-1)
		wg.Add(1)
		go func() {
			defer wg.Done()
			h := open(path, windows.FILE_FLAG_NO_BUFFERING, false)
			defer func() { _ = windows.CloseHandle(h) }()
			buf := aligned(chunk)
			for p := lo; p < hi; p += chunk {
				n := min(chunk, hi-p)
				ov := windows.Overlapped{Offset: uint32(p), OffsetHigh: uint32(p >> 32)}
				var got uint32
				check(windows.ReadFile(h, buf[:n], &got, &ov))
			}
		}()
	}
	wg.Wait()
	return time.Since(start)
}

// overlapped keeps qd reads of chunk bytes in flight on one handle, reissuing
// each slot as it completes, over the region in order.
func overlapped(path string, off, size int64, qd int, chunk int64) time.Duration {
	h := open(path, windows.FILE_FLAG_NO_BUFFERING|windows.FILE_FLAG_OVERLAPPED, false)
	defer func() { _ = windows.CloseHandle(h) }()
	type slot struct {
		ov  windows.Overlapped
		buf []byte
		ev  windows.Handle
	}
	slots := make([]*slot, qd)
	for i := range slots {
		ev, err := windows.CreateEvent(nil, 1, 0, nil)
		check(err)
		slots[i] = &slot{buf: aligned(chunk), ev: ev}
	}
	next, end := off, off+size
	issue := func(s *slot) bool {
		if next >= end {
			return false
		}
		n := min(chunk, end-next)
		s.ov = windows.Overlapped{Offset: uint32(next), OffsetHigh: uint32(next >> 32), HEvent: s.ev}
		check(windows.ResetEvent(s.ev))
		if err := windows.ReadFile(h, s.buf[:n], nil, &s.ov); err != nil && err != windows.ERROR_IO_PENDING {
			check(err)
		}
		next += n
		return true
	}
	start := time.Now()
	var fifo []*slot
	for _, s := range slots {
		if issue(s) {
			fifo = append(fifo, s)
		}
	}
	for len(fifo) > 0 {
		s := fifo[0]
		fifo = fifo[1:]
		_, err := windows.WaitForSingleObject(s.ev, windows.INFINITE)
		check(err)
		var got uint32
		check(windows.GetOverlappedResult(h, &s.ov, &got, false))
		if issue(s) {
			fifo = append(fifo, s)
		}
	}
	d := time.Since(start)
	for _, s := range slots {
		_ = windows.CloseHandle(s.ev)
	}
	return d
}

// view maps the whole file read-only; the returned slice and the release
// function are valid together.
func view(path string) ([]byte, func()) {
	h := open(path, 0, false)
	var st windows.ByHandleFileInformation
	check(windows.GetFileInformationByHandle(h, &st))
	size := int64(st.FileSizeHigh)<<32 | int64(st.FileSizeLow)
	m, err := windows.CreateFileMapping(h, nil, windows.PAGE_READONLY, 0, 0, nil)
	check(err)
	addr, err := windows.MapViewOfFile(m, windows.FILE_MAP_READ, 0, 0, 0)
	check(err)
	// The view's address comes from MapViewOfFile as an integer: this is the
	// conversion vet accepts for memory the Go heap does not own.
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&addr))
	b := unsafe.Slice((*byte)(ptr), size)
	return b, func() {
		_ = windows.UnmapViewOfFile(addr)
		_ = windows.CloseHandle(m)
		_ = windows.CloseHandle(h)
	}
}

// faults touches one byte per page of the region from t threads, each over its
// own contiguous part: what a tensor read through the loader's mapping does.
func faults(path string, off, size int64, t int) time.Duration {
	m, release := view(path)
	defer release()
	const page = 4096
	var wg sync.WaitGroup
	sums := make([]byte, t)
	start := time.Now()
	for k := 0; k < t; k++ {
		lo := off + size*int64(k)/int64(t)&^(page-1)
		hi := off + size*int64(k+1)/int64(t)&^(page-1)
		wg.Add(1)
		go func() {
			defer wg.Done()
			var s byte
			for p := lo; p < hi; p += page {
				s += m[p]
			}
			sums[k] = s
		}()
	}
	wg.Wait()
	d := time.Since(start)
	_ = sums
	return d
}

// prefetched asks Windows to read the region in calls of chunk bytes (0: one
// call), then touches every page from one thread. It returns the time of the
// calls and of the touch pass; if the calls only queue the reads, the touch
// pass waits for the rest, so the sum is the time until the region is usable.
func prefetched(path string, off, size, chunk int64) (calls, touch time.Duration) {
	m, release := view(path)
	defer release()
	if chunk <= 0 {
		chunk = size
	}
	base := uintptr(unsafe.Pointer(&m[0]))
	start := time.Now()
	for o := int64(0); o < size; o += chunk {
		entry := struct {
			addr uintptr
			size uintptr
		}{base + uintptr(off+o), uintptr(min(chunk, size-o))}
		r, _, err := procPrefetch.Call(uintptr(windows.CurrentProcess()), 1, uintptr(unsafe.Pointer(&entry)), 0)
		if r == 0 {
			check(fmt.Errorf("PrefetchVirtualMemory: %w", err))
		}
	}
	calls = time.Since(start)
	var s byte
	t0 := time.Now()
	for p := off; p < off+size; p += 4096 {
		s += m[p]
	}
	touch = time.Since(t0)
	if s == 42 {
		fmt.Fprintln(os.Stderr)
	}
	return calls, touch
}
