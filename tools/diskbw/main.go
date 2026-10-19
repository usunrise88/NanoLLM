// Command diskbw measures how fast a file can be read from the disk by the
// ways a model's weights reach memory: one sequential O_DIRECT stream,
// several parallel O_DIRECT streams (queue depth), page faults through an
// mmap with the kernel's readahead, and an madvise(WILLNEED) of the whole
// region. The ratio between them decides whether the layer streamer's
// madvise path (OPT-8) already reads at the device's speed or an explicit
// io_uring ring (OPT-9) has room to gain (docs/AUTOTUNE.md).
//
//	go run ./tools/diskbw -file MODEL.gguf [-offset-mib 2000] [-size-mib 1024] [-threads 1,4,16]
//
// Every measurement starts cold: the region is evicted from the page cache
// first (posix_fadvise DONTNEED; pages another process maps stay, so run it
// alone). Linux only.
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

var (
	file   = flag.String("file", "", "file to read")
	offMiB = flag.Int64("offset-mib", 2000, "start of the region, MiB")
	szMiB  = flag.Int64("size-mib", 1024, "size of the region, MiB")
	chunk  = flag.Int64("chunk-kib", 4096, "O_DIRECT read size, KiB")
	thr    = flag.String("threads", "1,4,16", "parallel streams to try")
)

func main() {
	flag.Parse()
	if *file == "" {
		fmt.Fprintln(os.Stderr, "diskbw: -file is required")
		os.Exit(2)
	}
	f, err := os.Open(*file)
	check(err)
	st, err := f.Stat()
	check(err)
	off, size := *offMiB<<20, *szMiB<<20
	if off+size > st.Size() {
		size = st.Size() - off
	}
	fmt.Printf("# %s region %d MiB at %d MiB, chunk %d KiB\n", *file, size>>20, off>>20, *chunk)

	report := func(name string, d time.Duration) {
		fmt.Printf("%-34s %7.0f MB/s  (%.2f s)\n", name, float64(size)/1e6/d.Seconds(), d.Seconds())
	}

	for _, t := range ints(*thr) {
		evict(f, off, size)
		report(fmt.Sprintf("O_DIRECT, %d stream(s)", t), direct(*file, off, size, t))
	}
	for _, t := range ints(*thr) {
		evict(f, off, size)
		report(fmt.Sprintf("mmap page faults, %d thread(s)", t), faults(f, off, size, t))
	}
	evict(f, off, size)
	report("madvise(WILLNEED) then wait", willneed(f, off, size))
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "diskbw:", err)
		os.Exit(1)
	}
}

func ints(s string) []int {
	var out []int
	for _, p := range strings.Split(s, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(p)); err == nil && n > 0 {
			out = append(out, n)
		}
	}
	return out
}

func evict(f *os.File, off, size int64) {
	check(unix.Fadvise(int(f.Fd()), off, size, unix.FADV_DONTNEED))
}

// direct reads the region with t parallel O_DIRECT streams, each over its own
// contiguous part.
func direct(path string, off, size int64, t int) time.Duration {
	var wg sync.WaitGroup
	start := time.Now()
	for k := 0; k < t; k++ {
		lo := off + size*int64(k)/int64(t)
		hi := off + size*int64(k+1)/int64(t)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fd, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECT, 0)
			check(err)
			defer unix.Close(fd)
			// O_DIRECT wants an aligned buffer, offset and length.
			raw := make([]byte, *chunk<<10+4096)
			buf := raw[(4096-int(uintptr(unsafe.Pointer(&raw[0])))%4096)%4096:][:*chunk<<10]
			lo := lo &^ 4095
			for p := lo; p < hi; p += int64(len(buf)) {
				n := int64(len(buf))
				if p+n > hi {
					n = (hi - p + 4095) &^ 4095
				}
				if _, err := unix.Pread(fd, buf[:n], p); err != nil {
					check(err)
				}
			}
		}()
	}
	wg.Wait()
	return time.Since(start)
}

// faults maps the file and touches one byte per page of the region from t
// threads, each over its own contiguous part: what a tensor read through the
// loader's mapping does.
func faults(f *os.File, off, size int64, t int) time.Duration {
	m, err := unix.Mmap(int(f.Fd()), 0, int(off+size), unix.PROT_READ, unix.MAP_PRIVATE)
	check(err)
	defer unix.Munmap(m)
	page := int64(os.Getpagesize())
	var wg sync.WaitGroup
	sums := make([]byte, t)
	start := time.Now()
	for k := 0; k < t; k++ {
		lo := off + size*int64(k)/int64(t)
		hi := off + size*int64(k+1)/int64(t)
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

// willneed asks the kernel to read the whole region and waits until it is
// resident.
func willneed(f *os.File, off, size int64) time.Duration {
	m, err := unix.Mmap(int(f.Fd()), 0, int(off+size), unix.PROT_READ, unix.MAP_PRIVATE)
	check(err)
	defer unix.Munmap(m)
	region := m[off : off+size]
	start := time.Now()
	check(unix.Madvise(region, unix.MADV_WILLNEED))
	vec := make([]byte, size/int64(os.Getpagesize()))
	for {
		_, _, e := unix.Syscall(unix.SYS_MINCORE, uintptr(unsafe.Pointer(&region[0])), uintptr(len(region)), uintptr(unsafe.Pointer(&vec[0])))
		if e != 0 {
			check(e)
		}
		missing := 0
		for _, v := range vec {
			if v&1 == 0 {
				missing++
			}
		}
		if missing == 0 {
			return time.Since(start)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
