//go:build linux

package memlock

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// mapping is one mapping of a file in this process: the addresses
// [start, end) show the file from offset off on.
type mapping struct {
	start, end uint64
	off        int64
}

// span is a page-aligned part of the address space.
type span struct {
	addr uintptr
	n    uintptr
}

// fileMappings lists the mappings of path in this process, from
// /proc/self/maps. A file that is not mapped (yet) has none.
func fileMappings(path string) ([]mapping, error) {
	want, err := filepath.EvalSymlinks(path)
	if err != nil {
		return nil, err
	}
	f, err := os.Open("/proc/self/maps")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []mapping
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		// start-end perms offset dev inode path
		fields := strings.Fields(sc.Text())
		if len(fields) < 6 || fields[5] != want {
			continue
		}
		lo, hi, ok := strings.Cut(fields[0], "-")
		if !ok {
			continue
		}
		start, err1 := strconv.ParseUint(lo, 16, 64)
		end, err2 := strconv.ParseUint(hi, 16, 64)
		off, err3 := strconv.ParseInt(fields[2], 16, 64)
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}
		out = append(out, mapping{start, end, off})
	}
	return out, sc.Err()
}

// spans maps a file range onto the mappings. The result covers whole pages:
// inward, only pages the range fills (so that advising it never touches a
// neighbour's page), outward, every page it touches.
func spanOf(ms []mapping, r Range, inward bool) []span {
	page := int64(os.Getpagesize())
	var out []span
	for _, m := range ms {
		mapEnd := m.off + int64(m.end-m.start)
		a, b := max(r.Off, m.off), min(r.Off+r.Len, mapEnd)
		if b <= a {
			continue
		}
		lo := int64(m.start) + (a - m.off)
		hi := int64(m.start) + (b - m.off)
		if inward {
			lo = (lo + page - 1) / page * page
			hi = hi / page * page
		} else {
			lo = lo / page * page
			hi = (hi + page - 1) / page * page
		}
		if hi > lo {
			out = append(out, span{addr: uintptr(lo), n: uintptr(hi - lo)})
		}
	}
	return out
}
