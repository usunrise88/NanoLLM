//go:build linux || windows

package memlock

import (
	"cmp"
	"os"
	"slices"
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

// mergeSpans joins spans whose gap is at most maxGap bytes.
func mergeSpans(sp []span, maxGap uintptr) []span {
	if len(sp) < 2 {
		return sp
	}
	slices.SortFunc(sp, func(a, b span) int { return cmp.Compare(a.addr, b.addr) })
	out := sp[:1]
	for _, c := range sp[1:] {
		last := &out[len(out)-1]
		if end := last.addr + last.n; c.addr <= end+maxGap {
			if ce := c.addr + c.n; ce > end {
				last.n = ce - last.addr
			}
			continue
		}
		out = append(out, c)
	}
	return out
}

// splitSpans cuts spans into pieces of at most limit bytes, rounded down to
// whole pages so every piece starts on one.
func splitSpans(sp []span, limit uintptr) []span {
	page := uintptr(os.Getpagesize())
	limit = max(limit&^(page-1), page)
	var out []span
	for _, c := range sp {
		for c.n > limit {
			out = append(out, span{addr: c.addr, n: limit})
			c.addr += limit
			c.n -= limit
		}
		out = append(out, c)
	}
	return out
}
