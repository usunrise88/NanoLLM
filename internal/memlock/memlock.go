// Package memlock pins byte ranges of a model file in RAM (SPEC §4, modes M7
// and M8: the core locked, the rest file-backed).
//
// libllama maps the GGUF itself; this package maps the same file a second
// time, read-only and shared, and locks the planned ranges in that mapping.
// Both mappings share the page cache, so the pages libllama reads through its
// own mapping stay resident too. That is the claim range-lock rests on, and
// the tests prove it by residency rather than by trusting the lock call:
// pages are pushed out through a second, unlocked mapping, and the locked
// ranges must still be resident there (SPEC §14, risk "range-lock does not
// cover libllama's mapping").
package memlock

import (
	"errors"
	"fmt"
	"os"
	"sort"
)

// Range is a span of the file, in bytes from its start.
type Range struct {
	Off int64 `json:"off"`
	Len int64 `json:"len"`
}

// ErrLimit is a lock refused for the locked-memory limit; the error says how
// much was needed and how to raise it.
var ErrLimit = errors.New("memlock: locked-memory limit too low")

// Pin is a set of locked ranges of one file. It lives as long as the model
// that uses the file; Close releases it.
type Pin struct {
	f      *os.File
	size   int64
	ranges []Range // page-aligned, merged
	locked int64
	sys    pinSys
}

// PageSize is the alignment ranges are widened to.
var PageSize = int64(os.Getpagesize())

// align widens ranges to whole pages, clips them to the file and merges the
// overlapping ones.
func align(ranges []Range, size int64) []Range {
	var out []Range
	for _, r := range ranges {
		if r.Len <= 0 || r.Off >= size {
			continue
		}
		start := r.Off / PageSize * PageSize
		end := min(r.Off+r.Len, size)
		end = (end + PageSize - 1) / PageSize * PageSize
		out = append(out, Range{start, end - start})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Off < out[j].Off })
	var merged []Range
	for _, r := range out {
		if n := len(merged); n > 0 && r.Off <= merged[n-1].Off+merged[n-1].Len {
			last := &merged[n-1]
			last.Len = max(last.Off+last.Len, r.Off+r.Len) - last.Off
			continue
		}
		merged = append(merged, r)
	}
	return merged
}

// Total is the bytes the ranges cover.
func Total(ranges []Range) int64 {
	var n int64
	for _, r := range ranges {
		n += r.Len
	}
	return n
}

// PinFile maps path and locks ranges in it, prefetching them first so the
// lock does not fault pages in one by one. The locked-memory limit is raised
// as far as the process may (soft to hard); if that is not enough, the error
// wraps ErrLimit with the shortfall.
func PinFile(path string, ranges []Range) (*Pin, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	p := &Pin{f: f, size: st.Size(), ranges: align(ranges, st.Size())}
	if err := p.lock(); err != nil {
		p.Close()
		return nil, err
	}
	return p, nil
}

// Locked is the number of bytes this pin holds.
func (p *Pin) Locked() int64 { return p.locked }

// Ranges are the page-aligned ranges locked.
func (p *Pin) Ranges() []Range { return append([]Range(nil), p.ranges...) }

// Close unlocks and unmaps.
func (p *Pin) Close() error {
	err := p.unlock()
	if p.f != nil {
		if cerr := p.f.Close(); err == nil {
			err = cerr
		}
		p.f = nil
	}
	return err
}

func limitError(need, have int64, how string) error {
	return fmt.Errorf("%w: %d MiB to lock, the limit allows %d MiB; %s", ErrLimit, need>>20, have>>20, how)
}

var errUnsupported = errors.New("memlock: not supported on this platform")
