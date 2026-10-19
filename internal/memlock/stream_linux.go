//go:build linux

package memlock

import (
	"cmp"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"golang.org/x/sys/unix"
)

// StreamOptions is how a Streamer feeds the layers.
type StreamOptions struct {
	// Ahead is how many layers after the one just computed to ask the kernel
	// to read (MADV_WILLNEED); the layers wrap around, because the next
	// token starts at layer 0 again.
	Ahead int
	// Keep is how many leading layers stay in the page cache for good.
	Keep int
	// Evict drops a layer's pages (MADV_PAGEOUT) once it is computed, unless
	// it is one of the Keep first layers.
	Evict bool
}

// Streamer reads the file-backed layers of a model ahead of the compute and
// drops them behind it (OPT-8). The page cache's LRU does the opposite of
// what a cyclic scan needs: when the layers do not fit it evicts exactly the
// pages the next token reads first, and every token reads everything from
// the disk. The streamer keeps the first Keep layers and cycles the rest
// through the room that is left, so the hit rate is Keep over the total
// instead of zero, and the disk reads in layer order, queued ahead of the
// compute.
//
// One WILLNEED call reads no more than the device's readahead limit
// (ReadaheadLimit) and drops the rest without an error, so the streamer cuts
// a layer into calls of that size; a single call for a 40 MiB layer would
// read its first 8 MiB on a disk with an 8 MiB window, 128 KiB on most.
//
// Layer(i) is called when layer i has been computed. It never blocks: the
// madvise calls run on a goroutine of the streamer's own, and a call that
// finds the queue full is dropped (and counted).
type Streamer struct {
	read [][]span // per layer, every page the layer touches
	drop [][]span // per layer, the pages it fills
	opt  StreamOptions

	ch   chan int
	done chan struct{}
	wg   sync.WaitGroup

	calls, dropped, errs atomic.Int64
	willneed, pageout    atomic.Int64 // bytes advised
	firstErr             atomic.Value // error
}

// NewStreamer maps the layers' file ranges onto the mappings of path in this
// process. The file has to be mapped already (the model is loaded); with no
// mapping, or no range inside one, it returns an error and the caller does
// without a streamer.
func NewStreamer(path string, layers map[int][]Range, opt StreamOptions) (*Streamer, error) {
	ms, err := fileMappings(path)
	if err != nil {
		return nil, err
	}
	if len(ms) == 0 {
		return nil, fmt.Errorf("memlock: %s is not mapped in this process", path)
	}
	n := 0
	for l := range layers {
		n = max(n, l+1)
	}
	s := &Streamer{
		read: make([][]span, n), drop: make([][]span, n), opt: opt,
		ch: make(chan int, 64), done: make(chan struct{}),
	}
	var total int64
	for l, rs := range layers {
		for _, r := range rs {
			// Outward for the read (a partial page costs nothing), inward for
			// the eviction (a neighbour's page is not ours to drop).
			s.read[l] = append(s.read[l], spanOf(ms, r, false)...)
			s.drop[l] = append(s.drop[l], spanOf(ms, r, true)...)
			for _, sp := range s.read[l] {
				total += int64(sp.n)
			}
		}
	}
	limit := uintptr(ReadaheadLimit(path))
	for l := range s.read {
		// A layer is a dozen tensors that lie next to each other; one
		// madvise per tensor is a dozen system calls where one will do. The
		// read bridges the small gaps (norms, a few KiB) and is then cut to
		// what one call reads; the eviction does not bridge, it takes only
		// what the ranges say.
		s.read[l] = splitSpans(mergeSpans(s.read[l], 256<<10), limit)
		s.drop[l] = mergeSpans(s.drop[l], 0)
	}
	if total == 0 {
		return nil, fmt.Errorf("memlock: no layer range lies in a mapping of %s", path)
	}
	s.wg.Add(1)
	go s.run()
	return s, nil
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

// ReadaheadLimit is the most one MADV_WILLNEED (or POSIX_FADV_WILLNEED) call
// reads from the block device that holds path: the kernel clamps the request
// to the larger of the device's readahead window and its largest request
// (force_page_cache_ra), and the rest of the range is not read. It falls back
// to 128 KiB, the usual window, when sysfs does not say.
func ReadaheadLimit(path string) int64 {
	var st unix.Stat_t
	if unix.Stat(path, &st) != nil {
		return 128 << 10
	}
	dir := fmt.Sprintf("/sys/dev/block/%d:%d", unix.Major(st.Dev), unix.Minor(st.Dev))
	q := dir + "/queue"
	if _, err := os.Stat(q); err != nil {
		q = dir + "/../queue" // a partition: the queue is its disk's
	}
	kb := int64(128)
	for _, name := range []string{"read_ahead_kb", "max_sectors_kb"} {
		if b, err := os.ReadFile(q + "/" + name); err == nil {
			if v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil {
				kb = max(kb, v)
			}
		}
	}
	return kb << 10
}

// Layers is the number of layers the streamer knows.
func (s *Streamer) Layers() int { return len(s.read) }

// Warm asks for the first layers, before the first token.
func (s *Streamer) Warm() {
	for l := 0; l < min(s.opt.Ahead+1, len(s.read)); l++ {
		s.advise(s.read[l], unix.MADV_WILLNEED)
	}
}

// Layer reports that layer il has been computed.
func (s *Streamer) Layer(il int) {
	if il < 0 || il >= len(s.read) {
		return
	}
	select {
	case s.ch <- il:
	default:
		s.dropped.Add(1)
	}
}

func (s *Streamer) run() {
	defer s.wg.Done()
	for {
		select {
		case il := <-s.ch:
			s.handle(il)
		case <-s.done:
			return
		}
	}
}

func (s *Streamer) handle(il int) {
	// The eviction first: it is quick and frees the room the reads fill,
	// while a WILLNEED call waits for the device's queue when it is full.
	if s.opt.Evict && il >= s.opt.Keep {
		s.advise(s.drop[il], unix.MADV_PAGEOUT)
	}
	n := len(s.read)
	for d := 1; d <= s.opt.Ahead && d < n; d++ {
		s.advise(s.read[(il+d)%n], unix.MADV_WILLNEED)
	}
}

func (s *Streamer) advise(spans []span, advice int) {
	for _, sp := range spans {
		s.calls.Add(1)
		if _, _, e := unix.Syscall(unix.SYS_MADVISE, sp.addr, sp.n, uintptr(advice)); e != 0 {
			s.errs.Add(1)
			s.firstErr.CompareAndSwap(nil, fmt.Errorf("madvise(%d): %w", advice, e))
			continue
		}
		if advice == unix.MADV_WILLNEED {
			s.willneed.Add(int64(sp.n))
		} else {
			s.pageout.Add(int64(sp.n))
		}
	}
}

// StreamStats are the streamer's counters.
type StreamStats struct {
	Calls, Dropped, Errors int64
	WillNeed, PageOut      int64 // bytes advised
	FirstError             error
}

// Stats reports what the streamer has done.
func (s *Streamer) Stats() StreamStats {
	st := StreamStats{
		Calls: s.calls.Load(), Dropped: s.dropped.Load(), Errors: s.errs.Load(),
		WillNeed: s.willneed.Load(), PageOut: s.pageout.Load(),
	}
	if e, ok := s.firstErr.Load().(error); ok {
		st.FirstError = e
	}
	return st
}

// Close stops the goroutine. Calls to Layer after it are harmless.
func (s *Streamer) Close() {
	select {
	case <-s.done:
	default:
		close(s.done)
	}
	s.wg.Wait()
}
