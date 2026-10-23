//go:build linux || windows

package memlock

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"sync/atomic"
)

// The methods a Streamer reads ahead by (StreamOptions.Method).
const (
	// MethodAdvise asks the system to read: madvise WILLNEED on Linux,
	// PrefetchVirtualMemory on Windows. The default.
	MethodAdvise = "advise"
	// MethodPread reads the file ranges itself, Depth reads at a time, into
	// a scratch buffer that is thrown away; the reads fill the page cache
	// the mapping is served from. A queue deeper than one is what an NVMe
	// disk needs to reach its rate (OPT-9: on the NV3 of the i7 laptop depth
	// 4 of 4 MiB read 4.2 GB/s against 1.1 for page faults and 2.5 for
	// PrefetchVirtualMemory). A hard disk gains nothing from it.
	MethodPread = "pread"
)

// preadChunk is the size of one read of MethodPread.
const preadChunk = 4 << 20

// defaultDepth is the reads in flight of MethodPread when the option is zero.
const defaultDepth = 4

// StreamOptions is how a Streamer feeds the layers.
type StreamOptions struct {
	// Ahead is how many layers after the one just computed to ask the system
	// to read (madvise WILLNEED, PrefetchVirtualMemory); the layers wrap
	// around, because the next token starts at layer 0 again.
	Ahead int
	// Keep is how many leading layers stay in memory for good.
	Keep int
	// Evict drops a layer's pages (MADV_PAGEOUT, a working-set trim) once it
	// is computed, unless it is one of the Keep first layers.
	Evict bool
	// Method is how the layers are read ahead: MethodAdvise (also "") or
	// MethodPread.
	Method string
	// Depth is the reads in flight of MethodPread; zero is four.
	Depth int
}

// advice says what a Streamer asks of the system for a span.
type advice int

const (
	willNeed advice = iota // read the pages now
	pageOut                // drop the pages
)

func (a advice) String() string {
	if a == willNeed {
		return "willneed"
	}
	return "pageout"
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
// One read call is cut to the size the system serves whole (readLimit: on
// Linux the device's readahead limit, beyond which madvise drops the rest
// without an error), so the streamer cuts a layer into calls of that size.
//
// Layer(i) is called when layer i has been computed. It never blocks: the
// system calls run on a goroutine of the streamer's own, and a call that
// finds the queue full is dropped (and counted).
type Streamer struct {
	read [][]span // per layer, every page the layer touches
	drop [][]span // per layer, the pages it fills
	opt  StreamOptions

	ch   chan int
	done chan struct{}
	wg   sync.WaitGroup

	// MethodPread: the file, the mappings that turn a span back into a file
	// offset, and a scratch buffer for each of the reads in flight.
	file *os.File
	ms   []mapping
	bufs [][]byte

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
	limit := readLimit(path)
	if opt.Method == MethodPread {
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		s.file, s.ms = f, ms
		depth := opt.Depth
		if depth <= 0 {
			depth = defaultDepth
		}
		for range depth {
			s.bufs = append(s.bufs, make([]byte, preadChunk))
		}
		limit = preadChunk
	} else if opt.Method != "" && opt.Method != MethodAdvise {
		return nil, fmt.Errorf("memlock: unknown streaming method %q", opt.Method)
	}
	for l := range s.read {
		// A layer is a dozen tensors that lie next to each other; one
		// call per tensor is a dozen system calls where one will do. The
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

// Layers is the number of layers the streamer knows.
func (s *Streamer) Layers() int { return len(s.read) }

// Warm asks for the first layers, before the first token.
func (s *Streamer) Warm() {
	for l := 0; l < min(s.opt.Ahead+1, len(s.read)); l++ {
		s.advise(s.read[l], willNeed)
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
	// while a read call waits for the device's queue when it is full.
	if s.opt.Evict && il >= s.opt.Keep {
		s.advise(s.drop[il], pageOut)
	}
	n := len(s.read)
	for d := 1; d <= s.opt.Ahead && d < n; d++ {
		s.advise(s.read[(il+d)%n], willNeed)
	}
}

func (s *Streamer) advise(spans []span, a advice) {
	if a == willNeed && s.file != nil {
		s.preadSpans(spans)
		return
	}
	for _, sp := range spans {
		s.calls.Add(1)
		if err := adviseSpan(sp, a); err != nil {
			s.errs.Add(1)
			s.firstErr.CompareAndSwap(nil, fmt.Errorf("%v: %w", a, err))
			continue
		}
		if a == willNeed {
			s.willneed.Add(int64(sp.n))
		} else {
			s.pageout.Add(int64(sp.n))
		}
	}
}

// preadSpans reads the spans' file ranges, len(bufs) of them at a time, and
// returns when all are read. The data is dropped: the reads are for the page
// cache.
func (s *Streamer) preadSpans(spans []span) {
	jobs := make(chan span)
	var wg sync.WaitGroup
	for _, buf := range s.bufs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for sp := range jobs {
				s.calls.Add(1)
				if err := s.preadSpan(sp, buf); err != nil {
					s.errs.Add(1)
					s.firstErr.CompareAndSwap(nil, fmt.Errorf("pread: %w", err))
					continue
				}
				s.willneed.Add(int64(sp.n))
			}
		}()
	}
	for _, sp := range spans {
		jobs <- sp
	}
	close(jobs)
	wg.Wait()
}

// preadSpan reads one span, in pieces that fit the buffer.
func (s *Streamer) preadSpan(sp span, buf []byte) error {
	for _, m := range s.ms {
		if uint64(sp.addr) < m.start || uint64(sp.addr)+uint64(sp.n) > m.end {
			continue
		}
		off := m.off + int64(uint64(sp.addr)-m.start)
		for done := int64(0); done < int64(sp.n); {
			n := min(int64(len(buf)), int64(sp.n)-done)
			got, err := s.file.ReadAt(buf[:n], off+done)
			done += int64(got)
			if errors.Is(err, io.EOF) {
				return nil // a span rounded out to a page may run past the end of the file
			}
			if err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("span %#x+%d is in no mapping", sp.addr, sp.n)
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
	if s.file != nil {
		_ = s.file.Close()
	}
}
