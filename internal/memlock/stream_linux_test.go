//go:build linux

package memlock

import (
	"os"
	"slices"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

var sink byte

// eventually polls cond: readahead is asynchronous.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for i := 0; i < 1000; i++ {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Errorf("%s did not happen within 10s", what)
}

func TestStreamerReadsAheadAndEvictsBehind(t *testing.T) {
	// Several times what one WILLNEED call reads on most disks (128 KiB to
	// 8 MiB): the streamer has to cut a layer into calls to read it whole.
	const layer = 32 << 20
	path := tempFile(t, 4*layer)
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	m, err := unix.Mmap(int(f.Fd()), 0, 4*layer, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Munmap(m)
	// Layers are not page-aligned: an unaligned start and end.
	rng := func(l int) Range { return Range{Off: int64(l*layer) + 100, Len: layer - 300} }
	layers := map[int][]Range{0: {rng(0)}, 1: {rng(1)}, 2: {rng(2)}, 3: {rng(3)}}
	view := func(l int) []byte { return m[l*layer : (l+1)*layer] }
	resident := func(l int) int64 {
		n, err := Resident(view(l))
		if err != nil {
			t.Fatal(err)
		}
		return n
	}
	touch := func(l int) {
		var sum byte
		for i := 0; i < layer; i += int(PageSize) {
			sum += view(l)[i]
		}
		sink += sum // a load nobody uses would be optimised away
	}
	reset := func() {
		for l := range 4 {
			if err := evict(view(l), f.Fd(), int64(l*layer)); err != nil {
				t.Fatal(err)
			}
		}
		for l := range 4 {
			if resident(l) > layer/2 {
				t.Skip("this filesystem keeps the pages after eviction; the test would prove nothing")
			}
		}
	}

	t.Run("ahead", func(t *testing.T) {
		reset()
		// Some filesystems and sandboxes ignore MADV_WILLNEED; then there is
		// nothing for the streamer to be tested against. One call reads no
		// more than the device's readahead limit, so the probe asks in pieces.
		lim := max(int(ReadaheadLimit(path))&^(int(PageSize)-1), int(PageSize))
		for o := 0; o < layer; o += lim {
			if err := unix.Madvise(view(3)[o:min(o+lim, layer)], unix.MADV_WILLNEED); err != nil {
				t.Skipf("madvise(WILLNEED): %v", err)
			}
		}
		honoured := false
		for i := 0; i < 500 && !honoured; i++ {
			honoured = resident(3) > layer/2
			time.Sleep(10 * time.Millisecond)
		}
		if !honoured {
			t.Skip("this filesystem does not read ahead on MADV_WILLNEED")
		}
		reset()
		s, err := NewStreamer(path, layers, StreamOptions{Ahead: 1})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		s.Layer(0)
		eventually(t, "layer 1 to be read ahead", func() bool { return resident(1) > layer*9/10 })
		time.Sleep(50 * time.Millisecond)
		if r := resident(3); r > layer/2 {
			t.Errorf("layer 3 is resident (%d bytes) though only one layer ahead was asked", r)
		}
		// The next token's first layer comes after the last layer.
		s.Layer(3)
		eventually(t, "layer 0 to be read ahead after layer 3 (wrap)", func() bool { return resident(0) > layer*9/10 })
		if st := s.Stats(); st.Errors != 0 || st.WillNeed == 0 {
			t.Errorf("stats %+v", st)
		}
	})

	t.Run("pread ahead", func(t *testing.T) {
		reset()
		// The reads are the streamer's own: no filesystem can ignore them, and
		// layers wider than one chunk (4 MiB) are read in several, at a depth of 3.
		s, err := NewStreamer(path, layers, StreamOptions{Ahead: 1, Method: MethodPread, Depth: 3})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		s.Layer(0)
		eventually(t, "layer 1 to be read by pread", func() bool { return resident(1) > layer*9/10 })
		if r := resident(3); r > layer/2 {
			t.Errorf("layer 3 is resident (%d bytes) though only one layer ahead was asked", r)
		}
		s.Layer(3) // wraps to layer 0
		eventually(t, "layer 0 to be read by pread after layer 3 (wrap)", func() bool { return resident(0) > layer*9/10 })
		if st := s.Stats(); st.Errors != 0 || st.WillNeed < layer {
			t.Errorf("stats %+v", st)
		}
		if _, err := NewStreamer(path, layers, StreamOptions{Ahead: 1, Method: "readv"}); err == nil {
			t.Error("an unknown method was accepted")
		}
	})

	t.Run("evict behind, keep the first", func(t *testing.T) {
		reset()
		for l := range 4 {
			touch(l)
		}
		s, err := NewStreamer(path, layers, StreamOptions{Evict: true, Keep: 1})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		s.Layer(0)
		s.Layer(1)
		eventually(t, "layer 1 to be dropped", func() bool { return resident(1) < layer/10 })
		if r := resident(0); r < layer*9/10 {
			t.Errorf("layer 0 is a kept layer but only %d bytes of it are resident", r)
		}
		if r := resident(2); r < layer*9/10 {
			t.Errorf("layer 2 was not computed yet but only %d bytes of it are resident", r)
		}
		if st := s.Stats(); st.PageOut == 0 {
			t.Errorf("stats %+v", st)
		}
	})
}

func TestStreamerRefusesAFileThatIsNotMapped(t *testing.T) {
	path := tempFile(t, 1<<20)
	if _, err := NewStreamer(path, map[int][]Range{0: {{0, 4096}}}, StreamOptions{Ahead: 1}); err == nil {
		t.Error("a streamer for an unmapped file")
	}
}

func TestLayerNotificationNeverBlocks(t *testing.T) {
	path := tempFile(t, 1<<20)
	f, _ := os.Open(path)
	defer f.Close()
	m, err := unix.Mmap(int(f.Fd()), 0, 1<<20, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Munmap(m)
	s, err := NewStreamer(path, map[int][]Range{0: {{0, 1 << 20}}}, StreamOptions{Ahead: 1, Evict: true})
	if err != nil {
		t.Fatal(err)
	}
	s.Close() // the goroutine is gone: the queue fills and the calls must not block
	done := make(chan struct{})
	go func() {
		for range 1000 {
			s.Layer(0)
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Layer blocked")
	}
	if s.Stats().Dropped == 0 {
		t.Error("nothing was dropped from a full queue")
	}
}

func TestSplitSpans(t *testing.T) {
	p := uintptr(PageSize)
	in := []span{{0, 10 * p}, {20 * p, 3 * p}, {40 * p, 100}}
	// The limit is rounded down to whole pages.
	got := splitSpans(in, 4*p+5)
	want := []span{{0, 4 * p}, {4 * p, 4 * p}, {8 * p, 2 * p}, {20 * p, 3 * p}, {40 * p, 100}}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestReadaheadLimit(t *testing.T) {
	if n := ReadaheadLimit(tempFile(t, 1)); n < 128<<10 {
		t.Errorf("ReadaheadLimit = %d, want at least the 128 KiB fallback", n)
	}
	if n := ReadaheadLimit("/does/not/exist"); n != 128<<10 {
		t.Errorf("missing file: %d, want 128 KiB", n)
	}
}

func TestMergeSpans(t *testing.T) {
	in := []span{{1000, 100}, {100, 100}, {210, 40}, {1100, 50}, {1400, 10}}
	got := mergeSpans(append([]span(nil), in...), 20)
	want := []span{{100, 150}, {1000, 150}, {1400, 10}}
	if !slices.Equal(got, want) {
		t.Errorf("gap 20: %v, want %v", got, want)
	}
	got = mergeSpans(append([]span(nil), in...), 0)
	want = []span{{100, 100}, {210, 40}, {1000, 150}, {1400, 10}}
	if !slices.Equal(got, want) {
		t.Errorf("gap 0: %v, want %v", got, want)
	}
}

// A span is rounded out to whole pages, so the last one of a file that is not
// a whole number of pages reaches past its end: reading it is not an error.
func TestStreamerPreadAtTheEndOfTheFile(t *testing.T) {
	const size = 8<<20 + 1000
	path := tempFile(t, size)
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	m, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Munmap(m)
	s, err := NewStreamer(path, map[int][]Range{0: {{Off: 0, Len: size}}, 1: {{Off: 0, Len: 4096}}}, StreamOptions{Ahead: 1, Method: MethodPread})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Layer(1) // layer 0 is next
	eventually(t, "layer 0 to be read", func() bool { return s.Stats().WillNeed >= 8<<20 })
	if st := s.Stats(); st.Errors != 0 {
		t.Errorf("the end of the file counted as an error: %+v", st)
	}
}
