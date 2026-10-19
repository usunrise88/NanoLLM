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
	for i := 0; i < 200; i++ {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Errorf("%s did not happen within 2s", what)
}

func TestStreamerReadsAheadAndEvictsBehind(t *testing.T) {
	const layer = 8 << 20
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
