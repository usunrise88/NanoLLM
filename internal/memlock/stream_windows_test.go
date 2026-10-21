//go:build windows

package memlock

import (
	"os"
	"testing"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// mapWhole maps the file the way llama.cpp does: one read-only view from offset 0.
func mapWhole(t *testing.T, path string) []byte {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	st, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	h, err := windows.CreateFileMapping(windows.Handle(f.Fd()), nil, windows.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = windows.CloseHandle(h) })
	addr, err := windows.MapViewOfFile(h, windows.FILE_MAP_READ, 0, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = windows.UnmapViewOfFile(addr) })
	ptr := *(*unsafe.Pointer)(unsafe.Pointer(&addr))
	return unsafe.Slice((*byte)(ptr), st.Size())
}

var sink byte

func touchAll(b []byte) {
	var sum byte
	for i := 0; i < len(b); i += int(PageSize) {
		sum += b[i]
	}
	sink += sum // a load nobody uses would be optimised away
}

func TestFileMappingsFindsTheModelView(t *testing.T) {
	const size = 8 << 20
	path := tempFile(t, size)
	if ms, err := fileMappings(path); err != nil || len(ms) != 0 {
		t.Fatalf("before the mapping: %v, %v", ms, err)
	}
	m := mapWhole(t, path)
	ms, err := fileMappings(path)
	if err != nil {
		t.Fatal(err)
	}
	base := uint64(uintptr(unsafe.Pointer(&m[0])))
	if len(ms) != 1 || ms[0].start != base || ms[0].end-ms[0].start < size || ms[0].off != 0 {
		t.Fatalf("mappings %+v, want one view at %#x of at least %d bytes", ms, base, size)
	}
}

func TestStreamerEvictsBehindAndPrefetchesAhead(t *testing.T) {
	const layer = 4 << 20
	path := tempFile(t, 4*layer)
	m := mapWhole(t, path)
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

	s, err := NewStreamer(path, layers, StreamOptions{Ahead: 1, Evict: true, Keep: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.Layers() != 4 {
		t.Fatalf("Layers() = %d, want 4", s.Layers())
	}
	for l := range 4 {
		touchAll(view(l))
		if resident(l) < layer/2 {
			t.Skip("the working set does not keep touched pages here; the test would prove nothing")
		}
	}
	s.Warm()
	// Layers 2 and 3 are computed: both leave the working set; layer 0 is a Keep layer and stays.
	s.Layer(2)
	s.Layer(3)
	eventually(t, "layers 2 and 3 leave the working set", func() bool {
		return resident(2) < layer/4 && resident(3) < layer/4
	})
	if resident(0) < layer/2 {
		t.Errorf("layer 0 is a Keep layer but only %d of %d bytes are resident", resident(0), layer)
	}
	st := s.Stats()
	if st.Errors != 0 || st.FirstError != nil {
		t.Errorf("errors: %+v", st)
	}
	if st.PageOut == 0 || st.WillNeed == 0 {
		t.Errorf("counters: %+v, want both prefetch and trim bytes", st)
	}
}

func TestStreamerRefusesAFileThatIsNotMapped(t *testing.T) {
	path := tempFile(t, 1<<20)
	if _, err := NewStreamer(path, map[int][]Range{0: {{0, 4096}}}, StreamOptions{Ahead: 1}); err == nil {
		t.Fatal("a file nobody maps got a streamer")
	}
}

func TestSplitSpansAndMerge(t *testing.T) {
	const p = 4096
	got := splitSpans([]span{{addr: 10 * p, n: 9 * p}}, 4*p+5)
	if len(got) != 3 || got[0].n != 4*p || got[1].addr != 14*p || got[2].n != p {
		t.Errorf("splitSpans = %+v", got)
	}
	m := mergeSpans([]span{{addr: 100, n: 10}, {addr: 115, n: 10}, {addr: 200, n: 5}}, 20)
	if len(m) != 2 || m[0].n != 25 {
		t.Errorf("mergeSpans = %+v", m)
	}
}

// eventually polls cond: the streamer works on a goroutine of its own.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for i := 0; i < 1000; i++ {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out: %s", what)
}
