//go:build native

package llamacpp

import (
	"context"
	"slices"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/gguf"
)

// OPT-8: a model that streams its layers through the eval callback computes
// exactly what one that does not stream computes, and the streamer is fed.
func TestNativeLayerStreaming(t *testing.T) {
	b := testBackend(t)
	path := testModelPath(t)
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	inv, err := gguf.BuildInventory(f)
	if err != nil {
		t.Fatal(err)
	}
	var ranges []backend.StreamRange
	layers := map[int]bool{}
	for _, tn := range inv.Tensors {
		if tn.Layer >= 0 && tn.Group != gguf.G1 && tn.Group != gguf.G4 {
			ranges = append(ranges, backend.StreamRange{Layer: tn.Layer, Offset: tn.Offset, Length: tn.Bytes})
			layers[tn.Layer] = true
		}
	}
	if len(ranges) == 0 {
		t.Skip("no layer tensors in the test model")
	}

	// File-backed weights: mmap, no repack.
	spec := baseSpec(path)
	spec.Repack = false
	spec.LoadMode = backend.LoadMMap
	spec.Slots = 1

	run := func(st *backend.StreamSpec) (*model, []backend.Token) {
		s := spec
		s.Stream = st
		mm, err := b.Load(context.Background(), s)
		if err != nil {
			t.Fatal(err)
		}
		m := mm.(*model)
		prompt := tokenize(t, m, "The three primary colours are")
		return m, decode(t, m, 0, prompt, 24, greedy)
	}

	base, want := run(nil)
	base.Close()

	for name, st := range map[string]*backend.StreamSpec{
		"ahead":       {Ranges: ranges, Ahead: 2},
		"ahead+evict": {Ranges: ranges, Ahead: 2, Evict: true, Keep: 2},
		"probe":       {Ranges: ranges, Probe: true},
	} {
		t.Run(name, func(t *testing.T) {
			m, got := run(st)
			defer m.Close()
			if !slices.Equal(got, want) {
				t.Errorf("streaming changed the output:\n got %v\nwant %v", got, want)
			}
			s := m.StreamStats()
			t.Logf("stats %+v; log: %v", s, m.loadLog[max(0, len(m.loadLog)-1):])
			if name == "probe" {
				return
			}
			if s.Calls == 0 || s.WillNeed == 0 {
				t.Errorf("the streamer was never fed: %+v", s)
			}
			if st.Evict && s.PageOut == 0 {
				t.Errorf("nothing was evicted: %+v", s)
			}
			if s.Errors != 0 {
				t.Errorf("madvise failed: %+v", s)
			}
		})
	}
}
