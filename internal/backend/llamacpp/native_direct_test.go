//go:build native

package llamacpp

import (
	"context"
	"runtime"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/gguf"
)

func nativeRingSetup(t *testing.T) (*llamaBackend, string, []backend.StreamRange) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skip("the layer ring reads with O_DIRECT; Linux only for now")
	}
	b := testBackend(t)
	if !b.lib.cxx["llama_model_get_tensor"] {
		t.Skip("the libraries have no llama_model_get_tensor (patches/llama.cpp/0010)")
	}
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
	for _, tn := range inv.Tensors {
		// The first six layers: O_DIRECT reads from the disk every time, and a hard disk
		// reads the whole model (4 GB a token) at 100 MB/s.
		if tn.Layer >= 0 && tn.Layer < 6 && tn.Group != gguf.G1 && tn.Group != gguf.G4 {
			ranges = append(ranges, backend.StreamRange{Layer: tn.Layer, Offset: tn.Offset, Length: tn.Bytes})
		}
	}
	if len(ranges) == 0 {
		t.Skip("no layer tensors in the test model")
	}
	return b, path, ranges
}

// A tensor found by name is the model's own: its name and data pointer lie at
// the offsets the ring writes through.
func TestNativeTensorLayout(t *testing.T) {
	b, path, _ := nativeRingSetup(t)
	spec := baseSpec(path)
	spec.Repack, spec.LoadMode, spec.Slots = false, backend.LoadMMap, 1
	mm, err := b.Load(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	m := mm.(*model)
	defer m.Close()
	p := llamaModelGetTensor(m.mdl, "blk.3.ffn_up.weight")
	if p == 0 {
		t.Fatal("no tensor blk.3.ffn_up.weight")
	}
	var name []byte
	for i := 0; ; i++ {
		c := *(*byte)(unsafe.Add(tensorPtr(p), tensorNameOffset+i))
		if c == 0 || i > 63 {
			break
		}
		name = append(name, c)
	}
	if string(name) != "blk.3.ffn_up.weight" {
		t.Errorf("the name at offset %d is %q", tensorNameOffset, name)
	}
	if data := *(*uintptr)(unsafe.Add(tensorPtr(p), tensorDataOffset)); data == 0 {
		t.Error("a mapped tensor has no data pointer at the offset the ring writes")
	}
	if llamaModelGetTensor(m.mdl, "no.such.tensor") != 0 {
		t.Error("a tensor that does not exist was found")
	}
}

// OPT-9, method direct: a model whose file-backed layers come through the ring
// computes exactly what the model that reads them from the mapping computes,
// and the ring is fed.
func TestNativeLayerRing(t *testing.T) {
	b, path, ranges := nativeRingSetup(t)
	spec := baseSpec(path)
	spec.Repack, spec.LoadMode, spec.Slots = false, backend.LoadMMap, 1
	run := func(st *backend.StreamSpec) (*model, []backend.Token) {
		s := spec
		s.Stream = st
		mm, err := b.Load(context.Background(), s)
		if err != nil {
			t.Fatal(err)
		}
		m := mm.(*model)
		return m, decode(t, m, 0, tokenize(t, m, "The three primary colours are"), 8, greedy)
	}
	base, want := run(nil)
	base.Close()

	for name, st := range map[string]*backend.StreamSpec{
		"ahead 2":       {Ranges: ranges, Ahead: 2, Method: "direct", Depth: 3},
		"no read-ahead": {Ranges: ranges, Ahead: 0, Method: "direct", Depth: 1},
		"keep 4":        {Ranges: ranges, Ahead: 3, Keep: 4, Method: "direct"},
	} {
		t.Run(name, func(t *testing.T) {
			m, got := run(st)
			defer m.Close()
			if !slices.Equal(got, want) {
				t.Errorf("the ring changed the output:\n got %v\nwant %v", got, want)
			}
			rs := m.RingStats()
			t.Logf("ring %+v; log: %v", rs, m.loadLog[max(0, len(m.loadLog)-1):])
			if rs.Reads == 0 || rs.Err != nil {
				t.Errorf("the ring was not fed, or failed: %+v", rs)
			}
			if len(m.loadLog) == 0 || !strings.Contains(m.loadLog[len(m.loadLog)-1], "direct reads") {
				// the note is among the last lines of the load log
				found := false
				for _, l := range m.loadLog {
					found = found || strings.Contains(l, "layer streaming is on: direct reads")
				}
				if !found {
					t.Errorf("no note that the ring is on: %v", m.loadLog[max(0, len(m.loadLog)-3):])
				}
			}
		})
	}
}
