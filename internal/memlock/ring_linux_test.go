//go:build linux

package memlock

import (
	"bytes"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"unsafe"
)

// ringFile is a file of pseudo-random bytes in a directory that takes O_DIRECT
// (the test skips where it does not).
func ringFile(t *testing.T, size int) (string, []byte) {
	t.Helper()
	data := make([]byte, size)
	rng := rand.New(rand.NewPCG(7, 11))
	for i := range data {
		data[i] = byte(rng.Uint32())
	}
	path := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path, data
}

// layersOf lays n layers of three tensors each over the file at offsets that
// are not multiples of the block, with a few kilobytes between the tensors.
func layersOf(n int) []RingTensor {
	var out []RingTensor
	off := int64(100)
	for l := range n {
		for _, size := range []int64{300_000, 120_001, 700_003} {
			out = append(out, RingTensor{Layer: l, Offset: off, Length: size})
			off += size + 10_240
		}
	}
	return out
}

// walk plays what the eval callback does for a model of n layers: layer il has
// been computed, tell the ring. Before each layer's turn its tensors have to
// hold the file's bytes.
func walk(t *testing.T, r *Ring, ts []RingTensor, data []byte, addr []uintptr, n, tokens int) {
	t.Helper()
	for tok := range tokens {
		for il := range n {
			for i, tn := range ts {
				if tn.Layer != il || addr[i] == 0 {
					continue
				}
				got := unsafe.Slice((*byte)(unsafe.Pointer(addr[i])), tn.Length) //nolint:govet // addr is a pointer into the ring's mapping, kept alive by r
				if !bytes.Equal(got, data[tn.Offset:tn.Offset+tn.Length]) {
					t.Fatalf("token %d, layer %d, tensor %d: the bytes differ from the file's", tok, il, i)
				}
			}
			r.Layer(il)
		}
	}
}

func newRingForTest(t *testing.T, n int, opt RingOptions) (*Ring, []RingTensor, []byte, []uintptr) {
	t.Helper()
	ts := layersOf(n)
	path, data := ringFile(t, int(ts[len(ts)-1].Offset+ts[len(ts)-1].Length)+5000)
	addr := make([]uintptr, len(ts))
	opt.Layers = n
	opt.Retarget = func(i int, a uintptr) { addr[i] = a }
	r, err := NewRing(path, ts, opt)
	if err != nil {
		t.Skipf("no direct reads here: %v", err)
	}
	t.Cleanup(r.Close)
	return r, ts, data, addr
}

// The tensors of the layer that is about to compute hold the file's bytes, over
// several tokens, with a ring of three buffers for eight layers (a buffer is
// reused five times per token) and with one buffer per layer.
func TestRingServesTheLayersInOrder(t *testing.T) {
	for name, opt := range map[string]RingOptions{
		"a small ring": {Ahead: 1, Depth: 3},
		"deeper":       {Ahead: 3, Depth: 4},
		"all resident": {Ahead: 20, Depth: 2},
		"no read-ahead": {Ahead: 0, Depth: 1},
	} {
		t.Run(name, func(t *testing.T) {
			r, ts, data, addr := newRingForTest(t, 8, opt)
			r.Warm()
			walk(t, r, ts, data, addr, 8, 3)
			st := r.Stats()
			if st.Err != nil || st.Reads == 0 {
				t.Fatalf("stats %+v", st)
			}
			if name == "all resident" && st.Reads != int64(len(ts)) {
				t.Errorf("with a buffer per layer each tensor is read once, not %d times for %d tensors", st.Reads, len(ts))
			}
			if name == "a small ring" && st.Reads != int64(3*len(ts)) {
				t.Errorf("reads %d, want %d: every tensor once per token", st.Reads, 3*len(ts))
			}
		})
	}
}

// Layers below Keep stay on the file mapping: the ring never points their
// tensors anywhere.
func TestRingKeepsTheLeadingLayers(t *testing.T) {
	r, ts, data, addr := newRingForTest(t, 6, RingOptions{Ahead: 1, Depth: 2, Keep: 2})
	r.Warm()
	walk(t, r, ts, data, addr, 6, 2)
	for i, tn := range ts {
		if tn.Layer < 2 && addr[i] != 0 {
			t.Errorf("tensor %d of layer %d was retargeted", i, tn.Layer)
		}
	}
	if st := r.Stats(); st.Reads != int64(2*4*3) {
		t.Errorf("reads %d, want %d (four managed layers, three tensors, two tokens)", st.Reads, 2*4*3)
	}
}

// Close gives every tensor back to the file mapping.
func TestRingGivesTheTensorsBack(t *testing.T) {
	r, ts, data, addr := newRingForTest(t, 4, RingOptions{Ahead: 1, Depth: 2})
	r.Warm()
	walk(t, r, ts, data, addr, 4, 1)
	r.Close()
	for i := range ts {
		if addr[i] != 0 {
			t.Errorf("tensor %d still points into the ring", i)
		}
	}
}

// A read that fails stops the ring and hands every tensor back: the model goes
// on from the file mapping.
func TestRingFailureFallsBackToTheMapping(t *testing.T) {
	ts := layersOf(3)
	path, _ := ringFile(t, int(ts[len(ts)-1].Offset+ts[len(ts)-1].Length)+5000)
	ts[7].Offset += 1 << 24 // layer 2's second tensor lies far past the end of the file
	addr := make([]uintptr, len(ts))
	r, err := NewRing(path, ts, RingOptions{Ahead: 1, Depth: 2, Layers: 3, Retarget: func(i int, a uintptr) { addr[i] = a }})
	if err != nil {
		t.Skipf("no direct reads here: %v", err)
	}
	defer r.Close()
	r.Warm()
	for il := range 3 {
		r.Layer(il)
	}
	if st := r.Stats(); st.Err == nil {
		t.Fatalf("a read past the end of the file did not fail the ring: %+v", st)
	}
	for i := range ts {
		if addr[i] != 0 {
			t.Errorf("tensor %d still points into the ring after the failure", i)
		}
	}
	r.Layer(0) // a failed ring ignores the model
}

func TestRingRefusesWhatItCannotServe(t *testing.T) {
	path, _ := ringFile(t, 1<<20)
	retarget := func(int, uintptr) {}
	for name, c := range map[string]struct {
		ts  []RingTensor
		opt RingOptions
	}{
		"no retarget":       {layersOf(2), RingOptions{Layers: 2}},
		"a layer past the model": {[]RingTensor{{Layer: 5, Offset: 0, Length: 10}}, RingOptions{Layers: 2, Retarget: retarget}},
		"an empty tensor":   {[]RingTensor{{Layer: 0, Offset: 0, Length: 0}}, RingOptions{Layers: 2, Retarget: retarget}},
		"everything kept":   {layersOf(2), RingOptions{Layers: 2, Keep: 2, Retarget: retarget}},
	} {
		if r, err := NewRing(path, c.ts, c.opt); err == nil {
			r.Close()
			t.Errorf("%s: accepted", name)
		}
	}
}
