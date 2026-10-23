package llamacpp

import (
	"fmt"
	"unsafe"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/memlock"
)

// The layer ring (OPT-9, variant "direct"; internal/memlock/ring_linux.go).
// Instead of asking the kernel to read the file-backed layers ahead, the ring
// reads them with O_DIRECT into buffers of its own, and the model's tensors
// are pointed at those buffers: while the layer computes, the tensor's data
// pointer is the ring's. The tensors are found by name through
// llama_model_get_tensor (patches/next/0010), the stream spec's ranges are
// matched to them by file offset, and a ggml_tensor's data pointer is written
// in place (the offset is checked against the library in a native test).

// tensorDataOffset is offsetof(struct ggml_tensor, data) in ggml b11600.
const tensorDataOffset = 248

// ringTensor is a tensor the ring points elsewhere: where its data pointer
// lives and what it was.
type ringTensor struct {
	slot     *uintptr // the tensor's data field
	original uintptr  // the file mapping's address
}

// attachRing builds the ring for a stream spec. It runs after the model is
// loaded (the tensors exist, mapped from the file) and returns an error that
// leaves the model as it was.
func (m *model) attachRing(spec *backend.StreamSpec) (*memlock.Ring, error) {
	if !m.b.lib.cxx["llama_model_get_tensor"] {
		return nil, fmt.Errorf("the library has no llama_model_get_tensor (patches/next/0010)")
	}
	f, err := gguf.Open(m.spec.Path)
	if err != nil {
		return nil, err
	}
	byOffset := map[int64]gguf.TensorInfo{}
	for _, t := range f.Tensors {
		byOffset[f.FileOffset(t)] = t
	}
	var (
		tensors []memlock.RingTensor
		held    []ringTensor
	)
	for _, r := range spec.Ranges {
		t, ok := byOffset[r.Offset]
		if !ok || t.Size != r.Length {
			return nil, fmt.Errorf("the range of layer %d at %d (%d bytes) is not a tensor of the file", r.Layer, r.Offset, r.Length)
		}
		p := llamaModelGetTensor(m.mdl, t.Name)
		if p == nil {
			return nil, fmt.Errorf("the model has no tensor %s", t.Name)
		}
		slot := (*uintptr)(unsafe.Add(p, tensorDataOffset))
		if *slot == 0 {
			return nil, fmt.Errorf("tensor %s has no data (it is not mapped from the file)", t.Name)
		}
		tensors = append(tensors, memlock.RingTensor{Layer: r.Layer, Offset: r.Offset, Length: r.Length})
		held = append(held, ringTensor{slot: slot, original: *slot})
	}
	ring, err := memlock.NewRing(m.spec.Path, tensors, memlock.RingOptions{
		Ahead: spec.Ahead, Depth: spec.Depth, Keep: spec.Keep, Layers: int(llamaModelNLayer(m.mdl)),
		Retarget: func(i int, addr uintptr) {
			if addr == 0 {
				addr = held[i].original
			}
			*held[i].slot = addr
		},
	})
	if err != nil {
		return nil, err
	}
	return ring, nil
}
