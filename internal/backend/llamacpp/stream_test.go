package llamacpp

import (
	"sync"
	"testing"
	"unsafe"
)

func nameBuf(s string) *[64]byte {
	var b [64]byte
	copy(b[:], s)
	return &b
}

func TestLayerOfNode(t *testing.T) {
	for name, want := range map[string]int{
		"l_out-0": 0, "l_out-7": 7, "l_out-41": 41, "l_out-123": 123,
		"l_out": -1, "l_out-": -1, "l_out-x": -1, "l_out-4x": -1, "l_out--1": -1,
		"l_out-12345": -1, "attn_out-3": -1, "": -1, "result_output": -1, "l_outer-3": -1,
	} {
		if got := layerOfNode(unsafe.Pointer(nameBuf(name))); got != want {
			t.Errorf("layerOfNode(%q) = %d, want %d", name, got, want)
		}
	}
}

type spy struct {
	mu  sync.Mutex
	got []int
}

func (s *spy) Layer(il int) {
	s.mu.Lock()
	s.got = append(s.got, il)
	s.mu.Unlock()
}

// The callback is C's: it gets a tensor pointer whose name sits at
// tensorNameOffset. A fake tensor of that shape stands for it.
func TestOnNode(t *testing.T) {
	var fake [tensorNameOffset + 64]byte
	setName := func(s string) {
		clear(fake[tensorNameOffset:])
		copy(fake[tensorNameOffset:], s)
	}
	sp := &spy{}
	streams.Store(uintptr(9001), layerSink(sp))
	defer streams.Delete(uintptr(9001))
	call := func(name string, ask uintptr) uintptr {
		setName(name)
		return onNode(&fake[0], ask, 9001)
	}

	if got := call("attn_out-3", 1); got != 0 {
		t.Errorf("a node that is no layer end was asked for: %d", got)
	}
	if got := call("l_out-3", 1); got != 1 {
		t.Errorf("the layer end was not asked for: %d", got)
	}
	sp.got = nil
	call("l_out-5", 0)
	call("attn_out-6", 0)
	call("l_out-6", 0xffffff00)
	if len(sp.got) != 2 || sp.got[0] != 5 || sp.got[1] != 6 {
		t.Errorf("notified layers %v, want [5 6]", sp.got)
	}
	// An unknown user_data and a probe (nil sink) are not errors.
	setName("l_out-1")
	if onNode(&fake[0], 0, 4242) != 1 {
		t.Error("an unknown user_data broke the graph")
	}
	streams.Store(uintptr(9002), nil)
	defer streams.Delete(uintptr(9002))
	if onNode(&fake[0], 0, 9002) != 1 {
		t.Error("a probe broke the graph")
	}
	if onNode(nil, 1, 9001) != 0 {
		t.Error("nil tensor")
	}
}
