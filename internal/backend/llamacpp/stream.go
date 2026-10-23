package llamacpp

import (
	"cmp"
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/memlock"
)

// Layer streaming (OPT-8, docs/STAGE2.md). A plan that leaves layer weights
// in the file (M8's cold FFN tensors, M9) reads them from the disk on every
// token, and the page cache's LRU makes a cyclic scan miss every time. ggml
// tells us when a node of the graph is done through the context's eval
// callback; the node "l_out-N" ends layer N, and from there the streamer
// (internal/memlock) asks the kernel for the next layers and drops the one
// just computed. The callback is only installed for a context that streams:
// asking the scheduler for a callback splits the graph at every node the
// callback wants, which costs a barrier per layer.

var (
	streamOnce sync.Once
	streamCb   uintptr
	streamNext atomic.Uintptr
	// streams maps the callback's user_data to a layerSink: the streamer, or
	// nil for a probe.
	streams sync.Map
)

// layerSink hears that a layer has been computed.
type layerSink interface{ Layer(il int) }

// ggml_tensor's name field: offsetof(struct ggml_tensor, name) in ggml b11600
// (ggml.h; checked against the header by TestTensorNameOffset when native).
const tensorNameOffset = 256

// layerOfNode is the layer a graph node ends, from the node's name:
// "l_out-12" ends layer 12, any other node, -1.
func layerOfNode(name unsafe.Pointer) int {
	const prefix = "l_out-"
	for i := 0; i < len(prefix); i++ {
		if *(*byte)(unsafe.Add(name, i)) != prefix[i] {
			return -1
		}
	}
	n := 0
	for i := len(prefix); i < len(prefix)+4; i++ {
		c := *(*byte)(unsafe.Add(name, i))
		if c == 0 {
			if i == len(prefix) {
				return -1
			}
			return n
		}
		if c < '0' || c > '9' {
			return -1
		}
		n = n*10 + int(c-'0')
	}
	return -1
}

// onNode is ggml_backend_sched_eval_callback, shared by every streaming
// context: user_data names the streamer. Asked, it wants the data of the
// layer-end nodes only, which makes the scheduler stop at each of them; told,
// it passes the layer on and lets the graph go on.
func onNode(t *byte, ask, ud uintptr) uintptr {
	if t == nil {
		return 0
	}
	il := layerOfNode(unsafe.Add(unsafe.Pointer(t), tensorNameOffset))
	if il < 0 {
		return 0
	}
	if ask&0xff != 0 {
		return 1
	}
	if s, ok := streams.Load(ud); ok {
		if sink, _ := s.(layerSink); sink != nil {
			sink.Layer(il)
		}
	}
	return 1
}

func streamCallbackPtr() uintptr {
	streamOnce.Do(func() {
		streamCb = purego.NewCallback(onNode)
	})
	return streamCb
}

// attachStream starts a streamer for the model's file and points the context
// parameters at the callback. It runs after the model is loaded (the file is
// mapped by then) and before the context is created. A spec that cannot be
// served — the file is not mapped, the platform has no streamer — leaves the
// model as it was and says why in the load log.
func (m *model) attachStream(spec *backend.StreamSpec, cp *LlamaContextParams) {
	note := func(format string, a ...any) {
		m.loadLog = append(m.loadLog, "nanollm: layer streaming "+fmt.Sprintf(format, a...))
	}
	var s *memlock.Streamer
	if spec.Ahead > 0 || spec.Evict {
		layers := map[int][]memlock.Range{}
		for _, r := range spec.Ranges {
			layers[r.Layer] = append(layers[r.Layer], memlock.Range{Off: r.Offset, Len: r.Length})
		}
		var err error
		s, err = memlock.NewStreamer(m.spec.Path, layers, memlock.StreamOptions{Ahead: spec.Ahead, Keep: spec.Keep, Evict: spec.Evict, Method: spec.Method, Depth: spec.Depth})
		if err != nil {
			note("is off: %v", err)
			if !spec.Probe {
				return
			}
			s = nil
		}
	}
	id := streamNext.Add(1)
	if s != nil {
		streams.Store(id, layerSink(s))
	} else {
		streams.Store(id, nil)
	}
	cp.CbEval = streamCallbackPtr()
	cp.CbEvalUserData = id
	m.streamID, m.stream = id, s
	if s != nil {
		s.Warm()
		note("is on: %d layers, ahead %d, keep %d, evict %v, method %s", s.Layers(), spec.Ahead, spec.Keep, spec.Evict, cmp.Or(spec.Method, memlock.MethodAdvise))
	} else {
		note("probe: the callback is installed and does nothing")
	}
}

// detachStream stops the streamer; Close calls it before the context goes.
func (m *model) detachStream() {
	if m.streamID == 0 {
		return
	}
	streams.Delete(m.streamID)
	if m.stream != nil {
		m.stream.Close()
	}
	m.streamID, m.stream = 0, nil
}

// StreamStats is what the layer streamer has asked of the kernel so far; the
// zero value when the model does not stream.
func (m *model) StreamStats() memlock.StreamStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.stream == nil {
		return memlock.StreamStats{}
	}
	return m.stream.Stats()
}
