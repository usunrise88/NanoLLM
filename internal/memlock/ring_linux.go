//go:build linux

package memlock

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

// The layer ring (OPT-9, variant "direct"; docs/AUTOTUNE.md). A plan that
// leaves layer weights in the file lets the page cache serve them, and the
// streamer (streamer.go) asks the kernel to read ahead. The ring takes the
// page cache out: the weights of the layers it manages are read with O_DIRECT,
// several reads at a time, into a few buffers of its own, and the model's
// tensors are pointed at those buffers for the layer that is computing. The
// disk is kept busy at a depth of its own choosing, nothing is cached twice,
// the memory the weights take is the ring and not whatever the cache grows to,
// and the pages of the file mapping are never touched.
//
// The model asks nothing of the ring but two things, both from the thread that
// computes the graph, between two layers: Layer(il) says layer il is done, and
// the ring makes the next layer's buffer ready (waiting for the disk if it
// must) and points that layer's tensors at it through Retarget.

// RingTensor is a tensor of a managed layer: where its bytes lie in the file.
type RingTensor struct {
	Layer          int
	Offset, Length int64
}

// RingOptions is how a Ring runs.
type RingOptions struct {
	// Ahead is how many layers after the one computing are read in advance;
	// the ring has Ahead+2 buffers.
	Ahead int
	// Depth is the reads in flight; zero is four.
	Depth int
	// Keep is the number of leading layers the ring leaves alone (their
	// tensors stay on the file mapping and the page cache).
	Keep int
	// Layers is the number of layers of the model.
	Layers int
	// Retarget points tensor i (the index in the slice given to NewRing) at
	// the bytes at addr; addr 0 points it back at the file mapping.
	Retarget func(i int, addr uintptr)
}

// ringBlock is the alignment O_DIRECT wants of offsets, lengths and buffers.
const ringBlock = 4096

type ringPlace struct {
	tensor  int   // index in the slice given to NewRing
	readOff int64 // file offset of the aligned read
	readLen int64
	slotOff int64 // where the read lands in the slot
	dataOff int64 // where the tensor's first byte lies in the slot
}

type ringSlot struct {
	mu      sync.Mutex
	cond    *sync.Cond
	layer   int // the layer whose bytes the slot holds or is loading, -1 when none
	loading bool
	done    bool // the layer has been computed: the slot may be reused
	pending int
	err     error
	base    []byte
}

// Ring reads the managed layers of a model file into a ring of buffers.
type Ring struct {
	opt     RingOptions
	fd      int
	tensors []RingTensor
	layers  [][]ringPlace // by layer; nil for a layer the ring does not manage
	order   []int         // the managed layers, ascending
	pos     map[int]int   // layer -> index in order
	slots   []ringSlot
	slotLen int64

	jobs chan ringJob
	wg   sync.WaitGroup
	stop chan struct{}

	failed atomic.Bool
	errMu  sync.Mutex
	err    error

	reads, bytes, stalls, skipped atomic.Int64
	stallNS                       atomic.Int64
}

type ringJob struct {
	layer int
	place ringPlace
}

// NewRing opens the file with O_DIRECT and lays the managed layers out in
// slots. Tensors of layers below opt.Keep are left to the file mapping.
func NewRing(path string, tensors []RingTensor, opt RingOptions) (*Ring, error) {
	if opt.Retarget == nil || opt.Layers <= 0 {
		return nil, errors.New("memlock: a ring needs Retarget and the number of layers")
	}
	if opt.Ahead < 0 {
		opt.Ahead = 0
	}
	if opt.Depth <= 0 {
		opt.Depth = 4
	}
	r := &Ring{opt: opt, tensors: tensors, layers: make([][]ringPlace, opt.Layers), pos: map[int]int{}, stop: make(chan struct{})}
	for i, t := range tensors {
		if t.Layer < 0 || t.Layer >= opt.Layers || t.Length <= 0 || t.Offset < 0 {
			return nil, fmt.Errorf("memlock: ring tensor %d (layer %d, %d bytes at %d) is out of range", i, t.Layer, t.Length, t.Offset)
		}
		if t.Layer < opt.Keep {
			continue
		}
		r.layers[t.Layer] = append(r.layers[t.Layer], ringPlace{tensor: i})
	}
	for l, ps := range r.layers {
		if len(ps) == 0 {
			continue
		}
		var cursor int64
		for k := range ps {
			t := tensors[ps[k].tensor]
			lo := t.Offset / ringBlock * ringBlock
			hi := (t.Offset + t.Length + ringBlock - 1) / ringBlock * ringBlock
			ps[k].readOff, ps[k].readLen = lo, hi-lo
			ps[k].slotOff, ps[k].dataOff = cursor, cursor+t.Offset-lo
			cursor += hi - lo
		}
		r.slotLen = max(r.slotLen, cursor)
		r.pos[l] = len(r.order)
		r.order = append(r.order, l)
	}
	if len(r.order) == 0 {
		return nil, errors.New("memlock: no layer for the ring to manage")
	}
	n := min(opt.Ahead+2, len(r.order))
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECT|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("memlock: open %s with O_DIRECT: %w", path, err)
	}
	r.fd = fd
	r.slots = make([]ringSlot, n)
	for i := range r.slots {
		s := &r.slots[i]
		s.cond = sync.NewCond(&s.mu)
		s.layer = -1
		if s.base, err = unix.Mmap(-1, 0, int(r.slotLen), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE); err != nil {
			r.free()
			return nil, fmt.Errorf("memlock: ring buffer of %d bytes: %w", r.slotLen, err)
		}
	}
	// One small read proves that the file system takes O_DIRECT at all (tmpfs of
	// older kernels, some network and overlay file systems do not).
	if err := r.probe(); err != nil {
		r.free()
		return nil, err
	}
	r.jobs = make(chan ringJob, 4*opt.Depth+64)
	for range opt.Depth {
		r.wg.Add(1)
		go r.worker()
	}
	return r, nil
}

func (r *Ring) probe() error {
	buf := r.slots[0].base[:ringBlock]
	if _, err := unix.Pread(r.fd, buf, 0); err != nil {
		return fmt.Errorf("memlock: a direct read of the model file fails: %w", err)
	}
	return nil
}

func (r *Ring) free() {
	for i := range r.slots {
		if r.slots[i].base != nil {
			_ = unix.Munmap(r.slots[i].base)
			r.slots[i].base = nil
		}
	}
	if r.fd > 0 {
		_ = unix.Close(r.fd)
		r.fd = 0
	}
}

func (r *Ring) slotOf(layer int) *ringSlot { return &r.slots[r.pos[layer]%len(r.slots)] }

func (r *Ring) worker() {
	defer r.wg.Done()
	for {
		select {
		case j := <-r.jobs:
			r.read(j)
		case <-r.stop:
			return
		}
	}
}

// read fills one tensor's part of its layer's slot.
func (r *Ring) read(j ringJob) {
	s := r.slotOf(j.layer)
	dst := s.base[j.place.slotOff : j.place.slotOff+j.place.readLen]
	t := r.tensors[j.place.tensor]
	need := int(t.Offset + t.Length - j.place.readOff) // a read at the end of the file may stop short of the block
	var err error
	for done := 0; done < need; {
		var n int
		n, err = unix.Pread(r.fd, dst[done:], j.place.readOff+int64(done))
		if errors.Is(err, unix.EINTR) {
			err = nil
			continue
		}
		if err != nil || n == 0 {
			if err == nil {
				err = fmt.Errorf("short read: %d of %d bytes", done, need)
			}
			break
		}
		done += n
	}
	r.reads.Add(1)
	r.bytes.Add(j.place.readLen)
	s.mu.Lock()
	if err != nil && s.err == nil {
		s.err = fmt.Errorf("layer %d: %w", j.layer, err)
	}
	if s.pending--; s.pending == 0 {
		s.loading = false
		s.cond.Broadcast()
	}
	s.mu.Unlock()
}

// start begins loading a layer into its slot, unless it is there already or the
// slot is still in use; it reports whether the layer is now loading or loaded.
// wait makes it sleep for a slot that is in use (a demand load), otherwise it
// gives up and counts the skip.
func (r *Ring) start(layer int, wait bool) bool {
	s := r.slotOf(layer)
	s.mu.Lock()
	for {
		if s.layer == layer {
			s.mu.Unlock()
			return true
		}
		if s.layer >= 0 && !s.done || s.loading {
			if !wait {
				r.skipped.Add(1)
				s.mu.Unlock()
				return false
			}
			s.cond.Wait()
			continue
		}
		break
	}
	s.layer, s.loading, s.done, s.err = layer, true, false, nil
	places := r.layers[layer]
	s.pending = len(places)
	s.mu.Unlock()
	for _, p := range places {
		select {
		case r.jobs <- ringJob{layer: layer, place: p}:
		case <-r.stop:
			return false
		}
	}
	return true
}

// ready waits until a layer's bytes are in its slot and returns the slot.
func (r *Ring) ready(layer int) (*ringSlot, error) {
	if !r.start(layer, true) {
		return nil, errors.New("memlock: ring is closing")
	}
	s := r.slotOf(layer)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.loading && s.layer == layer {
		t0 := time.Now()
		for s.loading && s.layer == layer {
			s.cond.Wait()
		}
		r.stalls.Add(1)
		r.stallNS.Add(int64(time.Since(t0)))
	}
	if s.layer != layer {
		return nil, fmt.Errorf("memlock: layer %d was overwritten before it was used", layer)
	}
	return s, s.err
}

// Warm makes the first managed layer ready and starts the ones after it, before
// the first token.
func (r *Ring) Warm() {
	if err := r.enter(r.order[0]); err != nil {
		r.fail(err)
	}
}

// Layers is the number of layers the ring manages.
func (r *Ring) Layers() int { return len(r.order) }

// Layer is told that layer il has been computed. It frees the layer's buffer,
// makes the next managed layer ready and points its tensors at the buffer.
func (r *Ring) Layer(il int) {
	if r.failed.Load() || il < 0 || il >= r.opt.Layers {
		return
	}
	if r.layers[il] != nil {
		s := r.slotOf(il)
		s.mu.Lock()
		if s.layer == il {
			s.done = true
			s.cond.Broadcast()
		}
		s.mu.Unlock()
	}
	next := il + 1
	if next >= r.opt.Layers {
		next = 0
	}
	if r.layers[next] == nil {
		return // a layer left to the file mapping (below Keep, or with nothing to stream)
	}
	if err := r.enter(next); err != nil {
		r.fail(err)
	}
}

// enter makes a layer ready, points its tensors at the buffer and starts the
// read of the layers after it.
func (r *Ring) enter(layer int) error {
	s, err := r.ready(layer)
	if err != nil {
		return err
	}
	for _, p := range r.layers[layer] {
		r.opt.Retarget(p.tensor, uintptr(unsafe.Pointer(&s.base[p.dataOff])))
	}
	at := r.pos[layer]
	for d := 1; d <= r.opt.Ahead && d < len(r.order); d++ {
		r.start(r.order[(at+d)%len(r.order)], false)
	}
	return nil
}

// fail stops the ring and gives every tensor back to the file mapping, which
// has the same bytes: the model goes on, slower.
func (r *Ring) fail(err error) {
	if r.failed.Swap(true) {
		return
	}
	r.errMu.Lock()
	r.err = err
	r.errMu.Unlock()
	for i := range r.tensors {
		if r.layers[r.tensors[i].Layer] != nil {
			r.opt.Retarget(i, 0)
		}
	}
}

// RingStats are the ring's counters.
type RingStats struct {
	Reads, Bytes int64
	// Stalls counts the times the computing thread had to wait for the disk,
	// StallTime the total wait.
	Stalls    int64
	StallTime time.Duration
	// Skipped counts read-aheads left out because the buffer was still in use.
	Skipped int64
	Slots   int
	Err     error
}

// Stats reports what the ring has done.
func (r *Ring) Stats() RingStats {
	r.errMu.Lock()
	defer r.errMu.Unlock()
	return RingStats{Reads: r.reads.Load(), Bytes: r.bytes.Load(), Stalls: r.stalls.Load(), StallTime: time.Duration(r.stallNS.Load()),
		Skipped: r.skipped.Load(), Slots: len(r.slots), Err: r.err}
}

// Data returns the bytes of tensor i of a layer that is ready (tests).
func (r *Ring) Data(layer, tensor int) ([]byte, error) {
	s, err := r.ready(layer)
	if err != nil {
		return nil, err
	}
	for _, p := range r.layers[layer] {
		if p.tensor == tensor {
			return s.base[p.dataOff : p.dataOff+r.tensors[tensor].Length], nil
		}
	}
	return nil, fmt.Errorf("memlock: tensor %d is not in layer %d", tensor, layer)
}

// Close stops the readers, gives the tensors back to the file mapping and
// frees the buffers. Nothing may be computing when it is called.
func (r *Ring) Close() {
	select {
	case <-r.stop:
		return
	default:
		close(r.stop)
	}
	r.wg.Wait()
	if !r.failed.Load() {
		for i := range r.tensors {
			if r.layers[r.tensors[i].Layer] != nil {
				r.opt.Retarget(i, 0)
			}
		}
	}
	r.free()
}
