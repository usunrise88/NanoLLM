package pool

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/sched"
)

// Key identifies one loaded instance. Two requests share an instance only if
// every field matches: a different preset or context size is a different
// placement, and a different revision is different weights.
type Key struct {
	Model    string `json:"model"`
	Revision string `json:"revision"`
	Backend  string `json:"backend"`
	Device   string `json:"device"`
	Mode     string `json:"mode,omitempty"`
	Preset   string `json:"preset,omitempty"`
	NCtx     int    `json:"n_ctx"`
}

func (k Key) String() string {
	return fmt.Sprintf("%s@%s %s:%s %s/%s ctx=%d", k.Model, short(k.Revision), k.Backend, k.Device, k.Preset, k.Mode, k.NCtx)
}

func short(rev string) string {
	if len(rev) > 8 {
		return rev[:8]
	}
	return rev
}

// LoadFunc loads the model a key names. Info is whatever the caller wants
// reported with the instance — the plan it was loaded under.
type LoadFunc func(ctx context.Context, k Key) (backend.Model, any, error)

// Instance is one loaded model and its scheduling loop.
type Instance struct {
	Key   Key
	Model backend.Model
	Loop  *sched.Loop
	Info  any

	loadedAt time.Time
	lastUse  time.Time
	refs     int
	pinned   bool
	retired  bool
}

// Status is an instance as /api/v1/models reports it.
type Status struct {
	Key      Key       `json:"key"`
	Refs     int       `json:"refs"`
	Busy     int       `json:"busy"`
	Pinned   bool      `json:"pinned"`
	Retired  bool      `json:"retired"`
	LoadedAt time.Time `json:"loaded_at"`
	LastUse  time.Time `json:"last_use"`
	Info     any       `json:"info,omitempty"`
	// Loop is the scheduling loop's state: slots, busy slots, queue.
	Loop sched.LoopStats `json:"loop"`
	// Memory is what the load log reported for the model's buffers.
	Memory backend.MemoryReport `json:"memory"`
}

// Pool keeps instances resident. A request acquires an instance and releases
// it when done; an instance nobody holds stays loaded until it is evicted for
// another (least recently used first, pinned ones never) or retired.
type Pool struct {
	load LoadFunc
	max  int
	// MaxPending bounds the requests that may wait for a slot of one model;
	// set before the first Acquire. Zero is no bound.
	MaxPending int
	// Pinned says whether an instance that has just loaded starts pinned: a
	// pin asked for before its model was resident, or one the operator keeps
	// across an unload. It runs under the pool's lock, so it must not call
	// back into the pool; set before the first Acquire. Nil pins nothing.
	Pinned func(Key) bool

	mu        sync.Mutex
	instances map[Key]*Instance
	loading   map[Key]*loadCall
	closed    bool
}

type loadCall struct {
	done chan struct{}
	inst *Instance
	err  error
}

// New returns a pool holding at most max instances at once (at least one).
func New(load LoadFunc, max int) *Pool {
	if max < 1 {
		max = 1
	}
	return &Pool{load: load, max: max, instances: map[Key]*Instance{}, loading: map[Key]*loadCall{}}
}

// ErrBusy is returned when the pool is full of instances in use.
var ErrBusy = errors.New("pool: every resident model is in use; raise pool.max_resident_models or retry")

// ErrClosed is returned after Close.
var ErrClosed = errors.New("pool: shutting down")

// Acquire returns the instance for k, loading it if needed. Concurrent
// acquisitions of one key share the load. The release function must be called
// exactly once.
func (p *Pool) Acquire(ctx context.Context, k Key) (*Instance, func(), error) {
	for {
		p.mu.Lock()
		if p.closed {
			p.mu.Unlock()
			return nil, nil, ErrClosed
		}
		if inst, ok := p.instances[k]; ok && !inst.retired {
			inst.refs++
			inst.lastUse = time.Now()
			p.mu.Unlock()
			return inst, p.releaser(inst), nil
		}
		if inst, ok := p.instances[k]; ok && inst.retired {
			// Reloading a retired key: the draining instance keeps running
			// its requests outside the map and unloads when they finish.
			delete(p.instances, k)
		}
		if call, ok := p.loading[k]; ok {
			p.mu.Unlock()
			select {
			case <-call.done:
			case <-ctx.Done():
				return nil, nil, ctx.Err()
			}
			if call.err != nil {
				return nil, nil, call.err
			}
			continue // take a reference under the lock
		}
		victims, err := p.makeRoomLocked()
		if err != nil {
			p.mu.Unlock()
			return nil, nil, err
		}
		call := &loadCall{done: make(chan struct{})}
		p.loading[k] = call
		p.mu.Unlock()

		for _, v := range victims {
			p.unload(v)
		}
		// The load outlives the request that started it: others may be
		// waiting on it, and an abandoned half-load wastes the work.
		m, info, err := p.load(context.WithoutCancel(ctx), k)
		p.mu.Lock()
		delete(p.loading, k)
		if err == nil {
			now := time.Now()
			loop := sched.New(m)
			loop.SetMaxPending(p.MaxPending)
			inst := &Instance{Key: k, Model: m, Loop: loop, Info: info, loadedAt: now, lastUse: now}
			// Asked under the lock that inserts the instance, so a Pin that
			// races the load either sees the instance or is seen here.
			inst.pinned = p.Pinned != nil && p.Pinned(k)
			if p.closed {
				err = ErrClosed
				p.mu.Unlock()
				p.unload(inst)
				p.mu.Lock()
			} else {
				p.instances[k] = inst
				call.inst = inst
			}
		}
		call.err = err
		close(call.done)
		p.mu.Unlock()
		if err != nil {
			return nil, nil, err
		}
	}
}

// makeRoomLocked picks instances to unload so one more fits: retired ones
// first, then idle unpinned ones by least recent use.
func (p *Pool) makeRoomLocked() ([]*Instance, error) {
	n := len(p.instances) + len(p.loading)
	if n < p.max {
		return nil, nil
	}
	var idle []*Instance
	for _, inst := range p.instances {
		if inst.refs == 0 && !inst.pinned {
			idle = append(idle, inst)
		}
	}
	sort.Slice(idle, func(i, j int) bool {
		if idle[i].retired != idle[j].retired {
			return idle[i].retired
		}
		return idle[i].lastUse.Before(idle[j].lastUse)
	})
	need := n - p.max + 1
	if len(idle) < need {
		return nil, ErrBusy
	}
	victims := idle[:need]
	for _, v := range victims {
		delete(p.instances, v.Key)
	}
	return victims, nil
}

func (p *Pool) releaser(inst *Instance) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			p.mu.Lock()
			inst.refs--
			inst.lastUse = time.Now()
			drop := inst.retired && inst.refs == 0
			// A retired key may have been loaded afresh since; only this
			// instance's own entry goes.
			if drop && p.instances[inst.Key] == inst {
				delete(p.instances, inst.Key)
			}
			p.mu.Unlock()
			if drop {
				p.unload(inst)
			}
		})
	}
}

func (p *Pool) unload(inst *Instance) {
	inst.Loop.Close()
	_ = inst.Model.Close()
}

// Pin keeps an instance resident through eviction; unpinning makes it
// evictable again. It reports whether the key is loaded.
func (p *Pool) Pin(k Key, pin bool) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	inst, ok := p.instances[k]
	if ok {
		inst.pinned = pin
	}
	return ok
}

// Retire marks every instance of model other than keep for unloading: idle
// ones go now, busy ones when their last request finishes. New requests go to
// keep. This is the hot swap: a new revision or placement takes over while
// requests on the old one complete.
func (p *Pool) Retire(model string, keep Key) {
	p.mu.Lock()
	var drop []*Instance
	for k, inst := range p.instances {
		if k.Model != model || k == keep {
			continue
		}
		inst.retired = true
		if inst.refs == 0 {
			delete(p.instances, k)
			drop = append(drop, inst)
		}
	}
	p.mu.Unlock()
	for _, inst := range drop {
		p.unload(inst)
	}
}

// Unload removes an instance now if it is idle. A busy instance is retired
// and goes when its requests finish.
func (p *Pool) Unload(k Key) error {
	p.mu.Lock()
	inst, ok := p.instances[k]
	if !ok {
		p.mu.Unlock()
		return fmt.Errorf("%s is not loaded", k)
	}
	inst.retired = true
	if inst.refs > 0 {
		p.mu.Unlock()
		return nil
	}
	delete(p.instances, k)
	p.mu.Unlock()
	p.unload(inst)
	return nil
}

// Status lists resident instances, sorted by key.
func (p *Pool) Status() []Status {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]Status, 0, len(p.instances))
	for _, inst := range p.instances {
		out = append(out, Status{
			Key: inst.Key, Refs: inst.refs, Busy: inst.Loop.Busy(), Pinned: inst.pinned,
			Retired: inst.retired, LoadedAt: inst.loadedAt, LastUse: inst.lastUse, Info: inst.Info,
			Loop: inst.Loop.Snapshot(), Memory: inst.Model.MemoryReport(),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key.String() < out[j].Key.String() })
	return out
}

// Close unloads everything. Requests still running end with an error.
func (p *Pool) Close() {
	p.mu.Lock()
	p.closed = true
	all := make([]*Instance, 0, len(p.instances))
	for k, inst := range p.instances {
		all = append(all, inst)
		delete(p.instances, k)
	}
	p.mu.Unlock()
	for _, inst := range all {
		p.unload(inst)
	}
}
