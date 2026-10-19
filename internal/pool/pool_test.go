package pool

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/sched"
)

type counter struct {
	loads  atomic.Int32
	closed atomic.Int32
	delay  time.Duration
	fail   bool
}

type closeCounting struct {
	backend.Model
	c *counter
}

func (m closeCounting) Close() error { m.c.closed.Add(1); return m.Model.Close() }

func (c *counter) load(ctx context.Context, k Key) (backend.Model, any, error) {
	c.loads.Add(1)
	time.Sleep(c.delay)
	if c.fail {
		return nil, nil, errors.New("load failed")
	}
	m, err := fake.New(fake.Options{}).Load(ctx, backend.LoadSpec{NCtx: k.NCtx})
	return closeCounting{m, c}, "plan for " + k.Model, err
}

func key(model string) Key {
	return Key{Model: model, Revision: "r1", Backend: "fake", Device: "cpu", NCtx: 4096}
}

func TestSharedLoadAndRefcount(t *testing.T) {
	c := &counter{delay: 50 * time.Millisecond}
	p := New(c.load, 2)
	defer p.Close()
	var wg sync.WaitGroup
	for range 5 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			inst, release, err := p.Acquire(context.Background(), key("a"))
			if err != nil || inst.Info != "plan for a" {
				t.Error(err)
				return
			}
			release()
			release() // idempotent
		}()
	}
	wg.Wait()
	if c.loads.Load() != 1 {
		t.Fatalf("%d loads for one key", c.loads.Load())
	}
	if st := p.Status(); len(st) != 1 || st[0].Refs != 0 {
		t.Fatalf("status %+v", st)
	}
}

func TestEvictionLRUAndPin(t *testing.T) {
	c := &counter{}
	p := New(c.load, 2)
	defer p.Close()
	use := func(m string) {
		_, release, err := p.Acquire(context.Background(), key(m))
		if err != nil {
			t.Fatal(m, err)
		}
		release()
	}
	use("a")
	use("b")
	p.Pin(key("a"), true)
	use("c") // evicts b, the unpinned one
	names := map[string]bool{}
	for _, s := range p.Status() {
		names[s.Key.Model] = true
	}
	if !names["a"] || names["b"] || !names["c"] {
		t.Fatalf("resident %v", names)
	}
	// Both resident instances busy or pinned: no room.
	_, releaseC, _ := p.Acquire(context.Background(), key("c"))
	if _, _, err := p.Acquire(context.Background(), key("d")); !errors.Is(err, ErrBusy) {
		t.Fatalf("expected ErrBusy, got %v", err)
	}
	releaseC()
	if c.closed.Load() != 1 {
		t.Fatalf("closed %d", c.closed.Load())
	}
}

func TestRetireDrainsBusyInstance(t *testing.T) {
	c := &counter{}
	p := New(c.load, 3)
	defer p.Close()
	old := key("a")
	inst, release, _ := p.Acquire(context.Background(), old)
	h, err := inst.Loop.Submit(context.Background(), sched0(inst))
	if err != nil {
		t.Fatal(err)
	}
	next := old
	next.Revision = "r2"
	_, releaseNew, err := p.Acquire(context.Background(), next)
	if err != nil {
		t.Fatal(err)
	}
	p.Retire("a", next)
	if c.closed.Load() != 0 {
		t.Fatal("busy instance unloaded at retire")
	}
	// The running request finishes on the old instance.
	for {
		ev, _ := h.Next(context.Background())
		if ev.Done {
			if ev.Err != nil {
				t.Fatal(ev.Err)
			}
			break
		}
	}
	release()
	if c.closed.Load() != 1 {
		t.Fatal("retired instance not unloaded after its last release")
	}
	releaseNew()
	if st := p.Status(); len(st) != 1 || st[0].Key.Revision != "r2" {
		t.Fatalf("status %+v", st)
	}
}

func TestLoadFailureIsReported(t *testing.T) {
	p := New((&counter{fail: true}).load, 1)
	if _, _, err := p.Acquire(context.Background(), key("a")); err == nil {
		t.Fatal("expected the load error")
	}
	if len(p.Status()) != 0 {
		t.Fatal("failed load left an instance")
	}
}

// An instance the pool loads starts pinned when the Pinned hook says so, and
// is then safe from eviction like one pinned afterwards.
func TestPinnedHookPinsNewInstances(t *testing.T) {
	c := &counter{}
	p := New(c.load, 1)
	defer p.Close()
	p.Pinned = func(k Key) bool { return k.Model == "a" }
	_, release, err := p.Acquire(context.Background(), key("a"))
	if err != nil {
		t.Fatal(err)
	}
	release()
	if st := p.Status(); len(st) != 1 || !st[0].Pinned {
		t.Fatalf("status %+v", st)
	}
	if _, _, err := p.Acquire(context.Background(), key("b")); !errors.Is(err, ErrBusy) {
		t.Fatalf("a pinned instance was evicted: %v", err)
	}
	p.Pin(key("a"), false)
	inst, release, err := p.Acquire(context.Background(), key("b"))
	if err != nil || inst.pinned {
		t.Fatalf("b after unpinning a: %v", err)
	}
	release()
}

// SetupLoop sees each freshly loaded instance's loop once, before any request
// can reach it, and what it attaches belongs to that instance.
func TestSetupLoopRunsOncePerLoad(t *testing.T) {
	c := &counter{}
	p := New(c.load, 1)
	defer p.Close()
	var seen []Key
	var mu sync.Mutex
	p.SetupLoop = func(k Key, l *sched.Loop) {
		if l == nil || l.Busy() != 0 {
			t.Errorf("setup got loop %v", l)
		}
		mu.Lock()
		seen = append(seen, k)
		mu.Unlock()
	}
	for range 3 {
		_, release, err := p.Acquire(context.Background(), key("a"))
		if err != nil {
			t.Fatal(err)
		}
		release()
	}
	_, release, err := p.Acquire(context.Background(), key("b"))
	if err != nil {
		t.Fatal(err)
	}
	release()
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 2 || seen[0].Model != "a" || seen[1].Model != "b" {
		t.Fatalf("setup ran for %v", seen)
	}
}
