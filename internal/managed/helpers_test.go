package managed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/tune"
)

func testBinary(t *testing.T) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Skip("no path to the test binary:", err)
	}
	return exe
}

// TestHelperWaits is the child process of TestWatchParentNoticesTheExit.
func TestHelperWaits(t *testing.T) {
	if os.Getenv("NANOLLM_TEST_HELPER") != "1" {
		t.Skip("a helper process, not a test")
	}
	time.Sleep(time.Minute)
}

// lines is an events stream the tests read while it is written.
type lines struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *lines) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

// events are the events written so far.
func (l *lines) events(t *testing.T) []map[string]any {
	t.Helper()
	l.mu.Lock()
	defer l.mu.Unlock()
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(l.buf.String()), "\n") {
		if line == "" {
			continue
		}
		var v map[string]any
		if err := json.Unmarshal([]byte(line), &v); err != nil {
			t.Fatalf("%q: %v", line, err)
		}
		out = append(out, v)
	}
	return out
}

// wait returns the first event that matches, waiting for it.
func (l *lines) wait(t *testing.T, match func(map[string]any) bool) map[string]any {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		for _, e := range l.events(t) {
			if match(e) {
				return e
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no such event in:\n%s", l.buf.String())
	return nil
}

func named(name string) func(map[string]any) bool {
	return func(e map[string]any) bool { return e["event"] == name }
}

func warning(code string) func(map[string]any) bool {
	return func(e map[string]any) bool { return e["event"] == EventWarning && e["code"] == code }
}

// fakeTuner stands for the server's autotuner: a measurement succeeds or
// fails at once, and a success is a profile of the mode and model.
type fakeTuner struct {
	a    *Autotune
	fail error
	// gate, when not nil, holds a measurement until it is closed.
	gate chan struct{}

	mu       sync.Mutex
	profiles map[string]*tune.Profile
	cur      *tune.Active
	reasons  []string
	key      tune.Key
}

func newFakeTuner() *fakeTuner {
	return &fakeTuner{profiles: map[string]*tune.Profile{}, key: tune.Key{Device: "cpu", Driver: "1"}}
}

func (f *fakeTuner) Current() *tune.Active {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.cur
}

func (f *fakeTuner) Look(_ context.Context, mode tune.Mode, model string) (*tune.Active, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	p := f.profiles[pair(model, mode)]
	return &tune.Active{Mode: mode, Profile: p, Missing: p == nil}, nil
}

func (f *fakeTuner) Activate(ctx context.Context, mode tune.Mode, model string) (*tune.Active, error) {
	act, err := f.Look(ctx, mode, model)
	f.mu.Lock()
	f.cur = act
	f.mu.Unlock()
	return act, err
}

func (f *fakeTuner) MachineKey(_ context.Context, mode tune.Mode, model string) (tune.Key, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	k := f.key
	k.Mode, k.Model = mode, model
	return k, nil
}

func (f *fakeTuner) Start(mode tune.Mode, model, reason string) (RunInfo, error) {
	f.mu.Lock()
	if f.cur != nil && f.cur.Mode == "measuring" {
		f.mu.Unlock()
		return RunInfo{}, errors.New("an optimization is already running")
	}
	f.reasons = append(f.reasons, reason)
	f.mu.Unlock()
	run := RunInfo{ID: "run" + reason, Mode: mode, Model: model, Reason: reason, EstimateS: 1, Apply: true}
	f.a.Started(run)
	go func() {
		if f.gate != nil {
			<-f.gate
		}
		if f.fail != nil {
			f.a.Done(run, nil, f.fail)
			return
		}
		p := &tune.Profile{Mode: mode, Key: tune.Key{Mode: mode, Model: model}}
		f.mu.Lock()
		f.profiles[pair(model, mode)] = p
		f.cur = &tune.Active{Mode: mode, Profile: p}
		f.mu.Unlock()
		f.a.Applied(run, p)
		f.a.Done(run, p, nil)
	}()
	return run, nil
}

func (f *fakeTuner) starts() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.reasons...)
}

// tuneRig is an Autotune on a fake tuner, with the settings in a variable.
type tuneRig struct {
	a        *Autotune
	tuner    *fakeTuner
	out      *lines
	mu       sync.Mutex
	settings Settings
	present  bool
	loaded   []string
	cancel   context.CancelFunc
}

func newTuneRig(t *testing.T, dir string, fail error) *tuneRig {
	t.Helper()
	r := &tuneRig{tuner: newFakeTuner(), out: &lines{}, present: true}
	r.tuner.fail = fail
	r.a = NewAutotune(AutotuneOptions{
		Tuner:    r.tuner,
		Events:   NewEvents(r.out),
		Failures: Failures{Dir: dir, Version: "test"},
		Settings: func() Settings {
			r.mu.Lock()
			defer r.mu.Unlock()
			return r.settings
		},
		Present: func(string) bool {
			r.mu.Lock()
			defer r.mu.Unlock()
			return r.present
		},
		Posture: func(tune.Mode) error { return nil },
		Load: func(_ context.Context, model string) error {
			r.mu.Lock()
			defer r.mu.Unlock()
			r.loaded = append(r.loaded, model)
			return nil
		},
	})
	r.tuner.a = r.a
	ctx, cancel := context.WithCancel(context.Background())
	r.cancel = cancel
	t.Cleanup(cancel)
	r.a.Prepare(ctx)
	go r.a.Run(ctx)
	return r
}

// waitState waits for the autotuner to report a state.
func (r *tuneRig) waitState(t *testing.T, want string) TuneState {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	var st TuneState
	for time.Now().Before(deadline) {
		if st = r.a.State(); st.State == want {
			return st
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("state %+v, want %s", st, want)
	return st
}

func (r *tuneRig) loads() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.loaded...)
}
