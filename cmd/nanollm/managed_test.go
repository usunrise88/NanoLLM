package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/posture"
	"github.com/usunrise88/nanollm/internal/tune"
)

// testRunner measures on a scripted backend in this process. A probe waits
// while the gate is closed, and fails while fail is set.
type testRunner struct {
	inner tune.Runner
	fail  atomic.Bool

	mu   sync.Mutex
	gate chan struct{}
}

func newTestRunner() *testRunner {
	fb := fake.New(fake.Options{
		PrefillRate: 40000, DecodeRate: 500, MaxDraft: 3, Acceptance: 0.6,
		BatchCost: func(tokens, outputs int) time.Duration {
			return time.Duration(10_000+400*(tokens-1)) * time.Microsecond
		},
	})
	return &testRunner{inner: tune.InProcessRunner{Backend: fb}}
}

func (r *testRunner) hold() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.gate = make(chan struct{})
}

func (r *testRunner) release() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.gate != nil {
		close(r.gate)
		r.gate = nil
	}
}

func (r *testRunner) Probe(ctx context.Context, spec tune.ProbeSpec) (tune.ProbeResult, error) {
	r.mu.Lock()
	gate := r.gate
	r.mu.Unlock()
	if gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
			return tune.ProbeResult{}, ctx.Err()
		}
	}
	if r.fail.Load() {
		return tune.ProbeResult{}, errors.New("the worker could not load the model")
	}
	return r.inner.Probe(ctx, spec)
}

// levels records the priorities managed asks for, in place of setting the
// test process's own.
type levels struct {
	mu  sync.Mutex
	all []posture.Level
}

func (l *levels) last() posture.Level {
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.all) == 0 {
		return -1
	}
	return l.all[len(l.all)-1]
}

// useTestRunner makes managed measure with r and record its priority.
func useTestRunner(t *testing.T, r *testRunner) *levels {
	t.Helper()
	lv := &levels{}
	runner, pos := managedRunner, managedPosture
	managedRunner = r
	managedPosture = func(l posture.Level) error {
		lv.mu.Lock()
		defer lv.mu.Unlock()
		lv.all = append(lv.all, l)
		return nil
	}
	t.Cleanup(func() { managedRunner, managedPosture = runner, pos })
	return lv
}

// instance is one `nanollm managed` run in this process, with its standard
// streams as pipes, the way an application starts it.
type instance struct {
	dir    string
	stdin  *io.PipeWriter
	events chan map[string]any
	done   chan error
	ready  map[string]any
}

func startManaged(t *testing.T, dir string, args ...string) *instance {
	t.Helper()
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	inst := &instance{dir: dir, stdin: inW, events: make(chan map[string]any, 64), done: make(chan error, 1)}
	args = append([]string{"-data-dir", dir, "-test-backend", "fake"}, args...)
	go func() {
		err := managedCommand(context.Background(), args, inR, outW)
		_ = outW.Close()
		inst.done <- err
	}()
	go func() {
		defer close(inst.events)
		sc := bufio.NewScanner(outR)
		for sc.Scan() {
			var ev map[string]any
			if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
				t.Errorf("not a JSON line on stdout: %q", sc.Text())
				continue
			}
			inst.events <- ev
		}
	}()
	t.Cleanup(func() { _ = inW.Close() })
	return inst
}

func (inst *instance) next(t *testing.T) map[string]any {
	t.Helper()
	select {
	case ev, ok := <-inst.events:
		if !ok {
			t.Fatal("stdout closed")
		}
		return ev
	case <-time.After(60 * time.Second):
		t.Fatal("no event within 60s")
	}
	return nil
}

// expect reads the next event and checks its name, and its code or reason
// when given.
func (inst *instance) expect(t *testing.T, name string, fields ...string) map[string]any {
	t.Helper()
	ev := inst.next(t)
	if ev["event"] != name {
		t.Fatalf("event %v, want %s", ev, name)
	}
	for i := 0; i+1 < len(fields); i += 2 {
		if fmt.Sprint(ev[fields[i]]) != fields[i+1] {
			t.Fatalf("event %v, want %s=%s", ev, fields[i], fields[i+1])
		}
	}
	return ev
}

func (inst *instance) waitReady(t *testing.T) {
	t.Helper()
	inst.ready = inst.expect(t, "ready")
}

func (inst *instance) request(t *testing.T, method, path, body string) (int, http.Header, map[string]any) {
	t.Helper()
	req, err := http.NewRequest(method, inst.ready["url"].(string)+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+inst.ready["token"].(string))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, resp.Header, out
}

func (inst *instance) status(t *testing.T) map[string]any {
	t.Helper()
	code, _, st := inst.request(t, http.MethodGet, "/api/v1/managed", "")
	if code != http.StatusOK {
		t.Fatalf("status: %d %v", code, st)
	}
	return st
}

// autotune is the autotuner's part of the status.
func autotune(st map[string]any) map[string]any {
	return st["features"].(map[string]any)["autotune"].(map[string]any)
}

// waitAutotune polls the status until the autotuner reports want.
func (inst *instance) waitAutotune(t *testing.T, want string) map[string]any {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	var st map[string]any
	for time.Now().Before(deadline) {
		st = inst.status(t)
		if autotune(st)["state"] == want {
			return st
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("autotune %v, want %s", autotune(st), want)
	return nil
}

const chatBody = `{"messages":[{"role":"user","content":"Say one word."}],"max_tokens":4}`

func (inst *instance) chat(t *testing.T) (int, http.Header) {
	t.Helper()
	code, h, _ := inst.request(t, http.MethodPost, "/v1/chat/completions", chatBody)
	return code, h
}

func (inst *instance) stop(t *testing.T) {
	t.Helper()
	_ = inst.stdin.Close()
	inst.expect(t, "stopping", "reason", "stdin closed")
	select {
	case err := <-inst.done:
		if err != nil {
			t.Fatalf("stopped with %v; the log ends:\n%s", err, inst.logTail())
		}
	case <-time.After(30 * time.Second):
		t.Fatalf("did not stop within 30s of stdin closing; the log ends:\n%s", inst.logTail())
	}
}

// logTail is the end of the instance's log, for a failure to show.
func (inst *instance) logTail() string {
	b, _ := os.ReadFile(filepath.Join(inst.dir, "logs", "nanollm.log"))
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	return strings.Join(lines[max(0, len(lines)-25):], "\n")
}

// The first start of an instance is its setup, and measures the quiet mode;
// every later one comes up with what was chosen then and measures nothing.
func TestManagedSetupIsKeptForTheNextStart(t *testing.T) {
	useTestRunner(t, newTestRunner())
	dir := t.TempDir()

	first := startManaged(t, dir)
	first.waitReady(t)
	if first.ready["configured"] != false || first.ready["api_version"] != float64(1) {
		t.Fatalf("first start: %v", first.ready)
	}
	if tn := first.ready["tune"].(map[string]any); tn["mode"] != "silent" || tn["state"] != "missing" {
		t.Fatalf("first start: tune %v", tn)
	}
	first.expect(t, "tune_started", "reason", "missing", "mode", "silent", "model", "gemma-4-e4b-qat")
	done := first.expect(t, "tune_done", "mode", "silent", "applied", "true")
	if !strings.HasPrefix(fmt.Sprint(done["profile"]), "silent@") {
		t.Fatalf("tune_done: %v", done)
	}

	// The token is the only way in, and only by the loopback name.
	url := first.ready["url"].(string)
	resp, err := http.Get(url + "/api/v1/managed")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("without the token: %d", resp.StatusCode)
	}
	req, _ := http.NewRequest(http.MethodGet, url+"/healthz", nil)
	req.Host = fmt.Sprintf("rebound.example:%v", first.ready["port"])
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusMisdirectedRequest {
		t.Fatalf("a foreign Host: %d", resp.StatusCode)
	}

	code, _, st := first.request(t, http.MethodPatch, "/api/v1/managed/settings", `{"thinking":true,"log_level":"debug"}`)
	if code != http.StatusOK || st["configured"] != true || st["restart_required"] != false {
		t.Fatalf("settings: %d %v", code, st)
	}
	if code, _, _ := first.request(t, http.MethodPatch, "/api/v1/managed/settings", `{"thinkng":true}`); code != http.StatusBadRequest {
		t.Errorf("a misspelt setting: %d", code)
	}
	if code, _, _ := first.request(t, http.MethodPatch, "/api/v1/managed/settings", `{"mode":"turbo"}`); code != http.StatusBadRequest {
		t.Errorf("a mode that is not one: %d", code)
	}

	// One instance per data directory.
	second := startManaged(t, dir)
	second.expect(t, "fatal", "code", "data_dir_locked")
	if err := <-second.done; err == nil {
		t.Fatal("the second instance did not fail")
	}

	first.stop(t)

	again := startManaged(t, dir)
	again.waitReady(t)
	if again.ready["configured"] != true || again.ready["token"] == first.ready["token"] {
		t.Fatalf("second start: %v", again.ready)
	}
	if tn := again.ready["tune"].(map[string]any); tn["state"] != "ready" {
		t.Fatalf("second start: tune %v, want the profile of the first", tn)
	}
	settings := again.status(t)["settings"].(map[string]any)
	if settings["thinking"] != true || settings["log_level"] != "debug" {
		t.Fatalf("the settings did not survive the restart: %v", settings)
	}
	// Nothing to measure: the next event is the stop.
	again.stop(t)
}

// The measurement at the start holds chat requests back with a Retry-After,
// applies itself, and a change of mode measures that mode.
func TestManagedTunesAtStartAndOnAChangeOfMode(t *testing.T) {
	runner := newTestRunner()
	lv := useTestRunner(t, runner)
	runner.hold()
	inst := startManaged(t, t.TempDir())
	inst.waitReady(t)
	started := inst.expect(t, "tune_started", "reason", "missing", "mode", "silent")
	if lv.last() != posture.Quiet {
		t.Errorf("the quiet mode runs at %v", lv.last())
	}

	code, h := inst.chat(t)
	if code != http.StatusServiceUnavailable || h.Get("Retry-After") == "" {
		t.Fatalf("a chat request during the measurement: %d, Retry-After %q", code, h.Get("Retry-After"))
	}
	st := inst.status(t)
	if at := autotune(st); at["state"] != "running" || at["run"] != started["run"] {
		t.Fatalf("while it measures: %v", at)
	}

	runner.release()
	inst.expect(t, "tune_done", "run", fmt.Sprint(started["run"]), "applied", "true")
	if code, _ := inst.chat(t); code != http.StatusOK {
		t.Fatalf("a chat request after the measurement: %d", code)
	}
	st = inst.waitAutotune(t, "ready")
	if chat := st["features"].(map[string]any)["chat"].(map[string]any); chat["state"] != "ready" {
		t.Errorf("chat: %v", chat)
	}

	code, _, st = inst.request(t, http.MethodPatch, "/api/v1/managed/settings", `{"mode":"optimal"}`)
	if code != http.StatusOK || st["settings"].(map[string]any)["mode"] != "optimal" {
		t.Fatalf("PATCH mode: %d %v", code, st)
	}
	inst.expect(t, "tune_started", "reason", "settings", "mode", "optimal")
	inst.expect(t, "tune_done", "mode", "optimal")
	st = inst.waitAutotune(t, "ready")
	if at := autotune(st); at["mode"] != "optimal" || !strings.HasPrefix(fmt.Sprint(at["profile"].(map[string]any)["id"]), "optimal@") {
		t.Fatalf("after the change: %v", at)
	}
	if lv.last() != posture.Below {
		t.Errorf("an application's background process runs below normal in the optimal mode, not at %v", lv.last())
	}
	if code, _ := inst.chat(t); code != http.StatusOK {
		t.Fatalf("a chat request in the new mode: %d", code)
	}
	inst.stop(t)
}

// A measurement that fails is said, recorded, and not repeated by the next
// start on the same machine; the model runs by the mode's limits meanwhile,
// and the application can still ask for a measurement.
func TestManagedDoesNotRepeatAFailedMeasurement(t *testing.T) {
	runner := newTestRunner()
	runner.fail.Store(true)
	useTestRunner(t, runner)
	dir := t.TempDir()

	first := startManaged(t, dir)
	first.waitReady(t)
	first.expect(t, "tune_started", "reason", "missing")
	first.expect(t, "warning", "code", "tune_failed")
	st := first.waitAutotune(t, "failed")
	if autotune(st)["message"] == "" {
		t.Fatalf("%v", autotune(st))
	}
	if code, _ := first.chat(t); code != http.StatusOK {
		t.Fatalf("a chat request after the failure: %d", code)
	}
	if _, err := os.Stat(filepath.Join(dir, "tune", "gemma-4-e4b-qat.silent.failed.json")); err != nil {
		t.Fatalf("the failure was not recorded: %v", err)
	}
	first.stop(t)

	again := startManaged(t, dir)
	again.waitReady(t)
	if tn := again.ready["tune"].(map[string]any); tn["state"] != "failed" {
		t.Fatalf("the next start: tune %v", tn)
	}
	again.expect(t, "warning", "code", "tune_skipped")

	runner.fail.Store(false)
	code, _, out := again.request(t, http.MethodPost, "/api/v1/managed/tune", "")
	if code != http.StatusAccepted || out["run"] == nil {
		t.Fatalf("POST tune: %d %v", code, out)
	}
	again.expect(t, "tune_started", "reason", "requested", "run", fmt.Sprint(out["run"]))
	again.expect(t, "tune_done", "applied", "true")
	again.waitAutotune(t, "ready")
	if _, err := os.Stat(filepath.Join(dir, "tune", "gemma-4-e4b-qat.silent.failed.json")); !os.IsNotExist(err) {
		t.Errorf("the failure stayed after a measurement that succeeded: %v", err)
	}
	again.stop(t)
}

// Settings that cannot be read do not keep the instance from starting: they
// are set aside and the instance comes up unconfigured.
func TestManagedResetsUnreadableSettings(t *testing.T) {
	useTestRunner(t, newTestRunner())
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "settings.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	inst := startManaged(t, dir, "-watch-stdin=true")
	inst.expect(t, "warning", "code", "settings_reset")
	inst.waitReady(t)
	if inst.ready["configured"] != false {
		t.Fatalf("%v", inst.ready)
	}
	if _, err := os.Stat(filepath.Join(dir, "settings.json.invalid")); err != nil {
		t.Errorf("the unreadable settings were not kept: %v", err)
	}
	inst.expect(t, "tune_started")
	inst.expect(t, "tune_done")
	inst.stop(t)
}

// A start that fails says why, once, and nothing after: no stop was asked for.
func TestManagedFailureIsOneFatalEvent(t *testing.T) {
	useTestRunner(t, newTestRunner())
	// A parent that has already exited.
	cmd := exec.Command(os.Args[0], "-test.run=^$")
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
	inst := startManaged(t, t.TempDir(), "-parent-pid", strconv.Itoa(cmd.Process.Pid))
	inst.expect(t, "fatal", "code", "parent_gone")
	if err := <-inst.done; err == nil {
		t.Fatal("the start did not fail")
	}
	for ev := range inst.events {
		t.Errorf("an event after the fatal one: %v", ev)
	}

	bad := startManaged(t, t.TempDir(), "-no-such-flag")
	bad.expect(t, "fatal", "code", "invalid_arguments")
	if err := <-bad.done; err == nil {
		t.Fatal("a bad flag did not fail")
	}
}
