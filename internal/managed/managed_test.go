package managed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/tune"
)

func TestSettingsRoundTripAndPatch(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	if _, configured, err := LoadSettings(path); err != nil || configured {
		t.Fatalf("no file: configured=%v err=%v, want an unconfigured instance", configured, err)
	}

	s := Settings{DefaultModel: "a", Mode: "optimal", Thinking: true}
	if err := SaveSettings(path, s); err != nil {
		t.Fatal(err)
	}
	got, configured, err := LoadSettings(path)
	if err != nil || !configured || got.DefaultModel != "a" || got.Mode != "optimal" || !got.Thinking {
		t.Fatalf("loaded %+v configured=%v err=%v", got, configured, err)
	}

	// A patch changes what it names and keeps the rest.
	next, err := got.Patch([]byte(`{"mode":"powerful","thinking":false}`))
	if err != nil {
		t.Fatal(err)
	}
	if next.Mode != "powerful" || next.Thinking || next.DefaultModel != "a" {
		t.Fatalf("patched %+v", next)
	}

	for _, body := range []string{`{"mdoe":"silent"}`, `{"mode":"silent"} {}`, `not json`} {
		if _, err := got.Patch([]byte(body)); core.AsError(err).Code != core.CodeInvalidRequest {
			t.Errorf("%s = %v, want invalid_request", body, err)
		}
	}
}

func TestSettingsDefaults(t *testing.T) {
	var s Settings
	if s.Model() != config.DefaultModel || s.TuneMode() != tune.Silent || !s.AutotuneOn() {
		t.Fatalf("an unset instance: %s %s %v", s.Model(), s.TuneMode(), s.AutotuneOn())
	}
	if (Settings{Autotune: AutotuneOff}).AutotuneOn() {
		t.Fatal("autotune off measures")
	}
}

func catalog(id string) (registry.Manifest, error) {
	if id == config.DefaultModel || id == "other" {
		return registry.Manifest{ID: id}, nil
	}
	return registry.Manifest{}, core.Errorf(core.CodeModelNotFound, "model %q is not in the catalog", id)
}

func TestValidateNamesTheField(t *testing.T) {
	for _, tc := range []struct {
		s     Settings
		param string
	}{
		{Settings{DefaultModel: "nope"}, "default_model"},
		{Settings{Mode: "turbo"}, "mode"},
		{Settings{Autotune: "sometimes"}, "autotune"},
		{Settings{Mirrors: []string{"http://plain.example"}}, "mirrors"},
		{Settings{LogLevel: "loud"}, "log_level"},
	} {
		err := tc.s.Validate(catalog)
		if e := core.AsError(err); e == nil || e.Param != tc.param {
			t.Errorf("%+v: %v, want a refusal of %s", tc.s, err, tc.param)
		}
	}
	ok := Settings{DefaultModel: "other", Mode: "powerful", Autotune: AutotuneOff, Mirrors: []string{"https://hf-mirror.example"}, LogLevel: "debug"}
	if err := ok.Validate(catalog); err != nil {
		t.Fatalf("valid settings refused: %v", err)
	}
}

// The plan is the configuration of a process in the background of an
// application: the loopback, one key, the quiet mode until told otherwise, no
// NUMA policy and nothing sent anywhere.
func TestPlan(t *testing.T) {
	paths := PathsFor(t.TempDir())
	cfg, err := NewPlan(Settings{}, Options{Paths: paths, Token: "sk-test-0123456789abcdef"})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.Addr != "127.0.0.1:0" || cfg.Server.DataDir != paths.Data || cfg.Auth.Mode != "apikey" || len(cfg.Auth.Keys) != 1 {
		t.Errorf("listener or auth: %q %q %q %d keys", cfg.Server.Addr, cfg.Server.DataDir, cfg.Auth.Mode, len(cfg.Auth.Keys))
	}
	if k := cfg.Auth.Keys[0]; k.Secret != "sk-test-0123456789abcdef" || !k.Admin || !k.Interactive {
		t.Errorf("the key: %+v", k)
	}
	if cfg.Tune.Mode != config.TuneSilent || cfg.Models.Default != config.DefaultModel || len(cfg.Models.Preload) != 0 {
		t.Errorf("mode %q, model %q, preload %v", cfg.Tune.Mode, cfg.Models.Default, cfg.Models.Preload)
	}
	if cfg.Devices.NUMA != "off" || cfg.Bench.Submit != "never" || cfg.UI.Enabled {
		t.Errorf("numa %q, submit %q, ui %v", cfg.Devices.NUMA, cfg.Bench.Submit, cfg.UI.Enabled)
	}
	if cfg.Log.File != paths.Log || cfg.TunePath() != paths.Tune || cfg.ModelsPath() != paths.Models {
		t.Errorf("paths: log %q tune %q models %q", cfg.Log.File, cfg.TunePath(), cfg.ModelsPath())
	}

	cfg, err = NewPlan(Settings{DefaultModel: "other", Mode: "optimal", Thinking: true, LogLevel: "debug"},
		Options{Paths: paths, Token: "sk-test-0123456789abcdef", UI: true, Backend: config.BackendFake})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Tune.Mode != config.TuneOptimal || cfg.Models.Default != "other" || !cfg.Generation.EnableThinking ||
		cfg.Log.Level != "debug" || !cfg.UI.Enabled || !slices.Equal(cfg.Backends.Enabled, []string{config.BackendFake}) {
		t.Errorf("the settings did not reach the plan: %+v", cfg)
	}
}

func TestEventsAreOneJSONObjectPerLine(t *testing.T) {
	var out bytes.Buffer
	e := NewEvents(&out)
	e.Emit(EventReady, map[string]any{"port": 1234, "token": "t"})
	e.Emit(EventStopping, nil)
	e.Warn(WarnTuneFailed, "no", map[string]any{"mode": "silent"})
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 3 || !strings.HasPrefix(lines[0], `{"event":"ready",`) || lines[1] != `{"event":"stopping"}` {
		t.Fatalf("lines: %q", lines)
	}
	var v map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &v); err != nil || v["port"] != float64(1234) {
		t.Fatalf("%q: %v", lines[0], err)
	}
	if err := json.Unmarshal([]byte(lines[2]), &v); err != nil || v["event"] != EventWarning || v["code"] != WarnTuneFailed || v["mode"] != "silent" {
		t.Fatalf("%q: %v", lines[2], err)
	}
}

func TestLockIsExclusive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nanollm.lock")
	first, err := Lock(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Lock(path); !errors.Is(err, ErrLocked) {
		t.Fatalf("a second lock = %v, want ErrLocked", err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	again, err := Lock(path)
	if err != nil {
		t.Fatalf("the lock was not given up: %v", err)
	}
	_ = again.Close()
}

func TestWatchParentNoticesTheExit(t *testing.T) {
	// The test binary itself, running only the helper below, is a process
	// that waits until it is killed on every platform.
	cmd := exec.Command(testBinary(t), "-test.run=^TestHelperWaits$", "-test.count=1")
	cmd.Env = append(os.Environ(), "NANOLLM_TEST_HELPER=1")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stopped := make(chan string, 1)
	if err := WatchParent(ctx, cmd.Process.Pid, func(reason string) { stopped <- reason }); err != nil {
		t.Fatal(err)
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
	select {
	case reason := <-stopped:
		if reason != "parent exited" {
			t.Fatalf("reason %q", reason)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the exit was not noticed")
	}
}

func TestWatchStdinStopsAtTheEnd(t *testing.T) {
	stopped := make(chan string, 1)
	WatchStdin(strings.NewReader("ignored"), func(reason string) { stopped <- reason })
	select {
	case reason := <-stopped:
		if reason != "stdin closed" {
			t.Fatalf("reason %q", reason)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the end of stdin was not noticed")
	}
}

// A failure stands while nothing changes: the same binary, the same key.
func TestFailuresMatchUntilSomethingChanges(t *testing.T) {
	f := Failures{Dir: t.TempDir(), Version: "v1"}
	key := tune.Key{Mode: tune.Silent, Model: "m", Device: "cpu", Driver: "1", LibsRelease: "b1"}
	if _, ok := f.Matches("m", tune.Silent, key); ok {
		t.Fatal("a failure before any was recorded")
	}
	if err := f.Record("m", tune.Silent, key, errors.New("the worker crashed")); err != nil {
		t.Fatal(err)
	}
	rec, ok := f.Matches("m", tune.Silent, key)
	if !ok || rec.Error != "the worker crashed" || rec.At.IsZero() {
		t.Fatalf("%+v %v", rec, ok)
	}
	changed := key
	changed.Driver = "2"
	if _, ok := f.Matches("m", tune.Silent, changed); ok {
		t.Error("a new driver still matches the failure")
	}
	if _, ok := f.Matches("m", tune.Optimal, key); ok {
		t.Error("another mode matches the failure")
	}
	if _, ok := (Failures{Dir: f.Dir, Version: "v2"}).Matches("m", tune.Silent, key); ok {
		t.Error("a new binary still matches the failure")
	}
	f.Clear("m", tune.Silent)
	if _, ok := f.Matches("m", tune.Silent, key); ok {
		t.Error("a cleared failure matches")
	}
}

// The first start measures the mode it has no profile for, applies it, and
// loads the model after it; a look again measures nothing.
func TestAutotuneMeasuresWhatIsMissing(t *testing.T) {
	r := newTuneRig(t, t.TempDir(), nil)
	if st := r.a.State(); st.State != TuneMissing || st.Mode != tune.Silent || st.Model != config.DefaultModel {
		t.Fatalf("before: %+v", st)
	}
	r.a.Kick("")
	started := r.out.wait(t, named(EventTuneStarted))
	if started["reason"] != ReasonMissing || started["mode"] != "silent" || started["model"] != config.DefaultModel {
		t.Fatalf("tune_started: %v", started)
	}
	done := r.out.wait(t, named(EventTuneDone))
	if done["applied"] != true || done["profile"] == "" {
		t.Fatalf("tune_done: %v", done)
	}
	st := r.waitState(t, TuneReady)
	if st.Profile == nil || !strings.HasPrefix(st.Profile.ID, "silent@") {
		t.Fatalf("ready without a profile: %+v", st)
	}
	deadline := time.Now().Add(5 * time.Second)
	for len(r.loads()) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if !slices.Equal(r.loads(), []string{config.DefaultModel}) {
		t.Errorf("loads %v", r.loads())
	}

	// A change of mode measures that mode, for the reason given.
	r.mu.Lock()
	r.settings.Mode = "optimal"
	r.mu.Unlock()
	r.a.Kick(ReasonSettings)
	r.out.wait(t, func(e map[string]any) bool {
		return e["event"] == EventTuneStarted && e["reason"] == ReasonSettings && e["mode"] == "optimal"
	})
	r.waitState(t, TuneReady)
	// Back to silent: its profile is current, nothing is measured.
	r.mu.Lock()
	r.settings.Mode = ""
	r.mu.Unlock()
	r.a.Kick(ReasonSettings)
	r.waitState(t, TuneReady)
	time.Sleep(50 * time.Millisecond)
	if got := r.tuner.starts(); !slices.Equal(got, []string{ReasonMissing, ReasonSettings}) {
		t.Errorf("measured for %v", got)
	}
}

// A measurement that fails is said once, recorded, and not repeated by the
// next start on the same machine; the application can still ask for one.
func TestAutotuneDoesNotRepeatAFailure(t *testing.T) {
	dir := t.TempDir()
	r := newTuneRig(t, dir, errors.New("the worker could not load the model"))
	r.a.Kick("")
	fail := r.out.wait(t, warning(WarnTuneFailed))
	if !strings.Contains(fail["message"].(string), "could not load") {
		t.Fatalf("tune_failed: %v", fail)
	}
	st := r.waitState(t, TuneFailed)
	if !strings.Contains(st.Message, "could not load") {
		t.Fatalf("%+v", st)
	}
	// The limits hold and the model loads without a profile.
	deadline := time.Now().Add(5 * time.Second)
	for len(r.loads()) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if len(r.loads()) == 0 {
		t.Error("the model was not loaded after the failure")
	}
	r.cancel()

	// The next start.
	next := newTuneRig(t, dir, errors.New("the worker could not load the model"))
	if st := next.a.State(); st.State != TuneFailed {
		t.Fatalf("the next start: %+v", st)
	}
	next.a.Kick("")
	next.out.wait(t, warning(WarnTuneSkipped))
	time.Sleep(50 * time.Millisecond)
	if got := next.tuner.starts(); len(got) != 0 {
		t.Fatalf("measured again for %v", got)
	}
	// A new driver is a new machine.
	next.tuner.mu.Lock()
	next.tuner.key.Driver = "2"
	next.tuner.fail = nil
	next.tuner.mu.Unlock()
	next.a.Kick(ReasonSettings)
	next.out.wait(t, named(EventTuneDone))
	next.waitState(t, TuneReady)
	if _, ok := (Failures{Dir: dir, Version: "test"}).Matches(config.DefaultModel, tune.Silent, tune.Key{Mode: tune.Silent, Model: config.DefaultModel, Device: "cpu", Driver: "2"}); ok {
		t.Error("the failure stayed after a measurement that succeeded")
	}
}

// The application asks for a measurement whatever the last one did.
func TestAutotuneRequestMeasuresAnyway(t *testing.T) {
	dir := t.TempDir()
	key := tune.Key{Mode: tune.Silent, Model: config.DefaultModel, Device: "cpu", Driver: "1"}
	if err := (Failures{Dir: dir, Version: "test"}).Record(config.DefaultModel, tune.Silent, key, errors.New("earlier")); err != nil {
		t.Fatal(err)
	}
	r := newTuneRig(t, dir, nil)
	run, err := r.a.Request(context.Background())
	if err != nil || run.Reason != ReasonRequested {
		t.Fatalf("%+v %v", run, err)
	}
	r.out.wait(t, named(EventTuneDone))
	r.waitState(t, TuneReady)
}

// No model on disk: nothing is measured until it arrives, and then for that
// reason.
func TestAutotuneWaitsForTheModel(t *testing.T) {
	r := newTuneRig(t, t.TempDir(), nil)
	r.mu.Lock()
	r.present = false
	r.mu.Unlock()
	r.a.Kick("")
	if st := r.waitState(t, TuneWaitingModel); st.Message == "" {
		t.Errorf("%+v", st)
	}
	if _, err := r.a.Request(context.Background()); core.AsError(err).Code != core.CodeConflict {
		t.Errorf("a request without the model: %v", err)
	}
	time.Sleep(50 * time.Millisecond)
	if got := r.tuner.starts(); len(got) != 0 {
		t.Fatalf("measured without the model: %v", got)
	}
	r.mu.Lock()
	r.present = true
	r.mu.Unlock()
	r.a.Kick(ReasonDownload)
	started := r.out.wait(t, named(EventTuneStarted))
	if started["reason"] != ReasonDownload {
		t.Fatalf("%v", started)
	}
}

// Settings changed while a measurement runs are taken on when it ends.
func TestAutotuneTakesASettingChangedMeanwhile(t *testing.T) {
	r := newTuneRig(t, t.TempDir(), nil)
	r.tuner.gate = make(chan struct{})
	r.a.Kick("")
	r.out.wait(t, named(EventTuneStarted))
	if st := r.a.State(); st.State != TuneRunning || st.Run == "" {
		t.Fatalf("%+v", st)
	}
	r.mu.Lock()
	r.settings.Mode = "powerful"
	r.mu.Unlock()
	r.a.Kick(ReasonSettings)
	time.Sleep(50 * time.Millisecond)
	if got := r.tuner.starts(); len(got) != 1 {
		t.Fatalf("a second measurement while one runs: %v", got)
	}
	close(r.tuner.gate)
	r.out.wait(t, func(e map[string]any) bool {
		return e["event"] == EventTuneStarted && e["mode"] == "powerful" && e["reason"] == ReasonSettings
	})
	st := r.waitState(t, TuneReady)
	if st.Mode != tune.Powerful {
		t.Fatalf("%+v", st)
	}
}

// A measurement the application ran itself through POST /api/v1/optimize
// waits for its apply request: the loop does not measure that pair on its
// own meanwhile.
func TestAutotuneLeavesTheApplicationsOwnRunToIt(t *testing.T) {
	r := newTuneRig(t, t.TempDir(), nil)
	r.mu.Lock()
	r.settings.Autotune = AutotuneOff
	r.mu.Unlock()
	r.a.Kick("")
	r.waitState(t, TuneOff)
	// The look is over once it has loaded the model.
	deadline := time.Now().Add(5 * time.Second)
	for len(r.loads()) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}

	run := RunInfo{ID: "manual", Mode: tune.Silent, Model: config.DefaultModel, Reason: "api", EstimateS: 1}
	r.a.Started(run)
	if st := r.a.State(); st.State != TuneRunning || st.Run != "manual" {
		t.Fatalf("%+v", st)
	}
	p := &tune.Profile{Mode: tune.Silent, Key: tune.Key{Mode: tune.Silent, Model: config.DefaultModel}}
	r.mu.Lock()
	r.settings.Autotune = "" // measuring is back on, and the pair still has no profile
	r.mu.Unlock()
	r.a.Done(run, p, nil)
	done := r.out.wait(t, named(EventTuneDone))
	if done["applied"] != false {
		t.Fatalf("tune_done: %v", done)
	}
	r.waitState(t, TuneMissing)
	time.Sleep(50 * time.Millisecond)
	if got := r.tuner.starts(); len(got) != 0 {
		t.Fatalf("the loop measured the application's pair itself: %v", got)
	}

	// The application applies it: the profile is in force.
	r.tuner.mu.Lock()
	r.tuner.profiles[pair(config.DefaultModel, tune.Silent)] = p
	r.tuner.cur = &tune.Active{Mode: tune.Silent, Profile: p}
	r.tuner.mu.Unlock()
	if !r.a.Applied(run, p) {
		t.Fatal("an apply request of the application's own run is not reported as one")
	}
	r.waitState(t, TuneReady)
}

func TestControllerSavesAndKicks(t *testing.T) {
	paths := PathsFor(t.TempDir())
	if err := paths.Create(); err != nil {
		t.Fatal(err)
	}
	r := newTuneRig(t, paths.Tune, nil)
	var applied []Settings
	c := NewController(ControllerOptions{
		Version: "test", Paths: paths, Lookup: catalog, Autotune: r.a,
		Present: func(string) bool { return true },
		Loaded:  func() bool { return false },
		Apply:   func(_, next Settings) { applied = append(applied, next) },
	})
	r.mu.Lock()
	r.settings = c.Settings()
	r.mu.Unlock()
	if st := c.Status(); st.Configured || st.Features.Chat.State != ChatLoading || len(st.MissingModels) != 0 || st.Settings.Mirrors == nil {
		t.Fatalf("first start: %+v", st)
	}
	if _, err := c.Update([]byte(`{"default_model":"nope"}`)); err == nil {
		t.Fatal("an unknown model was accepted")
	}
	st, err := c.Update([]byte(`{"thinking":true}`))
	if err != nil || !st.Configured || !st.Settings.Thinking {
		t.Fatalf("%+v %v", st, err)
	}
	if saved, configured, _ := LoadSettings(paths.Settings); !configured || !saved.Thinking {
		t.Fatalf("not saved: %+v", saved)
	}
	if len(applied) != 1 || !applied[0].Thinking {
		t.Fatalf("not applied: %+v", applied)
	}
	if err := c.AdoptMode(tune.Optimal); err != nil {
		t.Fatal(err)
	}
	if saved, _, _ := LoadSettings(paths.Settings); saved.Mode != "optimal" || !saved.Thinking {
		t.Fatalf("the adopted mode was not saved: %+v", saved)
	}
}
