package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"golang.org/x/term"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/backend/llamacpp"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/posture"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/tune"
)

// optimizeCommand is the autotuner (docs/OPTIMIZE.md).
//
//	nanollm optimize [--mode silent|optimal|powerful] [--yes] [--dry-run] [--json] [--model ID] [-config FILE]
//	nanollm optimize --status [--json]     (what the installers ask: the mode, the profile, is it stale)
//	nanollm optimize calibrate             (a development tool: a device measured the long way)
//	nanollm optimize worker                (internal: one probe, the spec on stdin)
func optimizeCommand(args []string) error {
	if len(args) > 0 && args[0] == "worker" {
		return optimizeWorker()
	}
	if len(args) > 0 && args[0] == "calibrate" {
		return optimizeCalibrate(args[1:])
	}
	fs := flag.NewFlagSet("optimize", flag.ExitOnError)
	cfgPath := configFlag(fs)
	modeFlag := fs.String("mode", "", "silent, optimal or powerful")
	yes := fs.Bool("yes", false, "apply the result without asking")
	dry := fs.Bool("dry-run", false, "look the machine over and list what each setup would cost; measure nothing")
	asJSON := fs.Bool("json", false, "print the result as JSON")
	status := fs.Bool("status", false, "print the mode and whether the profile is current, and exit")
	model := fs.String("model", "", "the model to tune (default models.default)")
	libDirFlag := fs.String("lib-dir", "", "native library directory (overrides backends.lib_dir)")
	local := fs.Bool("local", false, "measure in this process even when the server is running (it holds the GPU too: expect worse numbers)")
	apiKey := fs.String("api-key", "", "admin key for the running server (default: NANOLLM_API_KEY, or the first admin key in the configuration)")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if len(pos) > 0 && *model == "" {
		*model = pos[0]
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	interactive := term.IsTerminal(int(os.Stdin.Fd())) && term.IsTerminal(int(os.Stderr.Fd())) && !*yes && !*asJSON

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// A running server measures inside itself: the same account, the same
	// devices, and no second process on the GPU.
	var svc *optClient
	if !*local {
		if key, kerr := adminKey(cfg, *apiKey); kerr == nil {
			svc, _ = detectService(cfg, key)
		} else if _, up := detectService(cfg, ""); up {
			return kerr
		}
	}

	if *status {
		if svc != nil {
			return optimizeStatusViaService(ctx, svc, *asJSON)
		}
		return optimizeStatus(ctx, cfg, *model, *libDirFlag, *asJSON)
	}

	mode := tune.Mode(cfg.Tune.Mode)
	if *modeFlag != "" {
		if mode, err = tune.ParseMode(*modeFlag); err != nil {
			return err
		}
	}
	askForMode := *modeFlag == "" && interactive && !*dry

	// The local form opens the backends now: the menu shows this machine's
	// numbers, and the run reuses the survey it needs for them.
	var opts tune.Options
	var survey *tune.Survey
	if svc == nil {
		var closeAll func()
		if opts, closeAll, err = optimizeOptions(cfg, *model, *libDirFlag, mode); err != nil {
			return err
		}
		defer closeAll()
		if askForMode {
			survey = tune.TakeSurvey(ctx, opts.Survey)
		}
	}
	if askForMode {
		if mode, err = chooseMode(ctx, cfg, mode, opts, survey, svc); err != nil {
			return err
		}
	}
	if !mode.Valid() {
		if !*dry {
			return errors.New("optimize: choose a mode with --mode silent|optimal|powerful")
		}
		mode = tune.Optimal
	}
	if svc != nil {
		return optimizeViaService(ctx, svc, mode, *model, *dry, *asJSON, *yes, interactive)
	}
	opts.Mode = mode

	if *dry {
		var rep *tune.Report
		if survey != nil {
			rep = tune.Analyze(survey, opts)
		} else if rep, err = tune.Look(ctx, opts); err != nil {
			return err
		}
		if *asJSON {
			return writeJSONTo(os.Stdout, rep)
		}
		rep.WriteText(os.Stdout)
		return nil
	}

	progress := func(e tune.Event) {
		if *asJSON {
			return
		}
		switch e.Type {
		case "probe_start", "probe", "decision":
			fmt.Fprintf(os.Stderr, "[%3.0f s] %s\n", e.ElapsedS, e.Message)
		}
	}
	// The probes run in child processes, which load the model as the server
	// would. A running server holds its own copy; the form that goes through it
	// (above) pauses it first.
	res, err := tune.Run(ctx, tune.RunOptions{Options: opts, Runner: tune.SpawnRunner{}, Progress: progress, Survey: survey})
	if err != nil {
		if res != nil && res.Report != nil && !*asJSON {
			res.Report.WriteText(os.Stdout)
		}
		return err
	}
	if *asJSON {
		if !*yes {
			return writeJSONTo(os.Stdout, res.Profile)
		}
	} else {
		res.WriteResult(os.Stdout)
	}
	apply := *yes
	if !apply && interactive {
		apply = askKey(os.Stdin, os.Stderr, "\nApply? [Y/n] ", true)
	}
	if !apply {
		if !*asJSON {
			fmt.Fprintln(os.Stderr, "\nNothing was applied. Run again with --yes to apply.")
		}
		return nil
	}
	ar, err := tune.Apply(res.Profile, tune.ApplyOptions{Store: tune.Store{Dir: cfg.TunePath()}, ConfigPath: existingFile(*cfgPath)})
	if err != nil {
		return err
	}
	if *asJSON {
		return writeJSONTo(os.Stdout, map[string]any{"profile": res.Profile, "apply": ar})
	}
	printApply(os.Stderr, ar, res.Profile)
	return nil
}

// chooseMode asks which mode to measure. A mode that is set but whose profile
// is stale or missing is offered first, with what changed
// (docs/OPTIMIZE.md, §4.1); then the menu of the three.
func chooseMode(ctx context.Context, cfg config.Config, current tune.Mode, opts tune.Options, survey *tune.Survey, svc *optClient) (tune.Mode, error) {
	if current.Valid() {
		if msg := staleMessage(ctx, cfg, opts, svc); msg != "" {
			fmt.Fprintf(os.Stderr, "%s\n", strings.Join(wrap("The profile for "+string(current)+" needs a recheck: "+msg, 78), "\n"))
			if askKey(os.Stdin, os.Stderr, fmt.Sprintf("Measure again in the %s mode? [Y/n] ", current), true) {
				return current, nil
			}
		}
	}
	lines := map[tune.Mode]string{}
	if survey != nil {
		lines = tune.ModeLines(survey, opts)
	}
	items := make([]menuItem, len(tune.Modes))
	def := 1
	for i, m := range tune.Modes {
		detail := lines[m]
		if detail == "" {
			detail = m.Summary()
		}
		items[i] = menuItem{Label: string(m), Detail: detail}
		if m == current {
			def = i
		}
	}
	i, err := chooseFromMenu(os.Stdin, os.Stderr, "Choose a mode (arrows, Enter; q leaves):", items, def)
	if err != nil {
		// Not a terminal that can be put in raw mode: the plain question.
		return askMode(os.Stdin, os.Stderr, current)
	}
	if i < 0 {
		return "", errors.New("optimize: cancelled")
	}
	return tune.Modes[i], nil
}

// staleMessage is why the profile of the current mode is not in force, empty
// when it is.
func staleMessage(ctx context.Context, cfg config.Config, opts tune.Options, svc *optClient) string {
	if svc != nil {
		var st struct {
			Message string `json:"message"`
		}
		if svc.call(ctx, http.MethodGet, "/api/v1/optimize", nil, &st) == nil {
			return st.Message
		}
		return ""
	}
	var devs []backend.Device
	for _, b := range opts.Survey.Backends {
		if ds, err := b.Devices(ctx); err == nil {
			devs = append(devs, ds...)
		}
	}
	act, err := tune.LoadActive(ctx, tune.ActiveInput{
		Config: cfg, Manifest: opts.Survey.Manifest, Devices: devs, LibDir: opts.Survey.LibDir,
		LibsRelease: llamacpp.LibsRelease, LlamaCommit: llamacpp.PinnedCommit,
	})
	if err != nil || act == nil {
		return ""
	}
	return act.Message()
}

// askMode is the menu without a raw terminal: a numbered list and a line.
func askMode(in io.Reader, out io.Writer, current tune.Mode) (tune.Mode, error) {
	fmt.Fprintln(out, "Choose a mode:")
	for i, m := range tune.Modes {
		fmt.Fprintf(out, "  %d. %-9s %s\n", i+1, m, m.Summary())
	}
	def := 2
	if i := indexOfMode(current); i >= 0 {
		def = i + 1
	}
	fmt.Fprintf(out, "Mode [%d]: ", def)
	line, _ := bufio.NewReader(in).ReadString('\n')
	line = strings.TrimSpace(line)
	if line == "" {
		return tune.Modes[def-1], nil
	}
	for i, m := range tune.Modes {
		if line == fmt.Sprint(i+1) || strings.EqualFold(line, string(m)) {
			return m, nil
		}
	}
	return "", fmt.Errorf("optimize: %q is not a mode", line)
}

func indexOfMode(m tune.Mode) int {
	for i, x := range tune.Modes {
		if x == m {
			return i
		}
	}
	return -1
}

func existingFile(p string) string {
	if p == "" {
		return ""
	}
	if _, err := os.Stat(p); err != nil {
		return ""
	}
	return p
}

func printApply(w io.Writer, ar tune.ApplyResult, p *tune.Profile) {
	fmt.Fprintf(w, "\nProfile %s saved.\n", p.ID())
	switch {
	case ar.ConfigWritten:
		fmt.Fprintf(w, "tune.mode: %s is set in %s.\n", p.Mode, ar.ConfigPath)
	case ar.Note != "":
		fmt.Fprintln(w, ar.Note)
	default:
		fmt.Fprintf(w, "Set tune.mode: %s in the configuration to use it.\n", p.Mode)
	}
	for _, r := range ar.RestartRequired {
		fmt.Fprintf(w, "Restart the server to apply %s.\n", r)
	}
	if ar.ConfigWritten {
		fmt.Fprintln(w, "A running server takes the profile on when it is restarted, or through POST /api/v1/optimize/{id}/apply.")
	}
}

// optimizeStatus prints what an installer asks: the mode, whether a profile is
// in force, and what changed if it is stale.
func optimizeStatus(ctx context.Context, cfg config.Config, modelID, libDirFlag string, asJSON bool) error {
	type status struct {
		Mode    string        `json:"mode"`
		Profile string        `json:"profile,omitempty"`
		Stale   bool          `json:"stale"`
		Missing bool          `json:"missing"`
		Changes []tune.Change `json:"changes,omitempty"`
		Suggest string        `json:"suggest,omitempty"`
	}
	st := status{Mode: cfg.Tune.Mode}
	if cfg.Tune.Mode != "" {
		opts, closeAll, err := optimizeOptions(cfg, modelID, libDirFlag, tune.Mode(cfg.Tune.Mode))
		if err != nil {
			return err
		}
		defer closeAll()
		var devs []backend.Device
		for _, b := range opts.Survey.Backends {
			if ds, err := b.Devices(ctx); err == nil {
				devs = append(devs, ds...)
			}
		}
		act, err := tune.LoadActive(ctx, tune.ActiveInput{
			Config: cfg, Manifest: opts.Survey.Manifest, Devices: devs, LibDir: opts.Survey.LibDir,
			LibsRelease: llamacpp.LibsRelease, LlamaCommit: llamacpp.PinnedCommit,
		})
		if err != nil {
			return err
		}
		if act != nil {
			st.Stale, st.Missing, st.Changes = act.Stale, act.Missing, act.Changes
			if act.Profile != nil {
				st.Profile = act.Profile.ID()
			}
			if s := act.Staleness(); s.Suggest != nil {
				st.Suggest = string(s.Suggest.Mode)
			}
		}
	}
	if asJSON {
		return writeJSONTo(os.Stdout, st)
	}
	switch {
	case st.Mode == "":
		fmt.Println("The autotuner is off (tune.mode is empty). Run: nanollm optimize")
	case st.Profile != "":
		fmt.Printf("mode %s, profile %s is current\n", st.Mode, st.Profile)
	case st.Stale:
		var ch []string
		for _, c := range st.Changes {
			ch = append(ch, c.String())
		}
		fmt.Printf("mode %s, the profile is stale (%s). Recheck: nanollm optimize --mode %s\n", st.Mode, strings.Join(ch, "; "), st.Mode)
	default:
		fmt.Printf("mode %s, but this machine has no profile. Measure: nanollm optimize --mode %s\n", st.Mode, st.Mode)
	}
	return nil
}

// optimizeStatusViaService asks the running server what it holds.
func optimizeStatusViaService(ctx context.Context, c *optClient, asJSON bool) error {
	var st struct {
		Mode    string `json:"mode"`
		Profile *struct {
			ID string `json:"id"`
		} `json:"profile"`
		Stale   bool          `json:"stale"`
		Missing bool          `json:"missing"`
		Changes []tune.Change `json:"changes"`
		Suggest *tune.Suggest `json:"suggest"`
		Message string        `json:"message"`
	}
	if err := c.call(ctx, http.MethodGet, "/api/v1/optimize", nil, &st); err != nil {
		return err
	}
	out := map[string]any{"mode": st.Mode, "stale": st.Stale, "missing": st.Missing}
	if st.Profile != nil {
		out["profile"] = st.Profile.ID
	}
	if len(st.Changes) > 0 {
		out["changes"] = st.Changes
	}
	if st.Suggest != nil {
		out["suggest"] = st.Suggest.Mode
	}
	if asJSON {
		return writeJSONTo(os.Stdout, out)
	}
	switch {
	case st.Mode == "":
		fmt.Println("The autotuner is off (tune.mode is empty). Run: nanollm optimize")
	case st.Profile != nil:
		fmt.Printf("mode %s, profile %s is current\n", st.Mode, st.Profile.ID)
	default:
		fmt.Printf("mode %s: %s\n", st.Mode, st.Message)
	}
	return nil
}

func writeJSONTo(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

// optimizeWorker is `nanollm optimize worker`: one probe, the spec on stdin,
// the result on stdout. The process has been placed (NANOLLM_PLACE) before it
// gets here.
func optimizeWorker() error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	return tune.WorkerMain(ctx, os.Stdin, os.Stdout, func(spec tune.ProbeSpec) (backend.Backend, error) {
		if spec.LowPriority {
			if err := posture.Apply(true); err != nil {
				fmt.Fprintln(os.Stderr, "optimize worker: cannot lower the priority:", err)
			}
		}
		if spec.Backend == fake.Name {
			// An instant fake measures millions of tokens a second; a laptop-like
			// pace keeps the chain testable.
			return fake.New(fake.Options{PrefillRate: 4000, DecodeRate: 200, MaxDraft: 3, Acceptance: 0.6}), nil
		}
		return backend.Open(spec.Backend, spec.LibDir)
	})
}

// optimizeCalibrate measures one device the long way and prints a baseline for
// internal/tune/matrix.yaml (a development tool).
//
//	nanollm optimize calibrate [--device SEL] [--threads N] [--low] [-config FILE]
func optimizeCalibrate(args []string) error {
	fs := flag.NewFlagSet("optimize calibrate", flag.ExitOnError)
	cfgPath := configFlag(fs)
	device := fs.String("device", "", "the device to measure, e.g. llamacpp:cuda0 or llamacpp:cpu (default: the first)")
	threads := fs.Int("threads", 0, "CPU decode threads (default: every physical core)")
	low := fs.Bool("low", false, "measure under the quiet mode's priority: below normal, EcoQoS on Windows")
	asJSON := fs.Bool("json", false, "print the whole result as JSON")
	libDirFlag := fs.String("lib-dir", "", "native library directory (overrides backends.lib_dir)")
	if _, err := parseFlags(fs, args); err != nil {
		return err
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	opts, closeAll, err := optimizeOptions(cfg, "", *libDirFlag, tune.Powerful)
	if err != nil {
		return err
	}
	defer closeAll()
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	rep, err := tune.Calibrate(ctx, opts, *device, *threads, *low, tune.SpawnRunner{})
	if err != nil {
		return err
	}
	if *asJSON {
		return writeJSONTo(os.Stdout, rep)
	}
	rep.WriteText(os.Stdout)
	return nil
}

// optimizeOptions builds what the autotuner looks at from the configuration:
// the backends, the model files and the model's own overrides. The returned
// function closes the backends.
func optimizeOptions(cfg config.Config, modelID, libDirFlag string, mode tune.Mode) (tune.Options, func(), error) {
	cat, err := registry.Builtin()
	if err != nil {
		return tune.Options{}, nil, err
	}
	if modelID == "" {
		modelID = cfg.Models.Default
	}
	m, ok := registry.Find(cat, modelID)
	if !ok {
		return tune.Options{}, nil, fmt.Errorf("optimize: no %q in the catalog", modelID)
	}
	fakeOnly := len(cfg.Backends.Enabled) == 1 && cfg.Backends.Enabled[0] == config.BackendFake
	modelPath, drafterPath := "fake.gguf", "fake-drafter.gguf"
	if !fakeOnly {
		if modelPath, drafterPath, err = newStore(cfg).Ready(m); err != nil {
			return tune.Options{}, nil, err
		}
	}
	if ov := cfg.Models.Overrides[m.ID]; ov.Speculative.Type == config.SpecNone {
		drafterPath = ""
	}
	libDir, err := resolveLibDir(cfg, libDirFlag)
	if err != nil {
		return tune.Options{}, nil, err
	}
	var backends []backend.Backend
	for _, name := range cfg.Backends.Enabled {
		b, err := backend.Open(name, libDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "optimize: backend %s unavailable: %v\n", name, err)
			continue
		}
		backends = append(backends, b)
	}
	closeAll := func() {
		for _, b := range backends {
			_ = b.Close()
		}
	}
	if len(backends) == 0 {
		return tune.Options{}, nil, errors.New("optimize: no backend could be opened; see `nanollm devices`")
	}
	return tune.Options{
		Mode: mode,
		Survey: tune.SurveyOptions{
			Backends: backends, LibDir: libDir, Manifest: m, ModelPath: modelPath, DrafterPath: drafterPath,
			Version: version, LibsRelease: llamacpp.LibsRelease, LlamaTag: llamacpp.PinnedTag, LlamaCommit: llamacpp.PinnedCommit,
		},
		Plan: tune.PlanInput{
			Manifest: m, Override: cfg.Models.Overrides[m.ID], ModelPath: modelPath, DrafterPath: drafterPath,
			Threads: cfg.Devices.CPUThreads, ThreadsBatch: cfg.Devices.CPUThreadsBatch,
		},
	}, closeAll, nil
}
