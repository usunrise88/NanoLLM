package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/llamacpp"
	"github.com/usunrise88/nanollm/internal/bench"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/registry"
)

// benchCommand is the preset benchmark and the public benchmark client
// (docs/BENCHMARKS.md).
//
//	nanollm bench [MODEL] [--suite standard|dev] [--quick|--full] [--preset P]... [--submit] [--out FILE]
//	nanollm bench --suite dev --preset P --set threads=8,16 [--profile weak] [--numa interleave] [--cold] [--label L]
//	nanollm bench compare A.json [B.json]
//	nanollm bench submit FILE
//	nanollm bench withdraw ID
//	nanollm bench worker            (internal: one preset, spec on stdin)
func benchCommand(args []string) error {
	if len(args) > 0 {
		switch args[0] {
		case "worker":
			ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
			defer stop()
			return bench.Worker(ctx, os.Stdin, os.Stdout)
		case "compare":
			return benchCompare(args[1:])
		case "submit":
			return benchSubmit(args[1:])
		case "withdraw":
			return benchWithdraw(args[1:])
		}
	}
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	cfgPath := configFlag(fs)
	suite := fs.String("suite", bench.SuiteStandard, "standard (public) or dev (local only)")
	full := fs.Bool("full", false, "every applicable preset instead of auto and two representative ones")
	fs.Bool("quick", true, "auto and two representative presets (the default)")
	var presets multiFlag
	fs.Var(&presets, "preset", "run this preset (repeatable) instead of the suite's choice")
	reps := fs.Int("reps", 0, "repetitions per scenario (default 3, or 5 for the dev suite)")
	submit := fs.Bool("submit", false, "send the result without asking (for scripts)")
	out := fs.String("out", "", "result file (default bench/results/<machine>/<date>-<suite>[-<label>].json)")
	var sets multiFlag
	fs.Var(&sets, "set", "dev suite: sweep key=a,b (repeatable; the product of all axes runs)\n"+bench.VariantKeys())
	profile := fs.String("profile", "", "dev suite: strong, weak or <cpus>:<preset> (e.g. 0-3:ram-2g): where and with what budget to run")
	numaFlag := fs.String("numa", "", "dev suite: off, interleave, distribute or node:N")
	cold := fs.Bool("cold", false, "evict the model file from the page cache before each run (a strict budget does it anyway)")
	advisory := fs.Bool("advisory", false, "dev suite: plan for the budget but do not enforce it (no cgroup, no cold start): fast sweeps of compute settings")
	timeout := fs.Duration("timeout", 25*time.Minute, "one run's time limit (a worker that has not finished is recorded as failed, too slow); 0 is none")
	slow := fs.Bool("slow", false, "dev suite: pp and 12 decode tokens only, for budgets that stream the weights from a hard disk")
	label := fs.String("label", "", "name of this run, for the result file")
	machine := fs.String("machine", "", "machine directory under bench/results/ (default: the host's short name)")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if *suite == bench.SuiteStandard && (len(sets) > 0 || *profile != "" || *numaFlag != "" || *cold || *advisory || *slow) {
		return errors.New("bench: --set, --profile, --numa, --cold and --advisory change what is measured; use --suite dev")
	}
	variants, err := bench.ParseSets(sets)
	if err != nil {
		return err
	}
	var place bench.Placement
	place.NUMA = *numaFlag
	if _, err := bench.ParseNUMA(*numaFlag); err != nil {
		return err
	}
	if *profile != "" {
		prof, err := bench.ParseProfile(*profile)
		if err != nil {
			return err
		}
		if place.NUMA == "" {
			place.NUMA = prof.Place.NUMA
		}
		place.CPUs = prof.Place.CPUs
		*cold = *cold || prof.Cold
		if len(presets) == 0 && prof.Preset != "" {
			presets = append(presets, prof.Preset)
		}
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	cat, err := registry.Builtin()
	if err != nil {
		return err
	}
	id := cfg.Models.Default
	if len(pos) > 0 {
		id = pos[0]
	}
	m, ok := registry.Find(cat, id)
	if !ok {
		return fmt.Errorf("bench: no %q in the catalog", id)
	}
	// On the fake backend alone (development, the service's e2e) there are
	// no weights to read and the result is marked as the fake's.
	fakeOnly := len(cfg.Backends.Enabled) == 1 && cfg.Backends.Enabled[0] == config.BackendFake
	modelPath, drafterPath := "fake.gguf", "fake-drafter.gguf"
	store := newStore(cfg)
	if !fakeOnly {
		if modelPath, drafterPath, err = store.Ready(m); err != nil {
			return err
		}
	}
	if ov := cfg.Models.Overrides[m.ID]; ov.Speculative.Type == config.SpecNone {
		drafterPath = ""
	}
	libDir, err := backend.ResolveLibDir(cfg.Backends.LibDir)
	if err != nil {
		return err
	}
	name := config.BackendLlamaCpp
	if fakeOnly {
		name = config.BackendFake
	}
	b, err := backend.Open(name, libDir)
	if err != nil {
		return err
	}
	defer func() { _ = b.Close() }()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	res, err := bench.Run(ctx, bench.Options{
		Manifest: m, ModelPath: modelPath, DrafterPath: drafterPath, Backend: b, LibDir: libDir,
		Relaid: func() (registry.Relaid, error) { return store.Relaid(m) },
		Suite:  *suite, Full: *full, Presets: presets, Reps: *reps, Variants: variants, Place: place, Cold: *cold, Advisory: *advisory, Timeout: *timeout, Slow: *slow,
		Version: version, Tag: llamacpp.PinnedTag, Commit: llamacpp.PinnedCommit,
		Events: func(e bench.Event) {
			switch e.Type {
			case "start":
				fmt.Fprintf(os.Stderr, "[%d/%d] %s %s …\n", e.Index+1, e.Total, e.Preset, e.Variant)
			case "result":
				r := e.Run
				fmt.Fprintf(os.Stderr, "[%d/%d] %-8s %-24s %-8s pp %.1f tok/s  tg %.1f tok/s  %s\n", e.Index+1, e.Total, r.Preset, r.Variant, r.Status, r.Metrics.PPTokS, r.Metrics.TGTokS.D0, r.Reason)
			case "estimate":
				fmt.Fprintf(os.Stderr, "      about %.0f min left\n", e.RemainS/60)
			}
		},
	})
	if res == nil {
		return err
	}
	if *out == "" {
		*out = resultPath(*machine, res.Date, *suite, *label)
	}
	if werr := writeJSON(*out, res); werr != nil {
		return werr
	}
	fmt.Fprintln(os.Stderr, "result:", *out)
	if err != nil {
		return err
	}
	if *suite != bench.SuiteStandard || cfg.Bench.Submit == "never" {
		return nil
	}
	return offer(ctx, cfg, res, *submit)
}

// offer shows the exact JSON and the consent text and sends only on yes.
func offer(ctx context.Context, cfg config.Config, res *benchfmt.Result, yes bool) error {
	if !yes {
		b, _ := json.MarshalIndent(res, "", "  ")
		fmt.Printf("%s\n\n%s\n\nОтправить? / Send? [y/N]: ", b, bench.ConsentText)
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "y", "yes", "д", "да":
		default:
			fmt.Println("Не отправлено. Позже: nanollm bench submit FILE")
			return nil
		}
	}
	return send(ctx, cfg, res)
}

func send(ctx context.Context, cfg config.Config, res *benchfmt.Result) error {
	id := bench.Anonymous()
	if !cfg.Bench.Anonymous {
		var err error
		if id, err = bench.LoadIdentity(cfg.Server.DataDir); err != nil {
			return err
		}
	}
	c := &bench.Client{BaseURL: cfg.Bench.SubmitURL}
	r, err := c.Submit(ctx, res, id)
	if err != nil {
		return err
	}
	fmt.Printf("sent: %s %s\n", r.ID, r.URL)
	return nil
}

func benchSubmit(args []string) error {
	fs := flag.NewFlagSet("bench submit", flag.ExitOnError)
	cfgPath := configFlag(fs)
	yes := fs.Bool("yes", false, "do not ask")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("usage: nanollm bench submit FILE")
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	b, err := os.ReadFile(pos[0])
	if err != nil {
		return err
	}
	res, err := benchfmt.Parse(b)
	if err != nil {
		return err
	}
	res.InstallID, res.Signature = "", ""
	return offer(context.Background(), cfg, res, *yes)
}

func benchWithdraw(args []string) error {
	fs := flag.NewFlagSet("bench withdraw", flag.ExitOnError)
	cfgPath := configFlag(fs)
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("usage: nanollm bench withdraw ID")
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	id, err := bench.LoadIdentity(cfg.Server.DataDir)
	if err != nil {
		return err
	}
	c := &bench.Client{BaseURL: cfg.Bench.SubmitURL}
	if err := c.Withdraw(context.Background(), pos[0], id); err != nil {
		return err
	}
	fmt.Println("withdrawn:", pos[0])
	return nil
}

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

// resultPath is bench/results/<machine>/<date>-<suite>[-<label>].json; a
// result already there is never overwritten, the name gets -2, -3, …
func resultPath(machine, date, suite, label string) string {
	if machine == "" {
		machine, _ = os.Hostname()
		machine, _, _ = strings.Cut(machine, ".")
	}
	dir := filepath.Join("bench", "results", slug(machine))
	_ = os.MkdirAll(dir, 0o755)
	name := date + "-" + suite
	if label != "" {
		name += "-" + slug(label)
	}
	path := filepath.Join(dir, name+".json")
	for n := 2; ; n++ {
		if _, err := os.Stat(path); err != nil {
			return path
		}
		path = filepath.Join(dir, fmt.Sprintf("%s-%d.json", name, n))
	}
}

// slug keeps a name safe as a path element.
func slug(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			return r
		}
		return '-'
	}, s)
}

// benchCompare is `nanollm bench compare A.json [B.json]`: the runs of A
// against those of B, or the variants of one sweep against its first.
func benchCompare(args []string) error {
	pos := args
	if len(pos) < 1 || len(pos) > 2 {
		return errors.New("usage: nanollm bench compare A.json [B.json]")
	}
	load := func(path string) (*benchfmt.Result, error) {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		// Not benchfmt.Parse: a development sweep has more runs than the
		// public format allows.
		var r benchfmt.Result
		if err := json.Unmarshal(b, &r); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		return &r, nil
	}
	a, err := load(pos[0])
	if err != nil {
		return err
	}
	var b *benchfmt.Result
	if len(pos) == 2 {
		if b, err = load(pos[1]); err != nil {
			return err
		}
	}
	rows := bench.Compare(a, b)
	if len(rows) == 0 {
		return errors.New("bench compare: no run of the same preset and variant in both files")
	}
	bench.WriteComparison(os.Stdout, rows)
	return nil
}
