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
	"strings"
	"syscall"

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
	reps := fs.Int("reps", bench.Reps, "repetitions per scenario")
	submit := fs.Bool("submit", false, "send the result without asking (for scripts)")
	out := fs.String("out", "", "result file (default bench-<date>.json in bench/results/)")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
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
	modelPath, drafterPath, err := newStore(cfg).Ready(m)
	if err != nil {
		return err
	}
	if ov := cfg.Models.Overrides[m.ID]; ov.Speculative.Type == config.SpecNone {
		drafterPath = ""
	}
	libDir, err := backend.ResolveLibDir(cfg.Backends.LibDir)
	if err != nil {
		return err
	}
	b, err := backend.Open(config.BackendLlamaCpp, libDir)
	if err != nil {
		return err
	}
	defer b.Close()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	res, err := bench.Run(ctx, bench.Options{
		Manifest: m, ModelPath: modelPath, DrafterPath: drafterPath, Backend: b, LibDir: libDir,
		Suite: *suite, Full: *full, Presets: presets, Reps: *reps,
		Version: version, Tag: llamacpp.PinnedTag, Commit: llamacpp.PinnedCommit,
		Events: func(e bench.Event) {
			switch e.Type {
			case "start":
				fmt.Fprintf(os.Stderr, "[%d/%d] %s …\n", e.Index+1, e.Total, e.Preset)
			case "result":
				r := e.Run
				fmt.Fprintf(os.Stderr, "[%d/%d] %-16s %-8s pp %.1f tok/s  tg %.1f tok/s  %s\n", e.Index+1, e.Total, r.Preset, r.Status, r.Metrics.PPTokS, r.Metrics.TGTokS.D0, r.Reason)
			case "estimate":
				fmt.Fprintf(os.Stderr, "      about %.0f min left\n", e.RemainS/60)
			}
		},
	})
	if res == nil {
		return err
	}
	if *out == "" {
		dir := "bench/results"
		_ = os.MkdirAll(dir, 0o755)
		*out = fmt.Sprintf("%s/bench-%s-%s.json", dir, res.Date, *suite)
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
