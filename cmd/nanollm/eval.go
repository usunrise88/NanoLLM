package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"os/signal"
	"syscall"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/eval"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/server"
)

// evalCommand runs a task suite through the same core the server uses, and
// rates or compares the reports by hand (STAGE1 1.10).
//
//	nanollm eval MODEL --suite NAME|FILE [--thinking off|on|both] [--out FILE]
//	nanollm eval rate REPORT [--all]
//	nanollm eval compare A.json B.json [--out FILE]
func evalCommand(args []string) error {
	if len(args) > 0 {
		switch args[0] {
		case "rate":
			return evalRate(args[1:])
		case "compare":
			return evalCompare(args[1:])
		case "suites":
			for _, s := range eval.Suites() {
				fmt.Println(s)
			}
			return nil
		}
	}
	fs := flag.NewFlagSet("eval", flag.ExitOnError)
	cfgPath := configFlag(fs)
	suite := fs.String("suite", "", "a built-in suite (nanollm eval suites) or a JSONL file")
	thinking := fs.String("thinking", "off", "off, on or both")
	out := fs.String("out", "", "report file (default: eval-<suite>.json)")
	maxTokens := fs.Int("max-tokens", 1024, "per item, unless the item says otherwise")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if *suite == "" {
		return errors.New("eval: --suite is required; one of " + fmt.Sprint(eval.Suites()) + " or a file")
	}
	items, sha, err := eval.LoadSuite(*suite)
	if err != nil {
		return err
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	model := cfg.Models.Default
	if len(pos) > 0 {
		model = pos[0]
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	log := newLogger(config.Log{Level: "warn", Format: "text"}, os.Stderr)
	catalog, err := registry.Builtin()
	if err != nil {
		return err
	}
	backends := openBackends(cfg, log)
	defer func() {
		for _, b := range backends {
			_ = b.Close()
		}
	}()
	rt := server.New(ctx, server.Options{Config: cfg, Catalog: catalog, Store: newStore(cfg), Backends: backends, Log: log})
	defer rt.Close()

	rep, err := eval.Run(ctx, rt.Engine, *suite, sha, items, eval.Options{
		Model: model, Thinking: *thinking, MaxTokens: *maxTokens,
		Progress: func(done, total int, o eval.Outcome) {
			mark := "?"
			switch {
			case o.Pass == nil:
			case *o.Pass:
				mark = "ok"
			default:
				mark = "FAIL"
			}
			fmt.Fprintf(os.Stderr, "[%d/%d] %-28s %-4s %s\n", done, total, eval.RatingKey(o), mark, o.Detail)
		},
	})
	if rep == nil {
		return err
	}
	rep.Config = map[string]string{"device": cfg.Devices.Default}
	if *out == "" {
		*out = "eval-" + sanitize(*suite) + ".json"
	}
	if werr := writeJSON(*out, rep); werr != nil {
		return werr
	}
	for capability, t := range rep.Summary {
		fmt.Printf("%-24s %d/%d (%.0f%%)%s\n", capability, t.Pass, t.N, t.Rate*100, map[bool]string{true: fmt.Sprintf(", %d for a person", t.Human), false: ""}[t.Human > 0])
	}
	fmt.Println("report:", *out)
	return err
}

func evalRate(args []string) error {
	fs := flag.NewFlagSet("eval rate", flag.ExitOnError)
	all := fs.Bool("all", false, "rate every answer, not only the ones checkers leave to a person")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("usage: nanollm eval rate REPORT.json [--all]")
	}
	rep, items, err := loadReport(pos[0])
	if err != nil {
		return err
	}
	n, rerr := eval.RateReport(os.Stdin, os.Stdout, rep, items, *all, rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())))
	if err := writeJSON(pos[0], rep); err != nil {
		return err
	}
	fmt.Printf("\n%d rated, %d in the report\n", n, len(rep.Ratings))
	return rerr
}

func evalCompare(args []string) error {
	fs := flag.NewFlagSet("eval compare", flag.ExitOnError)
	out := fs.String("out", "ab.json", "where the preferences go")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 2 {
		return errors.New("usage: nanollm eval compare A.json B.json [--out FILE]")
	}
	a, items, err := loadReport(pos[0])
	if err != nil {
		return err
	}
	b, _, err := loadReport(pos[1])
	if err != nil {
		return err
	}
	if a.Suite.SHA256 != b.Suite.SHA256 {
		return errors.New("eval compare: the reports ran different suites")
	}
	prefs, cerr := eval.CompareReports(os.Stdin, os.Stdout, pos[0], a, pos[1], b, items, rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())))
	if err := writeJSON(*out, prefs); err != nil {
		return err
	}
	count := map[string]int{}
	for _, p := range prefs {
		count[p.Choice]++
	}
	fmt.Printf("\n%s: %d, %s: %d, tie: %d → %s\n", pos[0], count["a"], pos[1], count["b"], count["tie"], *out)
	return cerr
}

func loadReport(path string) (*eval.Report, map[string]eval.Item, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var rep eval.Report
	if err := json.Unmarshal(b, &rep); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	if rep.Schema != eval.ReportSchema {
		return nil, nil, fmt.Errorf("%s is not an %s report", path, eval.ReportSchema)
	}
	items, sha, err := eval.LoadSuite(rep.Suite.Name)
	if err != nil {
		return nil, nil, err
	}
	if sha != rep.Suite.SHA256 {
		return nil, nil, fmt.Errorf("suite %s changed since the report was made", rep.Suite.Name)
	}
	byID := map[string]eval.Item{}
	for _, it := range items {
		byID[it.ID] = it
	}
	return &rep, byID, nil
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

func sanitize(s string) string {
	out := []rune{}
	for _, r := range s {
		if r == '/' || r == '\\' || r == ' ' {
			r = '_'
		}
		out = append(out, r)
	}
	return string(out)
}
