package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"syscall"

	"github.com/usunrise88/nanollm/internal/backend/llamacpp"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/eval"
	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/server"
)

// evalCommand runs a task suite through the same core the server uses, and
// rates or compares the reports by hand (STAGE1 1.10).
//
//	nanollm eval MODEL --suite NAME|FILE [--thinking off|on|both] [--out FILE]
//	nanollm eval rate REPORT [--all]
//	nanollm eval compare A.json B.json [--out FILE]
//	nanollm eval gate REFERENCE.json CANDIDATE.json [--tolerance 0.1]
func evalCommand(args []string) error {
	if len(args) > 0 {
		switch args[0] {
		case "rate":
			return evalRate(args[1:])
		case "compare":
			return evalCompare(args[1:])
		case "gate":
			return evalGate(args[1:])
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
	rep.Config = runConfig(cfg, rt, model)
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

// runConfig is what the report records about the configuration it ran under,
// so that two reports can be told apart by it: the plan that loaded (preset,
// mode, cache type, MTP), the threads, the NUMA policy, the library build and
// the model file.
func runConfig(cfg config.Config, rt *server.Runtime, model string) map[string]string {
	c := map[string]string{
		"device":  cfg.Devices.Default,
		"numa":    cfg.Devices.NUMA,
		"threads": strconv.Itoa(cmp.Or(cfg.Devices.CPUThreads, hwinfo.DefaultThreads())),
		"tag":     llamacpp.PinnedTag,
		"commit":  llamacpp.PinnedCommit,
	}
	if cat, err := registry.Builtin(); err == nil {
		if m, ok := registry.Find(cat, model); ok {
			if f, ok := m.File(registry.RoleModel); ok {
				c["model_sha256"] = f.SHA256
			}
		}
	}
	for _, st := range rt.Pool.Status() {
		b, err := json.Marshal(st.Info)
		if err != nil {
			continue
		}
		var p struct {
			Preset string `json:"preset"`
			Mode   string `json:"mode"`
			KV     struct {
				TypeK string `json:"type_k"`
			} `json:"kv"`
			MTP struct {
				Enabled bool `json:"enabled"`
			} `json:"mtp"`
			Repack bool `json:"repack"`
		}
		if json.Unmarshal(b, &p) != nil || p.Mode == "" {
			continue
		}
		c["preset"], c["mode"], c["kv"] = p.Preset, p.Mode, p.KV.TypeK
		c["mtp"], c["repack"] = strconv.FormatBool(p.MTP.Enabled), strconv.FormatBool(p.Repack)
		break
	}
	return c
}

// evalGate is `nanollm eval gate REFERENCE.json CANDIDATE.json`: the two
// reports compared capability by capability. It exits non-zero when the
// candidate regressed, so a script can stop on it.
func evalGate(args []string) error {
	fs := flag.NewFlagSet("eval gate", flag.ExitOnError)
	tol := fs.Float64("tolerance", eval.DefaultTolerance, "the drop in a capability's pass rate that is let through")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 2 {
		return errors.New("usage: nanollm eval gate REFERENCE.json CANDIDATE.json [--tolerance 0.1]")
	}
	a, _, err := loadReport(pos[0])
	if err != nil {
		return err
	}
	b, _, err := loadReport(pos[1])
	if err != nil {
		return err
	}
	rows, ok, err := eval.Gate(a, b, *tol)
	if err != nil {
		return err
	}
	for _, side := range []struct {
		name string
		r    *eval.Report
	}{{"reference", a}, {"candidate", b}} {
		keys := make([]string, 0, len(side.r.Config))
		for k := range side.r.Config {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		fmt.Printf("%s:", side.name)
		for _, k := range keys {
			fmt.Printf(" %s=%s", k, side.r.Config[k])
		}
		fmt.Println()
	}
	eval.WriteGate(os.Stdout, rows)
	if !ok {
		return errors.New("the candidate regressed on at least one capability")
	}
	return nil
}
