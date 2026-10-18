// Command optimize measures whether a runtime configuration changes what the
// model computes: KL divergence and top-1 agreement against the reference
// configuration (CPU, KV f16, no MTP, the pinned tag), by llama-perplexity of
// the same tag (ROADMAP «Гейт качества», STAGE1 1.10).
//
//	go run ./tools/optimize base -tools DIR -model GGUF -corpus TXT -logits OUT.bin [-- llama-perplexity flags]
//	go run ./tools/optimize kld  -tools DIR -model GGUF -corpus TXT -logits BASE.bin -name NAME [-out REPORT.json] [-- flags]
//
// base saves the reference logits once; kld runs a candidate configuration
// (the flags after -- : -ngl, -ctk q8_0, -dev …) against them and writes an
// eval.v1 report whose extra block carries the numbers. The threshold is
// calibrated after the first CPU-versus-GPU run; nothing here passes or fails
// a configuration.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: optimize base|kld [flags] [-- llama-perplexity flags]")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "base":
		err = run(os.Args[2:], true)
	case "kld":
		err = run(os.Args[2:], false)
	default:
		err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "optimize:", err)
		os.Exit(1)
	}
}

func run(args []string, base bool) error {
	var extra []string
	for i, a := range args {
		if a == "--" {
			args, extra = args[:i], args[i+1:]
			break
		}
	}
	fs := flag.NewFlagSet("optimize", flag.ExitOnError)
	tools := fs.String("tools", "", "directory with llama-perplexity of the pinned tag (scripts/fetch-libs.sh --tools)")
	model := fs.String("model", "", "the GGUF")
	corpus := fs.String("corpus", "", "plain-text corpus (ru/en)")
	logits := fs.String("logits", "", "reference logits file")
	name := fs.String("name", "candidate", "kld: the configuration's name in the report")
	out := fs.String("out", "", "kld: report file (default kld-<name>.json)")
	ctx := fs.Int("ctx", 512, "chunk size in tokens")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *tools == "" || *model == "" || *corpus == "" || *logits == "" {
		return errors.New("-tools, -model, -corpus and -logits are required")
	}
	bin := filepath.Join(*tools, "llama-perplexity")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	cmdArgs := []string{"-m", *model, "-f", *corpus, "-c", strconv.Itoa(*ctx)}
	if base {
		cmdArgs = append(cmdArgs, "--kl-divergence-base", *logits)
	} else {
		cmdArgs = append(cmdArgs, "--kl-divergence-base", *logits, "--kl-divergence")
	}
	cmdArgs = append(cmdArgs, extra...)
	cmd := exec.CommandContext(context.Background(), bin, cmdArgs...)
	cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+*tools)
	var log strings.Builder
	cmd.Stdout = io.MultiWriter(&log, os.Stderr)
	cmd.Stderr = io.MultiWriter(&log, os.Stderr)
	start := time.Now()
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s: %w", bin, err)
	}
	if base {
		fmt.Fprintf(os.Stderr, "reference logits in %s (%s)\n", *logits, time.Since(start).Round(time.Second))
		return nil
	}
	nums, err := Parse(strings.NewReader(log.String()))
	if err != nil {
		return err
	}
	rep := map[string]any{
		"schema": "nanollm.eval/v1",
		"suite":  map[string]any{"name": "kld:" + filepath.Base(*corpus), "items": 0},
		"model":  filepath.Base(*model),
		"config": map[string]string{"name": *name, "flags": strings.Join(extra, " ")},
		"date":   time.Now().UTC().Format("2006-01-02"),
		"extra":  nums,
	}
	if *out == "" {
		*out = "kld-" + *name + ".json"
	}
	b, _ := json.MarshalIndent(rep, "", "  ")
	if err := os.WriteFile(*out, append(b, '\n'), 0o644); err != nil {
		return err
	}
	fmt.Printf("%s: mean KLD %.6f, p99 %.6f, top-1 %.3f%% → %s\n", *name, nums["kld_mean"], nums["kld_p99"], nums["top1_pct"], *out)
	return nil
}

var lines = map[string]*regexp.Regexp{
	"kld_mean":   regexp.MustCompile(`Mean\s+KLD:\s+([-0-9.eE+]+)`),
	"kld_median": regexp.MustCompile(`Median\s+KLD:\s+([-0-9.eE+]+)`),
	"kld_max":    regexp.MustCompile(`Maximum\s+KLD:\s+([-0-9.eE+]+)`),
	"kld_p99":    regexp.MustCompile(`99\.0%\s+KLD:\s+([-0-9.eE+]+)`),
	"top1_pct":   regexp.MustCompile(`Same top p:\s+([-0-9.eE+]+)`),
	"ppl_ratio":  regexp.MustCompile(`Mean PPL\(Q\)/PPL\(base\)\s*:\s+([-0-9.eE+]+)`),
}

// Parse reads the numbers llama-perplexity prints after a KL-divergence run.
func Parse(r io.Reader) (map[string]float64, error) {
	out := map[string]float64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for sc.Scan() {
		for k, re := range lines {
			if m := re.FindStringSubmatch(sc.Text()); m != nil {
				if v, err := strconv.ParseFloat(m[1], 64); err == nil {
					out[k] = v
				}
			}
		}
	}
	if _, ok := out["kld_mean"]; !ok {
		return out, errors.New("no KL divergence in the output; was -logits made by `optimize base` with the same model vocabulary?")
	}
	return out, nil
}
