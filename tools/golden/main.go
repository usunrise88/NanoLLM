// Command golden records reference output from llama.cpp's own llama-server
// of the pinned tag, for the binding's golden tests (STAGE1 1.5, Golden).
//
//	scripts/fetch-libs.sh --tools
//	go run ./tools/golden -model gemma-4-e4b.gguf \
//	    -out testdata/golden/b11600/gemma-4-e4b-qat/cpu-haswell.json
//
// The file name carries what the golden depends on: cpu-<variant>[-<kv>].json,
// and cpu-mtp-<variant>[-<kv>].json with a drafter (the KV type is left out
// for f16). The native tests pick every file whose variant and model match
// the machine they run on; the tool says when -out is not that name.
//
// It starts llama-server on loopback with the settings the native test uses
// (one slot, flash attention, f16 KV, repacked CPU weights, n_batch and
// n_ubatch 512), asks it to tokenize the prompt and to complete it greedily,
// and writes the prompt, both token lists, the text and the ggml CPU variant
// the server picked. Greedy decoding on the same CPU kernels is
// deterministic, so the test compares ids exactly; on another CPU variant
// the kernels differ and the test skips.
//
// With -drafter the server runs with the MTP drafter (--spec-type
// draft-mtp, greedy drafting) and the file also records how many tokens it
// drafted and accepted, the reference for the binding's MTP state machine
// (STAGE1 1.5.M): same tokens, same counts.
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Golden is the file format, shared with the native test by copy.
type Golden struct {
	Tag         string `json:"tag"`
	Commit      string `json:"commit"`
	Generator   string `json:"generator"`
	ModelFile   string `json:"model_file"`
	ModelSize   int64  `json:"model_size"`
	ModelSHA256 string `json:"model_sha256"`
	CPUVariant  string `json:"cpu_variant"`
	CPUFeatures string `json:"cpu_features,omitempty"`
	Threads     int    `json:"threads"`
	// KVType is the K and V cache type, and the drafter's with MTP; empty is
	// f16.
	KVType       string  `json:"kv_type,omitempty"`
	NCtx         int     `json:"n_ctx"`
	NBatch       int     `json:"n_batch"`
	NUBatch      int     `json:"n_ubatch"`
	Prompt       string  `json:"prompt"`
	PromptTokens []int32 `json:"prompt_tokens"`
	NPredict     int     `json:"n_predict"`
	Tokens       []int32 `json:"tokens"`
	Text         string  `json:"text"`
	// MTP, when the golden was recorded with a drafter.
	DrafterFile string   `json:"drafter_file,omitempty"`
	DrafterSize int64    `json:"drafter_size,omitempty"`
	DrafterSHA  string   `json:"drafter_sha256,omitempty"`
	DraftMax    int      `json:"draft_max,omitempty"`
	DraftPMin   float64  `json:"draft_p_min,omitempty"`
	Drafted     int      `json:"drafted,omitempty"`
	Accepted    int      `json:"accepted,omitempty"`
	Notes       []string `json:"notes,omitempty"`
}

// DefaultPrompt is the chat-wrapped Gemma 4 prompt the test uses.
const DefaultPrompt = "<|turn>user\nWrite a short paragraph about the history of Paris.<turn|>\n<|turn>model\n"

func main() {
	tools := flag.String("tools", defaultTools(), "directory with llama-server (scripts/fetch-libs.sh --tools)")
	model := flag.String("model", "", "GGUF model file")
	out := flag.String("out", "", "output JSON file")
	prompt := flag.String("prompt", DefaultPrompt, "prompt, sent as is with BOS")
	n := flag.Int("n", 32, "tokens to generate")
	threads := flag.Int("threads", 4, "CPU threads")
	kv := flag.String("kv", "f16", "KV cache type for K and V (and the drafter's): f16 or q8_0")
	drafter := flag.String("drafter", "", "MTP drafter GGUF: record a speculative run")
	draftMax := flag.Int("draft-max", 3, "tokens drafted per step at most")
	draftPMin := flag.Float64("draft-p-min", 0, "draft probability floor")
	flag.Parse()
	if *model == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "golden: -model and -out are required")
		os.Exit(2)
	}
	if *kv != "f16" && *kv != "q8_0" {
		fmt.Fprintln(os.Stderr, "golden: -kv must be f16 or q8_0")
		os.Exit(2)
	}
	if err := run(*tools, *model, *out, *prompt, *n, *threads, *kv, *drafter, *draftMax, *draftPMin); err != nil {
		fmt.Fprintln(os.Stderr, "golden:", err)
		os.Exit(1)
	}
}

func defaultTools() string {
	return filepath.Join("dist", runtime.GOOS+"_"+runtime.GOARCH, "tools")
}

func run(tools, model, out, prompt string, n, threads int, kv, drafter string, draftMax int, draftPMin float64) error {
	st, err := os.Stat(model)
	if err != nil {
		return err
	}
	tag, err := readTrimmed("scripts/llama.cpp.tag")
	if err != nil {
		return err
	}
	commit, err := readTrimmed("scripts/llama.cpp.commit")
	if err != nil {
		return err
	}
	var lc net.ListenConfig
	ln, err := lc.Listen(context.Background(), "tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()

	server := filepath.Join(tools, "llama-server")
	if runtime.GOOS == "windows" {
		server += ".exe"
	}
	g := Golden{
		Tag: tag, Commit: commit,
		Generator: "llama-server " + tag + " /tokenize + /completion, greedy (top_k 1)",
		ModelFile: filepath.Base(model), ModelSize: st.Size(), ModelSHA256: sha256File(model),
		Threads: threads, NCtx: 4096, NBatch: 512, NUBatch: 512,
		Prompt: prompt, NPredict: n,
	}
	if kv != "f16" {
		g.KVType = kv
	}
	var extra []string
	if drafter != "" {
		dst, err := os.Stat(drafter)
		if err != nil {
			return err
		}
		g.DrafterFile, g.DrafterSize, g.DraftMax, g.DraftPMin = filepath.Base(drafter), dst.Size(), draftMax, draftPMin
		g.DrafterSHA = sha256File(drafter)
		g.Generator += ", MTP draft-mtp"
		extra = []string{"-md", drafter, "--spec-type", "draft-mtp",
			"--spec-draft-n-max", strconv.Itoa(draftMax), "--spec-draft-p-min", strconv.FormatFloat(draftPMin, 'f', -1, 64),
			// The binding's drafter inherits the target's cache types.
			"-ctkd", kv, "-ctvd", kv}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, server, append([]string{
		"-m", model, "--host", "127.0.0.1", "--port", strconv.Itoa(port),
		"-c", strconv.Itoa(g.NCtx), "-np", "1", "-t", strconv.Itoa(threads), "-tb", strconv.Itoa(threads),
		"-b", strconv.Itoa(g.NBatch), "-ub", strconv.Itoa(g.NUBatch),
		"-fa", "on", "-ctk", kv, "-ctv", kv, "--no-webui",
		// -v: the CPU variant and features are logged at a level the
		// default verbosity hides.
		"-v"}, extra...)...)
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	cmd.Stdout = io.Discard
	if err := cmd.Start(); err != nil {
		return err
	}
	defer func() { _ = cmd.Process.Kill(); _ = cmd.Wait() }()

	cpu := regexp.MustCompile(`loaded CPU backend from (.+)$`)
	features := regexp.MustCompile(`system_info: .*?\| (CPU : .*)$`)
	var mu sync.Mutex
	var logTail []string
	go func() {
		sc := bufio.NewScanner(stderr)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		for sc.Scan() {
			line := sc.Text()
			mu.Lock()
			if m := cpu.FindStringSubmatch(line); m != nil {
				base := strings.TrimSuffix(filepath.Base(m[1]), filepath.Ext(m[1]))
				g.CPUVariant = strings.TrimPrefix(strings.TrimPrefix(strings.TrimPrefix(base, "lib"), "ggml-cpu"), "-")
			}
			if m := features.FindStringSubmatch(line); m != nil && g.CPUFeatures == "" {
				g.CPUFeatures = strings.TrimSpace(strings.TrimRight(m[1], "| "))
			}
			logTail = append(logTail, line)
			if len(logTail) > 20 {
				logTail = logTail[1:]
			}
			mu.Unlock()
		}
	}()

	base := fmt.Sprintf("http://127.0.0.1:%d", port)
	for {
		if ctx.Err() != nil {
			mu.Lock()
			defer mu.Unlock()
			return fmt.Errorf("llama-server did not become ready:\n%s", strings.Join(logTail, "\n"))
		}
		resp, err := get(ctx, base+"/health")
		if err == nil && resp == http.StatusOK {
			break
		}
		time.Sleep(time.Second)
	}

	var tok struct {
		Tokens []int32 `json:"tokens"`
	}
	if err := post(ctx, base+"/tokenize", map[string]any{"content": prompt, "add_special": true, "parse_special": true}, &tok); err != nil {
		return err
	}
	g.PromptTokens = tok.Tokens

	var comp struct {
		Content string  `json:"content"`
		Tokens  []int32 `json:"tokens"`
		Stop    string  `json:"stop_type"`
		Timings struct {
			DraftN         int `json:"draft_n"`
			DraftNAccepted int `json:"draft_n_accepted"`
		} `json:"timings"`
	}
	if err := post(ctx, base+"/completion", map[string]any{
		"prompt": prompt, "n_predict": n,
		// Greedy: top-k 1 as the last sampler; temperature 0 also keeps the
		// server from verifying drafts by rejection sampling.
		"samplers": []string{"top_k"}, "top_k": 1, "temperature": 0,
		"cache_prompt": false, "return_tokens": true, "n_probs": 0,
	}, &comp); err != nil {
		return err
	}
	g.Tokens = comp.Tokens
	g.Text = comp.Content
	g.Drafted, g.Accepted = comp.Timings.DraftN, comp.Timings.DraftNAccepted
	if comp.Stop != "" && comp.Stop != "limit" {
		g.Notes = append(g.Notes, "stopped by "+comp.Stop)
	}
	if len(g.Tokens) == 0 {
		return errors.New("the completion returned no tokens")
	}

	data, err := json.MarshalIndent(g, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(out, append(data, '\n'), 0o644); err != nil {
		return err
	}
	fmt.Printf("%s: %d prompt tokens, %d generated, cpu %s, kv %s, drafted %d accepted %d\n%s\n",
		out, len(g.PromptTokens), len(g.Tokens), g.CPUVariant, kv, g.Drafted, g.Accepted, g.Text)
	if want := FileName(g); filepath.Base(out) != want {
		fmt.Printf("note: the tests expect this golden to be named %s\n", want)
	}
	return nil
}

func get(ctx context.Context, url string) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	resp.Body.Close()
	return resp.StatusCode, nil
}

func post(ctx context.Context, url string, body, out any) error {
	b, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s: %s", url, resp.Status, data)
	}
	return json.Unmarshal(data, out)
}

// sha256File identifies a model file: the size alone does not, since a
// regenerated fixture has the same shape and different weights.
func sha256File(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

func readTrimmed(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("%w (run from the repository root)", err)
	}
	return strings.TrimSpace(string(b)), nil
}

// FileName is the name a golden file is stored under: cpu-<variant>[-<kv>] or,
// recorded with a drafter, cpu-mtp-<variant>[-<kv>].
func FileName(g Golden) string {
	name := "cpu-"
	if g.DrafterFile != "" {
		name += "mtp-"
	}
	name += g.CPUVariant
	if g.KVType != "" && g.KVType != "f16" {
		name += "-" + g.KVType
	}
	return name + ".json"
}
