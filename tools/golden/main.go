// Command golden records reference output from llama.cpp's own llama-server
// of the pinned tag, for the binding's golden tests (STAGE1 1.5, Golden).
//
//	scripts/fetch-libs.sh --tools
//	go run ./tools/golden -model gemma-4-e4b.gguf \
//	    -out testdata/golden/b11600/gemma-4-e4b-qat/cpu-haswell.json
//
// The file name carries what the golden depends on: cpu-<variant>[-<kv>].json,
// and cpu-mtp-<variant>[-<kv>].json with a drafter (the KV type is left out
// for f16). A generic CPU library has no variant (darwin), and the CPU's name
// stands in: cpu-apple-m1.json. With -device the whole model runs on that
// ggml device, and its kind and name stand in for the CPU's:
//
//	go run ./tools/golden -device mtl0 -model gemma-4-e4b.gguf \
//	    -out testdata/golden/b11600/gemma-4-e4b-qat/mtl-apple-m1.json
//
// The native tests pick every file whose kernels and model match the run
// (NANOLLM_TEST_DEVICE for a GPU); the tool says when -out is not that name.
//
// It starts llama-server on loopback with the settings the native test uses
// (one slot, flash attention, f16 KV, repacked CPU weights, n_batch and
// n_ubatch 512), asks it to tokenize the prompt and to complete it greedily,
// and writes the prompt, both token lists, the text and the ggml CPU variant
// the server picked (or the device). Greedy decoding on the same kernels is
// deterministic, so the test compares ids exactly; on another CPU variant or
// another GPU the kernels differ and the test skips.
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
	// CPUName names the CPU of a recording on it (sysctl's brand string on
	// darwin), for the file name where the library has no variant.
	CPUName string `json:"cpu_name,omitempty"`
	// Device is the ggml device the whole model ran on (mtl0, cuda0), and
	// DeviceName its description (Apple M1); empty is the CPU.
	Device     string `json:"device,omitempty"`
	DeviceName string `json:"device_name,omitempty"`
	Threads    int    `json:"threads"`
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
	device := flag.String("device", "", "ggml device to run the whole model on (mtl0, cuda0: llama-server --list-devices); empty is the CPU")
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
	if err := run(*tools, *model, *out, *prompt, *n, *threads, *kv, *device, *drafter, *draftMax, *draftPMin); err != nil {
		fmt.Fprintln(os.Stderr, "golden:", err)
		os.Exit(1)
	}
}

func defaultTools() string {
	return filepath.Join("dist", runtime.GOOS+"_"+runtime.GOARCH, "tools")
}

func run(tools, model, out, prompt string, n, threads int, kv, device, drafter string, draftMax int, draftPMin float64) error {
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
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	// The CPU alone, as the binding loads a model for the CPU device: with no
	// -dev llama-server offloads to every GPU it finds (Metal on a Mac).
	dev, layers := "none", "0"
	if device != "" {
		name, desc, err := findDevice(ctx, server, device)
		if err != nil {
			return err
		}
		g.Device, g.DeviceName = strings.ToLower(name), desc
		g.Generator += ", on " + name
		dev, layers = name, "all"
	} else {
		g.CPUName = cpuName(ctx)
	}
	extra := []string{"-dev", dev, "-ngl", layers}
	if drafter != "" {
		dst, err := os.Stat(drafter)
		if err != nil {
			return err
		}
		g.DrafterFile, g.DrafterSize, g.DraftMax, g.DraftPMin = filepath.Base(drafter), dst.Size(), draftMax, draftPMin
		g.DrafterSHA = sha256File(drafter)
		g.Generator += ", MTP draft-mtp"
		extra = append(extra, "-md", drafter, "--spec-type", "draft-mtp",
			"--spec-draft-n-max", strconv.Itoa(draftMax), "--spec-draft-p-min", strconv.FormatFloat(draftPMin, 'f', -1, 64),
			// The binding's drafter inherits the target's cache types and
			// runs on the target's device.
			"-ctkd", kv, "-ctvd", kv, "-devd", dev, "-ngld", layers)
	}
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
	on := "cpu " + g.CPUVariant
	if g.Device != "" {
		on = g.Device + " (" + g.DeviceName + ")"
	}
	fmt.Printf("%s: %d prompt tokens, %d generated, %s, kv %s, drafted %d accepted %d\n%s\n",
		out, len(g.PromptTokens), len(g.Tokens), on, kv, g.Drafted, g.Accepted, g.Text)
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

// listedDevice is a line of llama-server --list-devices: "  MTL0: Apple M1 (5461 MiB, 5460 MiB free)".
var listedDevice = regexp.MustCompile(`^\s+(\S+): (.+) \(\d+ MiB, \d+ MiB free\)$`)

// findDevice looks id up in llama-server --list-devices regardless of case
// (ggml lists MTL0 and Vulkan0) and returns the name -dev takes and the
// device's description, which tells GPUs of one backend apart.
func findDevice(ctx context.Context, server, id string) (name, desc string, err error) {
	out, err := exec.CommandContext(ctx, server, "--list-devices").CombinedOutput()
	if err != nil {
		return "", "", fmt.Errorf("llama-server --list-devices: %w\n%s", err, out)
	}
	var names []string
	for _, line := range strings.Split(string(out), "\n") {
		m := listedDevice.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		if strings.EqualFold(m[1], id) {
			return m[1], m[2], nil
		}
		names = append(names, m[1])
	}
	return "", "", fmt.Errorf("llama-server lists no device %q (it has: %s)", id, strings.Join(names, ", "))
}

// cpuName is the CPU's brand string on darwin, where ggml's CPU library has
// no variant to name a golden by; empty elsewhere.
func cpuName(ctx context.Context) string {
	if runtime.GOOS != "darwin" {
		return ""
	}
	out, err := exec.CommandContext(ctx, "sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// FileName is the name a golden file is stored under: cpu-<variant>[-<kv>] or,
// recorded with a drafter, cpu-mtp-<variant>[-<kv>]. A CPU library without a
// variant is named by the CPU (cpu-apple-m1), a device by its kind and name
// (mtl-apple-m1, mtl-mtp-apple-m1).
func FileName(g Golden) string {
	kind, who := "cpu", g.CPUVariant
	switch {
	case g.Device != "":
		kind, who = strings.TrimRight(g.Device, "0123456789"), slug(g.DeviceName)
	case who == "" && g.CPUName != "":
		who = slug(g.CPUName)
	case who == "":
		who = "generic"
	}
	name := kind + "-"
	if g.DrafterFile != "" {
		name += "mtp-"
	}
	name += who
	if g.KVType != "" && g.KVType != "f16" {
		name += "-" + g.KVType
	}
	return name + ".json"
}

// slug turns "Apple M1" into "apple-m1".
func slug(s string) string {
	var b strings.Builder
	for _, f := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return (r < 'a' || r > 'z') && (r < '0' || r > '9')
	}) {
		if b.Len() > 0 {
			b.WriteByte('-')
		}
		b.WriteString(f)
	}
	return b.String()
}
