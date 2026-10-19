// Command lastrow checks what llama-perplexity cannot: the next-token
// distribution at the END of a prompt, the one row a prompt-processing batch
// computes logits for. A change that crops the batch before the logits (OPT-11,
// docs/STAGE2.md 2.5) leaves llama-perplexity untouched, since it asks for
// every row; this asks for one.
//
//	go run ./tools/lastrow run  -tools DIR -model GGUF -corpus TXT -n 100 -out A.json
//	go run ./tools/lastrow diff A.json B.json
//
// run starts the llama-server of DIR, sends -n prompts cut from the corpus at
// fixed offsets (cache off, one token, the top 100 log-probabilities) and
// records them. diff compares two recordings, made with two builds on the same
// model: top-1 agreement and the KL divergence of the top-k distributions.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"time"
)

type tokenProb struct {
	ID      int     `json:"id"`
	Logprob float64 `json:"logprob"`
}

// Recording is the file `run` writes.
type Recording struct {
	Tools   string        `json:"tools"`
	Model   string        `json:"model"`
	Threads int           `json:"threads"`
	Prompts []string      `json:"prompts_sha256"`
	Top     [][]tokenProb `json:"top"` // per prompt, descending by probability
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: lastrow run|diff …")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "run":
		err = runCmd(os.Args[2:])
	case "diff":
		err = diffCmd(os.Args[2:])
	default:
		err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "lastrow:", err)
		os.Exit(1)
	}
}

func runCmd(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	tools := fs.String("tools", filepath.Join("dist", runtime.GOOS+"_"+runtime.GOARCH, "tools"), "directory with llama-server")
	model := fs.String("model", "", "GGUF model")
	corpus := fs.String("corpus", ".cache/kld/corpus.txt", "text to cut prompts from")
	n := fs.Int("n", 100, "prompts")
	chars := fs.Int("chars", 1500, "characters per prompt")
	threads := fs.Int("threads", 16, "CPU threads")
	out := fs.String("out", "", "recording file")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *model == "" || *out == "" {
		return errors.New("-model and -out are required")
	}
	text, err := os.ReadFile(*corpus)
	if err != nil {
		return err
	}
	runes := []rune(string(text))
	if len(runes) < *chars*2 {
		return errors.New("the corpus is too short")
	}
	step := (len(runes) - *chars) / *n
	var prompts []string
	for i := 0; i < *n; i++ {
		prompts = append(prompts, string(runes[i*step:i*step+*chars]))
	}

	var lc net.ListenConfig
	ln, err := lc.Listen(context.Background(), "tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, filepath.Join(*tools, "llama-server"),
		"-m", *model, "--host", "127.0.0.1", "--port", strconv.Itoa(port), "-c", "4096", "-np", "1",
		"-t", strconv.Itoa(*threads), "-tb", strconv.Itoa(*threads), "-fa", "on", "--no-webui")
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err := cmd.Start(); err != nil {
		return err
	}
	defer func() { _ = cmd.Process.Kill(); _ = cmd.Wait() }()
	base := fmt.Sprintf("http://127.0.0.1:%d", port)
	for {
		if ctx.Err() != nil {
			return errors.New("llama-server did not become ready")
		}
		if code, err := do(ctx, http.MethodGet, base+"/health", nil, nil); err == nil && code == http.StatusOK {
			break
		}
		time.Sleep(time.Second)
	}

	rec := Recording{Tools: *tools, Model: filepath.Base(*model), Threads: *threads}
	for i, p := range prompts {
		body, _ := json.Marshal(map[string]any{"prompt": p, "n_predict": 1, "n_probs": 100, "temperature": 0, "cache_prompt": false})
		var data []byte
		if code, err := do(ctx, http.MethodPost, base+"/completion", body, &data); err != nil || code != http.StatusOK {
			return fmt.Errorf("prompt %d: status %d: %w", i, code, err)
		}
		var r struct {
			Probs []struct {
				Top []struct {
					ID      int     `json:"id"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"completion_probabilities"`
		}
		if err := json.Unmarshal(data, &r); err != nil || len(r.Probs) == 0 || len(r.Probs[0].Top) == 0 {
			return fmt.Errorf("prompt %d: no probabilities in %.200s", i, data)
		}
		var top []tokenProb
		for _, t := range r.Probs[0].Top {
			top = append(top, tokenProb{t.ID, t.Logprob})
		}
		sort.SliceStable(top, func(a, b int) bool { return top[a].Logprob > top[b].Logprob })
		rec.Top = append(rec.Top, top)
		rec.Prompts = append(rec.Prompts, fmt.Sprintf("%x", len(p)+i))
		if (i+1)%10 == 0 {
			fmt.Fprintf(os.Stderr, "%d/%d\n", i+1, len(prompts))
		}
	}
	b, _ := json.Marshal(rec)
	return os.WriteFile(*out, b, 0o644)
}

func load(path string) (*Recording, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Recording
	return &r, json.Unmarshal(b, &r)
}

// kl is KL(a‖b) over a's top tokens; a token b did not list gets b's smallest
// listed probability, so a missing token costs something but not infinity.
func kl(a, b []tokenProb) float64 {
	pb := map[int]float64{}
	floor := 1.0
	for _, t := range b {
		p := math.Exp(t.Logprob)
		pb[t.ID] = p
		floor = math.Min(floor, p)
	}
	var sumA, sumB float64
	for _, t := range a {
		sumA += math.Exp(t.Logprob)
		if p, ok := pb[t.ID]; ok {
			sumB += p
		} else {
			sumB += floor
		}
	}
	var d float64
	for _, t := range a {
		p := math.Exp(t.Logprob) / sumA
		q, ok := pb[t.ID]
		if !ok {
			q = floor
		}
		q /= sumB
		d += p * math.Log(p/q)
	}
	return d
}

func diffCmd(args []string) error {
	if len(args) != 2 {
		return errors.New("usage: lastrow diff A.json B.json")
	}
	a, err := load(args[0])
	if err != nil {
		return err
	}
	b, err := load(args[1])
	if err != nil {
		return err
	}
	if len(a.Top) != len(b.Top) || len(a.Top) == 0 {
		return errors.New("the recordings have different numbers of prompts")
	}
	var ds []float64
	same := 0
	for i := range a.Top {
		ds = append(ds, kl(a.Top[i], b.Top[i]))
		if a.Top[i][0].ID == b.Top[i][0].ID {
			same++
		}
	}
	sort.Float64s(ds)
	var mean float64
	for _, d := range ds {
		mean += d
	}
	mean /= float64(len(ds))
	fmt.Printf("%d prompts; top-1 agree %d (%.1f%%); KL mean %.3g, median %.3g, p99 %.3g, max %.3g\n",
		len(ds), same, 100*float64(same)/float64(len(ds)), mean, ds[len(ds)/2], ds[min(len(ds)-1, len(ds)*99/100)], ds[len(ds)-1])
	return nil
}

// do sends one request; a response body is returned through out when given.
func do(ctx context.Context, method, url string, body []byte, out *[]byte) (int, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if out != nil {
		*out = data
	}
	return resp.StatusCode, err
}
