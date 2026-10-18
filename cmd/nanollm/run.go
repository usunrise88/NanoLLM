package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
)

// runCommand has two paths. On a catalog model (the default, or an ID) it is
// a chat through core.Service, the server's own path: one answer with
// --prompt, a REPL without. On a GGUF file it is the binding's one-shot: it
// drives the backend directly, prefill chunks of n_batch then Step, which is
// what the spike and the binding checks need.
func runCommand(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	cfgPath := configFlag(fs)
	libDirFlag := fs.String("lib-dir", "", "native library directory (overrides backends.lib_dir)")
	modelPath := fs.String("model", "", "path to a GGUF file")
	drafterPath := fs.String("drafter", "", "path to the model's MTP drafter GGUF; drafts tokens the model then verifies")
	draftMax := fs.Int("draft-max", 3, "tokens drafted per step at most")
	draftPMin := fs.Float64("draft-p-min", 0, "stop drafting below this probability")
	prompt := fs.String("prompt", "", "the prompt; without it run is an interactive chat")
	preset := fs.String("preset", "", "memory preset for a catalog model (SPEC §4), e.g. ram-8g")
	think := fs.Bool("think", false, "let the model think before it answers (catalog models)")
	system := fs.String("system", "", "system message (catalog models)")
	raw := fs.Bool("raw", false, "send the prompt as is, without the chat turn markers")
	nPredict := fs.Int("n", 256, "maximum tokens to generate")
	temp := fs.Float64("temp", 0, "temperature; 0 is greedy")
	topK := fs.Int("top-k", 64, "top-k when sampling")
	topP := fs.Float64("top-p", 0.95, "top-p when sampling")
	seed := fs.Int64("seed", -1, "sampling seed; -1 is random")
	device := fs.String("device", "", "device selector, e.g. cpu or llamacpp:vulkan0; a file defaults to cpu, a catalog model to devices.default")
	gpuLayers := fs.Int("gpu-layers", -1, "layers to offload to a GPU device; -1 is all")
	nCtx := fs.Int("ctx", 4096, "context size in tokens")
	nBatch := fs.Int("batch", 512, "logical batch size")
	nUBatch := fs.Int("ubatch", 512, "physical batch size")
	threads := fs.Int("threads", 0, "CPU threads; 0 is llama.cpp's default")
	kvType := fs.String("kv", "f16", "KV cache type: f16 or q8_0")
	noFA := fs.Bool("no-flash-attn", false, "disable flash attention")
	noRepack := fs.Bool("no-repack", false, "disable the CPU repacked weight layouts")
	verbose := fs.Bool("verbose", false, "show llama.cpp's log")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if *modelPath == "" && len(pos) > 0 {
		*modelPath = pos[0]
	}
	if strings.HasSuffix(strings.ToLower(*modelPath), ".gguf") && !isFile(*modelPath) {
		return fmt.Errorf("run: no such file %s", *modelPath)
	}
	if !isFile(*modelPath) {
		set := map[string]bool{}
		fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
		for _, name := range []string{"drafter", "draft-max", "draft-p-min", "raw", "gpu-layers", "ctx", "batch", "ubatch", "threads", "kv", "no-flash-attn", "no-repack", "lib-dir"} {
			if set[name] {
				return fmt.Errorf("run: --%s applies to a GGUF file; a catalog model takes its placement from the planner (--preset, --device)", name)
			}
		}
		var sp core.Sampling
		if set["temp"] {
			sp.Temperature = temp
		}
		if set["top-k"] {
			sp.TopK = topK
		}
		if set["top-p"] {
			sp.TopP = topP
		}
		if set["seed"] {
			sp.Seed = seed
		}
		maxTokens := 0
		if set["n"] {
			maxTokens = *nPredict
		}
		return runChat(chatOptions{
			cfgPath: *cfgPath, model: *modelPath, prompt: *prompt, preset: *preset, device: *device,
			system: *system, think: *think, maxTokens: maxTokens, sampling: sp, verbose: *verbose,
		}, os.Stdin, os.Stdout, os.Stderr)
	}
	if *prompt == "" {
		return errors.New("run: a GGUF file takes --prompt; the interactive chat runs on catalog models (nanollm run [MODEL_ID])")
	}
	if *preset != "" || *think || *system != "" {
		return errors.New("run: --preset, --think and --system apply to catalog models, not to a GGUF file")
	}
	if *device == "" {
		*device = "cpu"
	}
	if *verbose {
		slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	libDir, err := resolveLibDir(cfg, *libDirFlag)
	if err != nil {
		return err
	}
	b, err := backend.Open("llamacpp", libDir)
	if err != nil {
		return err
	}
	defer func() { _ = b.Close() }()

	start := time.Now()
	m, err := b.Load(context.Background(), backend.LoadSpec{
		Path:           *modelPath,
		Device:         *device,
		GPULayers:      *gpuLayers,
		Repack:         !*noRepack,
		NCtx:           *nCtx,
		Slots:          1,
		NBatch:         *nBatch,
		NUBatch:        *nUBatch,
		Threads:        *threads,
		KVTypeK:        *kvType,
		KVTypeV:        *kvType,
		FlashAttention: !*noFA,
		DrafterPath:    *drafterPath,
		DraftMax:       *draftMax,
		DraftPMin:      *draftPMin,
	})
	if err != nil {
		return err
	}
	defer func() { _ = m.Close() }()
	fmt.Fprintf(os.Stderr, "loaded %s in %.1fs\n", *modelPath, time.Since(start).Seconds())

	text := *prompt
	if !*raw {
		text = chatPrompt(m.Metadata()["general.architecture"], text)
	}
	r, err := generate(context.Background(), m, text, *nPredict, backend.SamplerSpec{
		Temperature: *temp, TopK: *topK, TopP: *topP, Seed: *seed,
	}, os.Stdout)
	if err != nil {
		return err
	}
	fmt.Println()
	fmt.Fprintf(os.Stderr, "prompt %d tokens, %.1f tok/s; generated %d tokens, %.2f tok/s\n",
		r.promptTokens, r.prefillRate(), r.generated, r.decodeRate())
	if r.drafted > 0 {
		fmt.Fprintf(os.Stderr, "MTP: %d drafted, %d accepted (%.0f%%) in %d steps\n",
			r.drafted, r.accepted, 100*float64(r.accepted)/float64(r.drafted), r.steps)
	}
	return nil
}

func isFile(p string) bool {
	if p == "" {
		return false
	}
	fi, err := os.Stat(p)
	return err == nil && fi.Mode().IsRegular()
}

// chatPrompt wraps one user message in the model's turn markers. The real
// chat templates are rendered in Go with the server (llama.cpp's built-in
// template engine does not know Gemma 4); this is the minimal Gemma 4 form.
func chatPrompt(arch, prompt string) string {
	if strings.HasPrefix(arch, "gemma4") {
		return "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n"
	}
	return prompt
}

type genResult struct {
	tokens       []backend.Token
	promptTokens int
	generated    int
	prefill      time.Duration
	decode       time.Duration
	// drafted and accepted count MTP draft tokens; steps counts decode steps.
	drafted, accepted, steps int
}

func (r genResult) prefillRate() float64 {
	return float64(r.promptTokens) / r.prefill.Seconds()
}

func (r genResult) decodeRate() float64 {
	if r.generated <= 1 {
		return 0
	}
	// The first token comes out of the prefill step.
	return float64(r.generated-1) / r.decode.Seconds()
}

// generate prefills the prompt in n_batch chunks on slot 0 and decodes until
// an end-of-generation token or n tokens, writing pieces to w as they come.
func generate(ctx context.Context, m backend.Model, prompt string, n int, sp backend.SamplerSpec, w io.Writer) (genResult, error) {
	var r genResult
	toks, err := m.Tokenize(prompt, true, true)
	if err != nil {
		return r, err
	}
	if len(toks) == 0 {
		return r, errors.New("run: the prompt tokenizes to nothing")
	}
	caps := m.Capabilities()
	if len(toks)+n > caps.NCtx {
		n = caps.NCtx - len(toks)
		if n <= 0 {
			return r, fmt.Errorf("run: the prompt is %d tokens, the context %d", len(toks), caps.NCtx)
		}
	}
	m.ClearSeq(0)
	if err := m.SetSampler(0, sp, toks); err != nil {
		return r, err
	}
	r.promptTokens = len(toks)
	start := time.Now()
	var out []backend.SeqOut
	for p := 0; p < len(toks); p += caps.NBatch {
		end := min(p+caps.NBatch, len(toks))
		out, err = m.Step(ctx, []backend.SeqStep{{Seq: 0, Tokens: toks[p:end], Pos: p, Sample: end == len(toks), Logprobs: -1}})
		if err != nil {
			return r, err
		}
	}
	r.prefill = time.Since(start)
	start = time.Now()
	pos := len(toks)
	for {
		r.drafted += out[0].Drafted
		r.accepted += out[0].Accepted
		toks := out[0].Tokens
		for i, tok := range toks {
			r.tokens = append(r.tokens, tok)
			r.generated++
			if out[0].EOG && i == len(toks)-1 {
				r.decode = time.Since(start)
				return r, nil
			}
			if w != nil {
				piece, err := m.TokenPiece(tok)
				if err != nil {
					return r, err
				}
				if _, err := io.WriteString(w, piece); err != nil {
					return r, err
				}
			}
			if r.generated >= n {
				r.decode = time.Since(start)
				return r, nil
			}
		}
		// All but the last returned token are in the cache.
		pos += len(toks) - 1
		draft := min(caps.MaxDraft, n-r.generated-1)
		out, err = m.Step(ctx, []backend.SeqStep{{Seq: 0, Tokens: toks[len(toks)-1:], Pos: pos, Sample: true, MaxDraft: max(draft, 0), Logprobs: -1}})
		if err != nil {
			return r, err
		}
		r.steps++
		pos++
	}
}
