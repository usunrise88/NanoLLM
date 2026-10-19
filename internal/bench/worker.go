package bench

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/chat"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/memlock"
	"github.com/usunrise88/nanollm/internal/sched"
)

// WorkerSpec is what the orchestrator hands a worker on stdin.
type WorkerSpec struct {
	Backend string           `json:"backend"`
	LibDir  string           `json:"lib_dir"`
	Load    backend.LoadSpec `json:"load"`
	Reps    int              `json:"reps"`
	// Locks are the plan's range-lock ranges of the model and the drafter
	// (M7, M8), pinned before the load as the server does.
	Locks        []memlock.Range `json:"locks,omitempty"`
	DrafterLocks []memlock.Range `json:"drafter_locks,omitempty"`
	// Slow shortens the run for a configuration that streams weights from a
	// disk (M8, M9 on a hard disk): pp, then SlowTGTokens of decode, and no
	// depth, chat or MTP scenario.
	Slow bool `json:"slow,omitempty"`
	// Random are ranges of the model file read at random (MADV_RANDOM).
	Random []memlock.Range `json:"random,omitempty"`
	// Quick skips the depth-N decode, the slowest scenario.
	Quick bool `json:"quick"`
	// Echo adds the echo scenario (dev suite): the model copies a given
	// text, which is where prompt lookup drafts pay.
	Echo bool `json:"echo,omitempty"`
	// Place is the worker's NUMA policy and pinned CPUs. numactl applies
	// it; it is here so two placements never share a plan hash.
	Place Placement `json:"place,omitempty"`
	// Env is added to the worker's environment (a sweep's env.* keys).
	Env map[string]string `json:"env,omitempty"`
	// Cold means the orchestrator evicted the model file before the start.
	Cold bool `json:"cold,omitempty"`
	// Strict means the worker runs alone in a cgroup with a memory limit:
	// its memory.peak and page cache are then the run's own, and recorded.
	Strict bool `json:"strict,omitempty"`
}

// WorkerResult is what a worker writes on stdout.
type WorkerResult struct {
	Metrics    benchfmt.Metrics `json:"metrics"`
	Output     benchfmt.Output  `json:"output"`
	CPUVariant string           `json:"cpu_variant,omitempty"`
	Error      string           `json:"error,omitempty"`
}

// cpuVarianter is implemented by backends that know which CPU kernels ggml
// picked (llamacpp).
type cpuVarianter interface{ CPUVariant() string }

// Worker runs every scenario on one placement: it is `nanollm bench worker`,
// one process per preset.
func Worker(ctx context.Context, in io.Reader, out io.Writer) error {
	var spec WorkerSpec
	if err := json.NewDecoder(in).Decode(&spec); err != nil {
		return fmt.Errorf("bench worker: read the spec: %w", err)
	}
	res, err := measure(ctx, spec)
	if err != nil {
		res.Error = err.Error()
	}
	return json.NewEncoder(out).Encode(res)
}

func measure(ctx context.Context, spec WorkerSpec) (WorkerResult, error) {
	var res WorkerResult
	b, err := backend.Open(spec.Backend, spec.LibDir)
	if err != nil {
		return res, err
	}
	if spec.Backend == fake.Name {
		// An instant fake measures millions of tokens a second, which no
		// result format or service accepts; a laptop-like pace keeps the
		// whole chain (format, signing, service, pages) testable on it.
		_ = b.Close()
		b = fake.New(fake.Options{PrefillRate: 4000, DecodeRate: 200, MaxDraft: 2, Acceptance: 0.6})
	}
	defer func() { _ = b.Close() }()
	if cv, ok := b.(cpuVarianter); ok {
		res.CPUVariant = cv.CPUVariant()
	}
	var cg *cgroupWatch
	if spec.Strict {
		cg = watchCgroup()
	}
	start := time.Now()
	for path, ranges := range map[string][]memlock.Range{spec.Load.Path: spec.Locks, spec.Load.DrafterPath: spec.DrafterLocks} {
		if len(ranges) == 0 || path == "" {
			continue
		}
		pin, err := memlock.PinFile(path, ranges)
		if err != nil {
			return res, fmt.Errorf("range-lock: %w", err)
		}
		defer func() { _ = pin.Close() }()
	}
	m, err := b.Load(ctx, spec.Load)
	if err != nil {
		return res, err
	}
	defer func() { _ = m.Close() }()
	if n, err := memlock.AdviseRandom(spec.Load.Path, spec.Random); err == nil && n > 0 {
		fmt.Fprintf(os.Stderr, "madvise random: %d MiB\n", n>>20)
	}
	res.Metrics.LoadMS = ms(time.Since(start))
	loop := sched.New(m)
	defer loop.Close()
	res, err = run(ctx, m, loop, spec, res)
	if cg != nil {
		// Read while the model is still loaded: the peak and the page cache
		// at that point are the run's.
		res.Metrics.PeakCgroup, res.Metrics.PeakPageCache = cg.finish()
	}
	return res, err
}

// run measures the scenarios through the scheduling loop the server uses, so
// the numbers include what serving costs and nothing that it does not.
func run(ctx context.Context, m backend.Model, loop *sched.Loop, spec WorkerSpec, res WorkerResult) (WorkerResult, error) {
	reps := max(spec.Reps, 1)
	caps := m.Capabilities()
	greedy := backend.SamplerSpec{Temperature: 0, TopK: 1, TopP: 1, RepeatPenalty: 1, Seed: Seed, LogitBias: banEOG(m, caps.NVocab)}
	filler, err := repeatTokens(m, ppText, max(PPTokens, DepthN)+64)
	if err != nil {
		return res, err
	}
	gen := func(prompt []backend.Token, n int, noCache, noDraft bool) (*sched.Stats, []backend.Token, error) {
		h, err := loop.Submit(ctx, sched.Request{Prompt: prompt, Sampler: greedy, MaxTokens: n, Logprobs: -1, NoCache: noCache, NoDraft: noDraft})
		if err != nil {
			return nil, nil, err
		}
		var toks []backend.Token
		for {
			ev, err := h.Next(ctx)
			if err != nil {
				return nil, nil, err
			}
			toks = append(toks, ev.Tokens...)
			if ev.Done {
				if ev.Err != nil {
					return nil, nil, ev.Err
				}
				return ev.Stats, toks, nil
			}
		}
	}
	tokPerSec := func(n int, d time.Duration) float64 {
		if d <= 0 || n <= 0 {
			return 0
		}
		return float64(n) / d.Seconds()
	}
	std := map[string]float64{}

	// pp512: prefill only, the cache cleared each time.
	var pp []float64
	for range reps {
		st, _, err := gen(filler[:PPTokens], 1, true, true)
		if err != nil {
			return res, fmt.Errorf("pp%d: %w", PPTokens, err)
		}
		pp = append(pp, tokPerSec(PPTokens, st.Prefill))
	}
	res.Metrics.PPTokS, std["pp_tok_s"] = medianStd(pp)

	// tg128 at depth 0: a short prompt, then TGTokens of decode.
	short := filler[:8]
	var tg0 []float64
	tgTokens := TGTokens
	if spec.Slow {
		tgTokens = SlowTGTokens
	}
	for range reps {
		st, toks, err := gen(short, tgTokens, true, true)
		if err != nil {
			return res, fmt.Errorf("tg%d: %w", tgTokens, err)
		}
		tg0 = append(tg0, tokPerSec(len(toks)-1, st.Decode))
	}
	res.Metrics.TGTokS.D0, std["tg_tok_s.d0"] = medianStd(tg0)
	if spec.Slow {
		res.Metrics.StdDev = std
		return withMemory(m, res), nil
	}

	// tg128 at depth N: the context already holds N tokens. The prefix is
	// prefilled once and reused, as the prompt cache would in serving.
	if !spec.Quick {
		depth := min(DepthN, caps.NCtx-TGTokens-256)
		if depth > PPTokens {
			res.Metrics.TGTokS.DepthN = depth
			var tgN []float64
			for range reps {
				st, toks, err := gen(filler[:depth], TGTokens, false, true)
				if err != nil {
					return res, fmt.Errorf("tg%d at depth %d: %w", TGTokens, depth, err)
				}
				tgN = append(tgN, tokPerSec(len(toks)-1, st.Decode))
			}
			res.Metrics.TGTokS.DN, std["tg_tok_s.dN"] = medianStd(tgN)
		}
	}

	// chat-ru: the whole prompt path, with and without MTP drafting.
	prompt := chat.PromptGemma4([]core.Message{{Role: core.RoleSystem, Content: chatSystem}, {Role: core.RoleUser, Content: chatPrompt}}, nil, chat.Options{AddGenerationPrompt: true})
	chatToks, err := m.Tokenize(prompt, false, true)
	if err != nil {
		return res, err
	}
	runOn := func(promptToks []backend.Token, plainKey, draftKey string, noDraft bool) (ttft, tps float64, acc *float64, out []backend.Token, err error) {
		var ttfts, rates, accs []float64
		for range reps {
			st, toks, err := gen(promptToks, ChatTokens, true, noDraft)
			if err != nil {
				return 0, 0, nil, nil, err
			}
			ttfts = append(ttfts, ms(st.Prefill))
			rates = append(rates, tokPerSec(len(toks)-1, st.Decode))
			if st.Drafted > 0 {
				accs = append(accs, float64(st.Accepted)/float64(st.Drafted))
			}
			out = toks
		}
		var ttftSD, tpsSD float64
		ttft, ttftSD = medianStd(ttfts)
		tps, tpsSD = medianStd(rates)
		key := plainKey
		if !noDraft {
			key = draftKey
		}
		std[key] = tpsSD
		if noDraft {
			std["ttft_ms"] = ttftSD
		}
		if len(accs) > 0 {
			a, _ := medianStd(accs)
			acc = &a
		}
		return ttft, tps, acc, out, nil
	}
	run := func(noDraft bool) (ttft, tps float64, acc *float64, out []backend.Token, err error) {
		return runOn(chatToks, "chat_tok_s", "mtp_tok_s", noDraft)
	}
	ttft, plain, _, out, err := run(true)
	if err != nil {
		return res, fmt.Errorf("chat-ru: %w", err)
	}
	res.Metrics.TTFTMS, res.Metrics.ChatTokS = ttft, plain
	res.Output = fingerprint(out)
	if caps.MaxDraft > 0 {
		_, drafted, acc, _, err := run(false)
		if err != nil {
			return res, fmt.Errorf("chat-ru with MTP: %w", err)
		}
		mtp := &benchfmt.MTP{}
		if acc != nil {
			mtp.Acceptance = *acc
		}
		mtp.TokS = drafted
		if plain > 0 {
			mtp.Speedup = drafted / plain
		}
		res.Metrics.MTP = mtp
	}
	if spec.Echo && caps.MaxDraft > 0 {
		echoToks, err := m.Tokenize(chat.PromptGemma4([]core.Message{{Role: core.RoleUser, Content: echoPrompt + echoText}}, nil, chat.Options{AddGenerationPrompt: true}), false, true)
		if err != nil {
			return res, err
		}
		_, plainEcho, _, plainOut, err := runOn(echoToks, "echo_tok_s", "echo_draft_tok_s", true)
		if err != nil {
			return res, fmt.Errorf("echo: %w", err)
		}
		_, draftEcho, acc, draftOut, err := runOn(echoToks, "echo_tok_s", "echo_draft_tok_s", false)
		if err != nil {
			return res, fmt.Errorf("echo with drafting: %w", err)
		}
		e := &benchfmt.MTP{TokS: draftEcho}
		if acc != nil {
			e.Acceptance = *acc
		}
		if plainEcho > 0 {
			e.Speedup = draftEcho / plainEcho
		}
		res.Metrics.Echo = e
		// Greedy decoding: drafting must not change a token.
		if !slices.Equal(plainOut, draftOut) {
			return res, fmt.Errorf("echo: drafting changed the output (%d tokens plain, %d drafted)", len(plainOut), len(draftOut))
		}
	}
	res.Metrics.StdDev = std
	return withMemory(m, res), nil
}

// withMemory adds the process's peaks and the weight buffers to a result.
func withMemory(m backend.Model, res WorkerResult) WorkerResult {
	res.Metrics.PeakRSS, res.Metrics.PeakLocked = peakMemory()
	for _, d := range m.MemoryReport().Devices {
		if d.Device != "cpu" && !strings.HasSuffix(d.Device, ":cpu") {
			res.Metrics.PeakVRAM += d.Weights + d.KV + d.Compute
		}
		for buft, n := range d.WeightBuffers {
			if res.Metrics.Buffers == nil {
				res.Metrics.Buffers = map[string]int64{}
			}
			res.Metrics.Buffers[buft] += n
		}
	}
	return res
}

// banEOG keeps every run to its full length: an end-of-generation token
// would end one configuration's answer earlier than another's.
func banEOG(m backend.Model, nvocab int) map[backend.Token]float64 {
	bias := map[backend.Token]float64{}
	for t := range nvocab {
		if m.IsEOG(backend.Token(t)) {
			bias[backend.Token(t)] = math.Inf(-1)
		}
	}
	return bias
}

func repeatTokens(m backend.Model, text string, n int) ([]backend.Token, error) {
	unit, err := m.Tokenize(text, false, false)
	if err != nil || len(unit) == 0 {
		return nil, errors.Join(err, errors.New("bench: the filler text tokenized to nothing"))
	}
	bos, _ := m.Tokenize("", true, false)
	out := append([]backend.Token(nil), bos...)
	for len(out) < n {
		out = append(out, unit...)
	}
	return out[:n], nil
}

func fingerprint(toks []backend.Token) benchfmt.Output {
	h := sha256.New()
	var b [4]byte
	for _, t := range toks {
		binary.LittleEndian.PutUint32(b[:], uint32(t))
		h.Write(b[:])
	}
	return benchfmt.Output{SHA256: hex.EncodeToString(h.Sum(nil)), Tokens: len(toks)}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// medianStd is the median and the sample standard deviation.
func medianStd(xs []float64) (float64, float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	med := s[len(s)/2]
	if len(s)%2 == 0 {
		med = (s[len(s)/2-1] + s[len(s)/2]) / 2
	}
	if len(s) < 2 {
		return med, 0
	}
	var mean, ss float64
	for _, x := range s {
		mean += x
	}
	mean /= float64(len(s))
	for _, x := range s {
		ss += (x - mean) * (x - mean)
	}
	return med, math.Sqrt(ss / float64(len(s)-1))
}
