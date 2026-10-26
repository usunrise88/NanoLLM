package tune

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/memlock"
	"github.com/usunrise88/nanollm/internal/procmem"
	"github.com/usunrise88/nanollm/internal/sched"
)

// A probe is one load of the model and the short measurements made on it
// (docs/OPTIMIZE.md, §3.3). It runs in a worker process, `nanollm optimize
// worker`: the process has its own peak memory, and a crash fails the candidate,
// not the server.

// DraftVariant is one drafting setting to measure on the loaded model.
type DraftVariant struct {
	Name string `json:"name"`
	// DraftMax is the most tokens drafted per step; 0 runs without drafting.
	DraftMax int     `json:"draft_max"`
	PMin     float64 `json:"p_min,omitempty"`
}

// ThreadVariant is one CPU team to measure.
type ThreadVariant struct {
	Name         string `json:"name"`
	Threads      int    `json:"threads"`
	ThreadsBatch int    `json:"threads_batch"`
}

// ProbeSpec is what a worker is asked to measure.
type ProbeSpec struct {
	Backend string           `json:"backend"`
	LibDir  string           `json:"lib_dir"`
	Load    backend.LoadSpec `json:"load"`
	// Locks, DrafterLocks and Random are the plan's range-lock and random-read
	// ranges, pinned or advised as the server does before the load.
	Locks        []memlock.Range   `json:"locks,omitempty"`
	DrafterLocks []memlock.Range   `json:"drafter_locks,omitempty"`
	Random       []memlock.Range   `json:"random,omitempty"`
	Env          map[string]string `json:"env,omitempty"`
	// LowPriority puts the worker below normal priority first (the quiet
	// mode), so that what it measures is what the quiet server will do.
	LowPriority bool `json:"low_priority,omitempty"`

	// Variants are the drafting settings measured one after another on the
	// loaded model, each on the same prompt out of the cache. The first one
	// whose name is StaticVariant is what the server runs without a profile.
	Variants      []DraftVariant `json:"variants"`
	StaticVariant string         `json:"static_variant,omitempty"`
	// Mechanism adds a variant chosen by the cost of a batch (BestDraftMax):
	// for a device whose drafting nobody has measured, the probe times batches
	// of 1, 3 and 4 tokens first and measures the draft length they favour, when
	// one beats decoding by the margin. Kind is the device kind (cuda, vulkan,
	// cpu) for the cost of a draft round.
	Mechanism bool   `json:"mechanism,omitempty"`
	Kind      string `json:"kind,omitempty"`
	// Threads are CPU teams measured with the best drafting setting.
	Threads []ThreadVariant `json:"threads,omitempty"`
	// BatchSizes are the batch sizes whose cost is timed (the mechanism probe
	// of MTP, docs/OPTIMIZE.md §3.8).
	BatchSizes []int `json:"batch_sizes,omitempty"`
	// Verify runs the check that drafting leaves the answer unchanged.
	Verify bool `json:"verify,omitempty"`
	// Slow shortens the probe for a setup that streams its weights from a disk:
	// SlowDecodeTokens of decode, one variant.
	Slow bool `json:"slow,omitempty"`
	// Reuse says the cache of a slot can be cut back to a shared prefix (the
	// plan keeps the full sliding-window cache), so drafting variants can all
	// start from the one prefilled prompt. Without it they decode from the short
	// prompt.
	Reuse bool `json:"reuse,omitempty"`
	// PromptTokens is the length of the reference prompt (RefPromptTokens, or
	// less when the context is small).
	PromptTokens int `json:"prompt_tokens"`
	// DepthPP and DepthTG carry a probe made on the short prompt over to the
	// reference depth (Baseline.DepthPP and DepthTG).
	DepthPP float64 `json:"depth_pp"`
	DepthTG float64 `json:"depth_tg"`
	// Calibrate makes the probe measure what the matrix's baselines hold, with
	// no shortcuts: the whole reference prompt whatever it costs, decode at an
	// empty context and after it, every variant at both depths
	// (`nanollm optimize calibrate`).
	Calibrate bool `json:"calibrate,omitempty"`
	// Budget is how long the probe may take after the load; what is not done by
	// then is skipped and named in the result.
	Budget time.Duration `json:"budget"`
}

// Probe lengths.
const (
	// ShortPrefix is the prompt prefix whose prefill times the rest.
	ShortPrefix = 512
	// FullPrefillLimit is the longest forecast prefill of the whole reference
	// prompt the probe is willing to wait for; beyond it the prompt stays short
	// and the numbers are carried over (Extrapolated).
	FullPrefillLimit = 25 * time.Second
	DecodeTokens     = 48
	MaxDecodeTokens  = 128
	SlowDecodeTokens = 8
	VerifyPromptLen  = 64
	VerifyTokens     = 32
	// NearTieNats is how close the two logprobs may be where drafted and plain
	// decoding part ways for the difference to count as numerics: a batch of
	// verified tokens and a batch of one may run different kernels (CUDA sums in
	// another order), and where the model hesitates between two words the greedy
	// pick can go either way. A fault shows as a token the plain run gave little
	// probability. The value is a judgement, not a measurement: on the shipped
	// CUDA kernels the native test finds no parting at all.
	NearTieNats = 0.1
	// verifyAlternatives is how many alternatives the plain run reports at each
	// token: the drafted run's token has to be among them to be a near tie.
	verifyAlternatives = 5
	WarmupTokens       = 16
)

// MemoryResult is what the process held at the end of the probe.
type MemoryResult struct {
	// PeakPrivate is the highest private resident memory seen, sampled every
	// 100 ms, across the load and the probes.
	PeakPrivate int64 `json:"peak_private"`
	PeakCommit  int64 `json:"peak_commit,omitempty"`
	// VRAMDedicated and VRAMShared are the process's GPU memory (procmem), and
	// VRAMSource says where from.
	VRAMDedicated int64  `json:"vram_dedicated,omitempty"`
	VRAMShared    int64  `json:"vram_shared,omitempty"`
	VRAMSource    string `json:"vram_source,omitempty"`
	// Buffers is the sum of the engine's own buffers on devices, the fallback
	// when no counter reads the process.
	Buffers int64 `json:"buffers,omitempty"`
}

// PrefillResult is the prefill of the reference prompt.
type PrefillResult struct {
	Tokens int `json:"tokens"`
	// PP512 is the rate over the first ShortPrefix tokens, PP the rate over the
	// whole prompt at the reference depth.
	PP512 float64 `json:"pp512"`
	PP    float64 `json:"pp"`
	// Extrapolated says PP is carried over from the short prefix by DepthPP.
	Extrapolated bool `json:"extrapolated,omitempty"`
}

// VariantResult is one drafting setting at the reference depth.
type VariantResult struct {
	Name     string  `json:"name"`
	DraftMax int     `json:"draft_max"`
	PMin     float64 `json:"p_min,omitempty"`
	// TokS is decode tokens per second at the reference depth: measured after
	// the full prompt, or after the short one and carried over (Extrapolated).
	TokS         float64 `json:"tok_s"`
	Extrapolated bool    `json:"extrapolated,omitempty"`
	Tokens       int     `json:"tokens"`
	Drafted      int     `json:"drafted,omitempty"`
	Accepted     int     `json:"accepted,omitempty"`
	// Tried[j] and AcceptedAt[j] are the steps that proposed a draft at
	// position j+1 after accepting the j before it, and those that were
	// accepted (sched.Stats.DraftTried and AcceptedRuns).
	Tried      []int `json:"tried,omitempty"`
	AcceptedAt []int `json:"accepted_at,omitempty"`
	// Threads is the team the variant ran with (decode, prompt).
	Threads      int `json:"threads,omitempty"`
	ThreadsBatch int `json:"threads_batch,omitempty"`
}

// Calibration is a device's speed in the terms of a matrix baseline: prefill
// at the start of a prompt and averaged over the reference prompt, decode at an
// empty context and after the prompt, with the best drafting setting.
type Calibration struct {
	PP512  float64 `json:"pp512"`
	PP4096 float64 `json:"pp4096"`
	TG0    float64 `json:"tg0"`
	TG4096 float64 `json:"tg4096"`
	// ShortTG4096 is the decode rate after the short prefix, the figure a probe
	// carries over to the reference depth with DepthTG.
	ShortTG float64 `json:"short_tg"`
	// Depth factors: PP4096/PP512 and TG4096/TG0.
	DepthPP float64 `json:"depth_pp"`
	DepthTG float64 `json:"depth_tg"`
	// Variants are the drafting settings at both depths: name to {0, 4096} rates.
	Variants map[string][2]float64 `json:"variants,omitempty"`
}

// VerifyResult is the correctness check of drafting.
type VerifyResult struct {
	Checked bool `json:"checked"`
	Match   bool `json:"match"`
	// At is the index of the first token that differs, -1 when they match.
	At     int `json:"at"`
	Tokens int `json:"tokens"`
	// NearTie says the tokens part at At but the model gave both almost the same
	// probability (Gap nats apart): numerics, not a fault, and Match is true.
	NearTie bool    `json:"near_tie,omitempty"`
	Gap     float64 `json:"gap,omitempty"`
}

// GPUState is a GPU's condition at the start or the end of a probe.
type GPUState struct {
	TempC    int      `json:"temp_c"`
	ClockMHz int      `json:"clock_mhz,omitempty"`
	Throttle []string `json:"throttle,omitempty"`
}

// ProbeResult is what a worker reports.
type ProbeResult struct {
	LoadMS   float64         `json:"load_ms"`
	Memory   MemoryResult    `json:"memory"`
	Prefill  PrefillResult   `json:"prefill"`
	Variants []VariantResult `json:"variants,omitempty"`
	// ThreadVariants are the team variants (with the best drafting setting).
	ThreadVariants []VariantResult `json:"thread_variants,omitempty"`
	// Calibration is set by a calibrating probe.
	Calibration *Calibration `json:"calibration,omitempty"`
	// BatchMS is the median time in milliseconds of a batch of k tokens, by k.
	BatchMS  map[int]float64 `json:"batch_ms,omitempty"`
	Verify   *VerifyResult   `json:"verify,omitempty"`
	GPUBegin *GPUState       `json:"gpu_begin,omitempty"`
	GPUEnd   *GPUState       `json:"gpu_end,omitempty"`
	// Skipped names what the budget cut off.
	Skipped  []string `json:"skipped,omitempty"`
	ElapsedS float64  `json:"elapsed_s"`
	Error    string   `json:"error,omitempty"`
}

// Best is the fastest drafting variant by decode rate; false with none.
func (r *ProbeResult) Best() (VariantResult, bool) {
	var best VariantResult
	ok := false
	for _, v := range r.Variants {
		if !ok || v.TokS > best.TokS {
			best, ok = v, true
		}
	}
	return best, ok
}

// Variant finds a drafting variant by name.
func (r *ProbeResult) Variant(name string) (VariantResult, bool) {
	for _, v := range r.Variants {
		if v.Name == name {
			return v, true
		}
	}
	return VariantResult{}, false
}

// RunProbe loads the model with spec.Load on b and measures it. It is the body
// of the worker; tests call it with a scripted backend. Partial results come
// back with an error.
func RunProbe(ctx context.Context, b backend.Backend, spec ProbeSpec) (ProbeResult, error) {
	var res ProbeResult
	began := time.Now()
	defer func() { res.ElapsedS = time.Since(began).Seconds() }()
	meter := procmem.Start()
	defer meter.Stop()
	res.GPUBegin = gpuState(ctx, spec.Load)

	for path, ranges := range map[string][]memlock.Range{spec.Load.Path: spec.Locks, spec.Load.DrafterPath: spec.DrafterLocks} {
		if len(ranges) == 0 || path == "" || spec.Backend == "fake" {
			continue // the scripted backend has no files to pin
		}
		pin, err := memlock.PinFile(path, ranges)
		if err != nil {
			return res, fmt.Errorf("range-lock: %w", err)
		}
		defer func() { _ = pin.Close() }()
	}
	loadStart := time.Now()
	m, err := b.Load(ctx, spec.Load)
	if err != nil {
		return res, err
	}
	defer func() { _ = m.Close() }()
	if n, err := memlock.AdviseRandom(spec.Load.Path, spec.Random); err == nil && n > 0 {
		fmt.Fprintf(os.Stderr, "madvise random: %d MiB\n", n>>20)
	}
	res.LoadMS = float64(time.Since(loadStart).Microseconds()) / 1000

	loop := sched.New(m)
	defer loop.Close()
	p := &prober{ctx: ctx, m: m, loop: loop, spec: spec, res: &res, deadline: time.Now().Add(spec.Budget)}
	err = p.run()
	// Memory is read with the model still loaded, before the loop stops.
	res.Memory = readMemory(m, meter)
	res.GPUEnd = gpuState(ctx, spec.Load)
	return res, err
}

type prober struct {
	ctx      context.Context
	m        backend.Model
	loop     *sched.Loop
	spec     ProbeSpec
	res      *ProbeResult
	deadline time.Time
	greedy   backend.SamplerSpec
	prompt   []backend.Token // the reference prompt
	short    []backend.Token // its first ShortPrefix tokens
	// cached is the length of the prompt the slot holds, so that a variant
	// knows what it starts from.
	cached int
}

func (p *prober) timeLeft() time.Duration { return time.Until(p.deadline) }

func (p *prober) skip(what string) { p.res.Skipped = append(p.res.Skipped, what) }

// gen runs one request through the loop and returns its stats and tokens.
func (p *prober) gen(prompt []backend.Token, n int, noCache, noDraft bool) (*sched.Stats, []backend.Token, error) {
	st, toks, _, err := p.genLogprobs(prompt, n, noCache, noDraft, -1)
	return st, toks, err
}

// genLogprobs is gen with the alternatives of every token (-1 for none).
func (p *prober) genLogprobs(prompt []backend.Token, n int, noCache, noDraft bool, logprobs int) (*sched.Stats, []backend.Token, []backend.TokenLogprobs, error) {
	h, err := p.loop.Submit(p.ctx, sched.Request{Prompt: prompt, Sampler: p.greedy, MaxTokens: n, Logprobs: logprobs, NoCache: noCache, NoDraft: noDraft})
	if err != nil {
		return nil, nil, nil, err
	}
	var toks []backend.Token
	var lps []backend.TokenLogprobs
	for {
		ev, err := h.Next(p.ctx)
		if err != nil {
			return nil, nil, nil, err
		}
		toks = append(toks, ev.Tokens...)
		lps = append(lps, ev.Logprobs...)
		if ev.Done {
			if ev.Err != nil {
				return nil, nil, nil, ev.Err
			}
			return ev.Stats, toks, lps, nil
		}
	}
}

func rate(n int, d time.Duration) float64 {
	if n <= 0 || d <= 0 {
		return 0
	}
	return float64(n) / d.Seconds()
}

func (p *prober) run() error {
	caps := p.m.Capabilities()
	p.greedy = backend.SamplerSpec{Temperature: 0, TopK: 1, TopP: 1, RepeatPenalty: 1, Seed: 1, LogitBias: banEOG(p.m, caps.NVocab)}
	n := p.spec.PromptTokens
	if n <= 0 {
		n = RefPromptTokens
	}
	n = min(n, caps.NCtx-DecodeTokens-8)
	if n <= ShortPrefix {
		return fmt.Errorf("tune: a context of %d tokens is too small for the reference prompt", caps.NCtx)
	}
	var err error
	if p.prompt, err = RefPrompt(p.m, n); err != nil {
		return err
	}
	p.short = p.prompt[:ShortPrefix]

	// 1. Warm-up: the first requests pay for graph capture, caches and clocks.
	if _, _, err := p.gen(p.prompt[:32], WarmupTokens, true, true); err != nil {
		return fmt.Errorf("warm-up: %w", err)
	}
	if p.spec.Calibrate {
		return p.calibrate(n)
	}

	// 2-3. Prefill: the short prefix first, then the whole prompt if the
	// forecast says it fits the time.
	if err := p.prefill(n); err != nil {
		return err
	}

	// 4. Decode with each drafting variant.
	tokens := DecodeTokens
	if p.spec.Slow {
		tokens = SlowDecodeTokens
	}
	variants := append([]DraftVariant(nil), p.spec.Variants...)
	if p.spec.Slow && len(variants) > 1 {
		variants = variants[:1]
	}
	if p.spec.Mechanism && !p.spec.Slow {
		if err := p.batchCost(); err != nil {
			p.skip("batch cost: " + err.Error())
		} else if t, ok := p.m.(backend.Tuner); ok {
			k, _ := BestDraftMax(p.res.BatchMS, DefaultAcceptance, DraftCostFraction(p.spec.Kind), t.DraftCeiling())
			if k > 0 && !hasDraft(variants, k) {
				variants = append(variants, DraftVariant{Name: fmt.Sprintf("mech%d", k), DraftMax: k, PMin: 0.5})
			}
		}
	}
	for _, v := range variants {
		if p.timeLeft() <= 0 {
			p.skip("variant " + v.Name)
			continue
		}
		vr, err := p.decodeVariant(v, tokens)
		if err != nil {
			return fmt.Errorf("variant %s: %w", v.Name, err)
		}
		p.res.Variants = append(p.res.Variants, vr)
		if v.DraftMax == 0 && !p.spec.Slow {
			// A fast device decodes a longer stretch in the same time, and a longer
			// stretch averages out how well the drafts happen to match the text.
			tokens = sampleTokens(vr.TokS)
		}
	}
	if p.spec.Slow {
		return nil
	}

	// 5. The mechanism probe: what a batch of k tokens costs on the loaded
	// model. It runs at the end of the cache the last variant left.
	if len(p.spec.BatchSizes) > 0 && p.res.BatchMS == nil {
		if p.timeLeft() <= 0 {
			p.skip("batch cost")
		} else if err := p.batchCost(); err != nil {
			p.skip("batch cost: " + err.Error())
		}
	}

	// 6. CPU team variants, with the best drafting setting so far.
	if best, ok := p.res.Best(); ok {
		for _, tv := range p.spec.Threads {
			if p.timeLeft() <= 0 {
				p.skip("threads " + tv.Name)
				continue
			}
			vr, err := p.decodeThreads(best, tv, tokens)
			if err != nil {
				p.skip("threads " + tv.Name + ": " + err.Error())
				continue
			}
			p.res.ThreadVariants = append(p.res.ThreadVariants, vr)
		}
	}

	// 7. Drafting must not change the answer.
	if p.spec.Verify {
		if p.timeLeft() <= 0 {
			p.skip("verify")
		} else if v, err := p.verify(); err != nil {
			p.skip("verify: " + err.Error())
		} else {
			p.res.Verify = v
		}
	}
	return nil
}

// prefill times the short prefix and, when it is not too long, the whole prompt.
func (p *prober) prefill(n int) error {
	t512, _, err := p.gen(p.short, 1, true, true)
	if err != nil {
		return fmt.Errorf("prefill: %w", err)
	}
	pr := &p.res.Prefill
	pr.Tokens = n
	pr.PP512 = rate(ShortPrefix, t512.Prefill)
	depth := p.spec.DepthPP
	if depth <= 0 {
		depth = 0.85
	}
	forecast := time.Duration(float64(n) / (pr.PP512 * depth) * float64(time.Second))
	if forecast > FullPrefillLimit || (!p.spec.Reuse && forecast > 5*time.Second) {
		// The prompt stays short: the numbers are carried over to the depth of
		// the reference request.
		pr.PP, pr.Extrapolated = pr.PP512*depth, true
		p.cached = ShortPrefix
		return nil
	}
	full, _, err := p.gen(p.prompt, 1, false, true)
	if err != nil {
		return fmt.Errorf("prefill: %w", err)
	}
	// The first ShortPrefix tokens came from the cache when the slot could be
	// cut back to them; otherwise the whole prompt was computed here.
	total := full.Prefill
	computed := n - full.CachedTokens
	if full.CachedTokens >= ShortPrefix {
		total += t512.Prefill
		computed = n
	}
	_ = computed
	pr.PP = rate(n, total)
	p.cached = n
	return nil
}

// decodeVariant decodes tokens with one drafting setting.
func (p *prober) decodeVariant(v DraftVariant, tokens int) (VariantResult, error) {
	vr := VariantResult{Name: v.Name, DraftMax: v.DraftMax, PMin: v.PMin}
	if t, ok := p.m.(backend.Tuner); ok {
		max := v.DraftMax
		if max > t.DraftCeiling() {
			max = t.DraftCeiling()
		}
		if p.m.Capabilities().MaxDraft > 0 || t.DraftCeiling() > 0 {
			if err := p.loop.Do(p.ctx, func(backend.Model) error { return t.SetDraft(max, v.PMin) }); err != nil {
				return vr, err
			}
		}
	}
	prompt, extrapolated := p.decodePrompt()
	st, toks, err := p.gen(prompt, tokens, false, v.DraftMax == 0)
	if err != nil {
		return vr, err
	}
	vr.Tokens = len(toks)
	vr.TokS = rate(len(toks)-1, st.Decode)
	vr.Drafted, vr.Accepted = st.Drafted, st.Accepted
	vr.Tried, vr.AcceptedAt = slices.Clone(st.DraftTried), slices.Clone(st.AcceptedRuns)
	if extrapolated {
		depth := p.spec.DepthTG
		if depth <= 0 {
			depth = 0.66
		}
		vr.TokS *= depth
		vr.Extrapolated = true
	}
	return vr, nil
}

// decodePrompt is the prompt the decode variants start from: the whole
// reference prompt when the prefill ran over all of it (out of the cache, or
// rebuilt for each variant where the slot cannot be cut back, which prefill
// allowed only when that is quick), the short one when it was carried over.
func (p *prober) decodePrompt() (prompt []backend.Token, extrapolated bool) {
	if p.cached >= len(p.prompt) {
		return p.prompt, false
	}
	return p.short, true
}

// decodeThreads decodes with another CPU team.
func (p *prober) decodeThreads(best VariantResult, tv ThreadVariant, tokens int) (VariantResult, error) {
	t, ok := p.m.(backend.Tuner)
	if !ok {
		return VariantResult{}, errors.New("the backend cannot change threads on a loaded model")
	}
	if err := p.loop.Do(p.ctx, func(backend.Model) error { return t.SetThreads(tv.Threads, tv.ThreadsBatch) }); err != nil {
		return VariantResult{}, err
	}
	vr, err := p.decodeVariant(DraftVariant{Name: best.Name, DraftMax: best.DraftMax, PMin: best.PMin}, tokens)
	vr.Name = tv.Name
	vr.Threads, vr.ThreadsBatch = tv.Threads, tv.ThreadsBatch
	return vr, err
}

// batchCost times a batch of k tokens for each k, five times, on the loaded
// model: the median is the cost c(k) of verifying k-1 draft tokens.
func (p *prober) batchCost() error {
	bt, ok := p.m.(backend.BatchTimer)
	if !ok {
		return errors.New("the backend cannot time a batch")
	}
	maxK := slices.Max(p.spec.BatchSizes)
	filler := make([]backend.Token, maxK)
	for i := range filler {
		filler[i] = p.prompt[len(p.prompt)-1-i%max(len(p.prompt), 1)]
	}
	p.res.BatchMS = map[int]float64{}
	for _, k := range p.spec.BatchSizes {
		var ds []float64
		for range 5 {
			var d time.Duration
			err := p.loop.Do(p.ctx, func(backend.Model) error {
				var err error
				d, err = bt.TimeBatch(p.ctx, 0, filler[:k], k)
				return err
			})
			if err != nil {
				return err
			}
			ds = append(ds, float64(d.Microseconds())/1000)
		}
		sort.Float64s(ds)
		p.res.BatchMS[k] = ds[len(ds)/2]
	}
	return nil
}

// verify decodes a short prompt with the best drafting setting and without,
// and compares the tokens: they must be the same.
func (p *prober) verify() (*VerifyResult, error) {
	short, err := ShortPrompt(p.m)
	if err != nil {
		return nil, err
	}
	best, ok := p.res.Best()
	if !ok || best.DraftMax == 0 {
		return &VerifyResult{Checked: false, Match: true, At: -1}, nil
	}
	if t, ok := p.m.(backend.Tuner); ok {
		if err := p.loop.Do(p.ctx, func(backend.Model) error { return t.SetDraft(min(best.DraftMax, t.DraftCeiling()), best.PMin) }); err != nil {
			return nil, err
		}
	}
	_, with, err := p.gen(short, VerifyTokens, true, false)
	if err != nil {
		return nil, err
	}
	_, without, lps, err := p.genLogprobs(short, VerifyTokens, true, true, verifyAlternatives)
	if err != nil {
		return nil, err
	}
	v := CompareDrafts(with, without, lps)
	return &v, nil
}

// CompareDrafts compares a drafted generation with a plain one (and the plain
// one's alternatives at each token). Equal tokens match; tokens that part where
// the plain run saw the drafted run's token almost as likely as its own (within
// NearTieNats) match too, as a near tie; anything else is a mismatch.
func CompareDrafts(with, without []backend.Token, plain []backend.TokenLogprobs) VerifyResult {
	v := VerifyResult{Checked: true, Match: true, At: -1, Tokens: len(without)}
	if slices.Equal(with, without) {
		return v
	}
	v.At = min(len(with), len(without))
	for i := range min(len(with), len(without)) {
		if with[i] != without[i] {
			v.At = i
			break
		}
	}
	v.Match = false
	if v.At >= len(with) || v.At >= len(without) || v.At >= len(plain) {
		return v
	}
	lp := plain[v.At]
	for _, alt := range lp.Top {
		if alt.Token == with[v.At] {
			if gap := lp.Logprob - alt.Logprob; gap >= 0 && gap <= NearTieNats {
				v.Match, v.NearTie, v.Gap = true, true, gap
			}
			break
		}
	}
	return v
}

// banEOG keeps every probe to its full length: an end-of-generation token
// would end one setting's answer earlier than another's.
func banEOG(m backend.Model, nvocab int) map[backend.Token]float64 {
	bias := map[backend.Token]float64{}
	for t := range nvocab {
		if m.IsEOG(backend.Token(t)) {
			bias[backend.Token(t)] = math.Inf(-1)
		}
	}
	return bias
}

// readMemory reads what the process holds: the private peak, and the video
// memory by the best source there is.
func readMemory(m backend.Model, meter *procmem.Meter) MemoryResult {
	mr := MemoryResult{PeakPrivate: meter.Peak(), PeakCommit: procmem.PeakCommit()}
	if v, err := procmem.ProcessVRAM(os.Getpid()); err == nil {
		mr.VRAMDedicated, mr.VRAMShared, mr.VRAMSource = v.Dedicated, v.Shared, v.Source
	}
	for _, d := range m.MemoryReport().Devices {
		if d.Device != "cpu" {
			mr.Buffers += d.Weights + d.KV + d.Compute
		}
	}
	return mr
}

// gpuState reads the temperature and clocks of the first NVIDIA GPU when the
// load is on a GPU.
func gpuState(ctx context.Context, l backend.LoadSpec) *GPUState {
	if !l.FullyOnGPU() && l.GPULayers == 0 {
		return nil
	}
	live := hwinfo.ReadLive(ctx, 20*time.Millisecond)
	if len(live.GPUs) == 0 {
		return nil
	}
	g := live.GPUs[0]
	return &GPUState{TempC: g.TempC, ClockMHz: g.SMClockMHz, Throttle: g.Throttle}
}

// hasDraft reports whether a variant with this draft length exists.
func hasDraft(vs []DraftVariant, k int) bool {
	for _, v := range vs {
		if v.DraftMax == k {
			return true
		}
	}
	return false
}

// calibrate measures a device the long way, for the matrix's baselines.
func (p *prober) calibrate(n int) error {
	cal := &Calibration{Variants: map[string][2]float64{}}
	p.res.Calibration = cal
	set := func(v DraftVariant) error {
		t, ok := p.m.(backend.Tuner)
		if !ok || t.DraftCeiling() == 0 {
			return nil
		}
		return p.loop.Do(p.ctx, func(backend.Model) error { return t.SetDraft(min(v.DraftMax, t.DraftCeiling()), v.PMin) })
	}
	decode := func(prompt []backend.Token, v DraftVariant) (float64, error) {
		if err := set(v); err != nil {
			return 0, err
		}
		st, toks, err := p.gen(prompt, 64, false, v.DraftMax == 0)
		if err != nil {
			return 0, err
		}
		return rate(len(toks)-1, st.Decode), nil
	}
	// Prefill: the start of the prompt, then all of it.
	t512, _, err := p.gen(p.short, 1, true, true)
	if err != nil {
		return err
	}
	cal.PP512 = rate(ShortPrefix, t512.Prefill)
	full, _, err := p.gen(p.prompt, 1, true, true)
	if err != nil {
		return err
	}
	cal.PP4096 = rate(n, full.Prefill)
	p.res.Prefill = PrefillResult{Tokens: n, PP512: cal.PP512, PP: cal.PP4096}
	// Decode at an empty context (a short prompt) and after the whole prompt,
	// for the plain setting and each drafting variant.
	empty := p.prompt[:16]
	off := DraftVariant{Name: VariantOff}
	var err2 error
	if cal.TG0, err2 = decode(empty, off); err2 != nil {
		return err2
	}
	if cal.ShortTG, err2 = decode(p.short, off); err2 != nil {
		return err2
	}
	if cal.TG4096, err2 = decode(p.prompt, off); err2 != nil {
		return err2
	}
	cal.Variants[VariantOff] = [2]float64{cal.TG0, cal.TG4096}
	for _, v := range p.spec.Variants {
		if v.DraftMax == 0 {
			continue
		}
		d0, err := decode(empty, v)
		if err != nil {
			return err
		}
		d4, err := decode(p.prompt, v)
		if err != nil {
			return err
		}
		cal.Variants[v.Name] = [2]float64{d0, d4}
	}
	if cal.PP512 > 0 {
		cal.DepthPP = cal.PP4096 / cal.PP512
	}
	if cal.TG0 > 0 {
		cal.DepthTG = cal.TG4096 / cal.TG0
	}
	if len(p.spec.BatchSizes) > 0 {
		if err := p.batchCost(); err != nil {
			p.skip("batch cost: " + err.Error())
		}
	}
	return nil
}

// sampleTokens is how many tokens the drafting variants decode, given the
// plain rate: about a second and a half of decoding at the plain rate, between
// DecodeTokens and MaxDecodeTokens.
func sampleTokens(plainTokS float64) int {
	return min(max(int(plainTokS*1.5), DecodeTokens), MaxDecodeTokens)
}
