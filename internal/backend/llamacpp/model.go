package llamacpp

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/memlock"
	"github.com/usunrise88/nanollm/internal/ngram"
)

// loadMu serializes model loads, so the log lines a load produces (buffer
// sizes, KV cells) are attributed to that load.
var loadMu sync.Mutex

type model struct {
	// mu guards the handles against Close: the tokenizer methods take it
	// shared from any goroutine, the slot methods from the model's loop.
	mu sync.RWMutex

	b     *llamaBackend
	spec  backend.LoadSpec
	mdl   LlamaModel
	ctx   LlamaContext
	vocab LlamaVocab
	mem   LlamaMemoryI
	batch LlamaBatchExt

	nVocab    int
	nSWA      int
	swaFull   bool
	recurrent bool
	caps      backend.Capabilities
	meta      map[string]string
	memory    backend.MemoryReport
	// loadLog is what llama.cpp said while loading this model and creating
	// its context, for tests and diagnostics.
	loadLog []string

	// nOutputsMax is the context's n_outputs_max: llama.cpp asserts that a
	// batch asks for no more output rows, so Step counts them first.
	nOutputsMax int

	// batchThreads is the n_threads_batch the context currently has, which
	// tuneThreads moves between the decode and the prefill team.
	batchThreads int32

	// draftCeiling is the most draft tokens per step the context has output
	// rows for: the DraftMax it was created with (backend.Tuner.SetDraft may
	// lower the limit, never raise it).
	draftCeiling int

	// dft is the MTP drafter, nil without one.
	dft *drafter
	// ngN is the match length of prompt-lookup drafting (internal/ngram), 0
	// when off; it drafts only where no MTP drafter does.
	ngN int

	// stream feeds the file-backed layers (OPT-8); streamID is the key of
	// the eval callback's user_data. Both are zero without streaming.
	stream   *memlock.Streamer
	streamID uintptr

	slots []slot
}

// slot is one sequence's sampling state and its MTP carry-over.
type slot struct {
	chain     LlamaSampler // owned; frees the samplers added to it
	penalties LlamaSampler // inside chain, fed the history; 0 when unused
	spec      backend.SamplerSpec
	// onDevice: the chain is attached to the context and runs in its graph
	// (LoadSpec.BackendSampling); llama_sampler_sample returns the token the
	// graph picked, and no logits row reaches the host.
	onDevice bool

	// pendingH is the target's hidden state (h_nextn) at position
	// pendingPos: the input whose logits produced the token the caller
	// feeds next. Drafting needs exactly that pair — the next token at
	// pendingPos+1 with the state before it — so a step drafts only when
	// its position is pendingPos+1. Anything that moves the cache another
	// way (trim, clear, restore, a prompt-cache hit that prefills afresh)
	// leaves the positions out of step, and that step runs without drafts
	// while it computes a fresh state.
	pendingH   []float32
	pendingPos int

	// hist is the token at every position of the slot's KV cache, kept for
	// prompt-lookup drafting; histBad says it can no longer be trusted (a
	// restored state, a gap), until the slot is cleared. ng throttles a slot
	// whose drafts keep missing.
	hist    []backend.Token
	histBad bool
	ng      ngram.Backoff
}

func (b *llamaBackend) Load(ctx context.Context, spec backend.LoadSpec) (backend.Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	files := []string{spec.Path}
	if spec.DrafterPath != "" {
		files = append(files, spec.DrafterPath)
	}
	applyCudaNano(spec.CudaNano)
	for _, p := range files {
		if st, err := os.Stat(p); err != nil {
			return nil, fmt.Errorf("llamacpp: model: %w", err)
		} else if st.IsDir() {
			return nil, fmt.Errorf("llamacpp: model %s is a directory", p)
		}
	}
	mp, keep, err := b.modelParams(spec)
	if err != nil {
		return nil, err
	}
	cp, err := b.contextParams(spec)
	if err != nil {
		return nil, err
	}

	loadMu.Lock()
	defer loadMu.Unlock()
	spec = spec.WithDefaultThreads()
	m := &model{b: b, spec: spec, batchThreads: int32(spec.ThreadsBatch)}
	untap := addLogTap(func(_ GgmlLogLevel, line string) { m.loadLog = append(m.loadLog, line) })
	defer untap()

	m.mdl = llamaModelLoadFromFile(spec.Path, mp)
	runtime.KeepAlive(keep)
	if m.mdl == 0 {
		return nil, fmt.Errorf("llamacpp: load %s failed%s", spec.Path, lastErrors(m.loadLog))
	}
	if st := spec.Stream.WithEnv(); st.Active() {
		m.attachStream(st, &cp)
	}
	m.ctx = llamaInitFromModel(m.mdl, cp)
	if m.ctx == 0 {
		m.detachStream()
		llamaModelFree(m.mdl)
		return nil, fmt.Errorf("llamacpp: create a context for %s failed%s", spec.Path, lastErrors(m.loadLog))
	}
	m.vocab = llamaModelGetVocab(m.mdl)
	m.mem = llamaGetMemory(m.ctx)
	m.batch = llamaBatchExtInit(m.ctx)
	m.nVocab = int(llamaVocabNTokens(m.vocab))
	m.nSWA = int(llamaModelNSwa(m.mdl))
	m.swaFull = cp.SwaFull
	m.recurrent = llamaModelIsRecurrent(m.mdl) || llamaModelIsHybrid(m.mdl)
	m.caps = backend.Capabilities{
		NCtx:      int(llamaNCtxSeq(m.ctx)),
		Slots:     int(llamaNSeqMax(m.ctx)),
		NBatch:    int(llamaNBatch(m.ctx)),
		SWAFull:   m.swaFull,
		SWAWindow: m.nSWA,
		NVocab:    m.nVocab,
	}
	m.nOutputsMax = int(cp.NOutputsMax)
	if m.nOutputsMax == 0 {
		m.nOutputsMax = m.caps.NBatch
	}
	m.slots = make([]slot, m.caps.Slots)
	for i := range m.slots {
		m.slots[i].pendingPos = -1
	}
	if spec.DrafterPath != "" {
		d, err := m.loadDrafter(spec, mp, cp)
		if err != nil {
			llamaBatchExtFree(m.batch)
			llamaFree(m.ctx)
			llamaModelFree(m.mdl)
			return nil, err
		}
		m.dft = d
		m.caps.MaxDraft = d.maxDraft
	} else if spec.NGram > 0 && spec.DraftMax > 0 {
		m.ngN = spec.NGram
		m.caps.MaxDraft = min(spec.DraftMax, max(m.nOutputsMax/max(spec.Slots, 1)-1, 0))
	}
	m.draftCeiling = m.caps.MaxDraft
	m.meta = m.readMetadata()
	m.memory = memoryFromLog(m.loadLog, b)
	return m, nil
}

// modelParams turns the placement part of a spec into llama_model_params.
// The returned values must stay alive until the load returns: llama.cpp
// reads the device list and the override patterns during the call.
func (b *llamaBackend) modelParams(spec backend.LoadSpec) (LlamaModelParams, []any, error) {
	p := llamaModelDefaultParams()
	var keep []any

	dev, onCPU, err := b.device(spec.Device)
	if err != nil {
		return p, nil, err
	}
	// An explicit list stops llama.cpp from spreading the model over every
	// GPU it finds: the planner chose one device. An empty list is CPU only.
	devs := []GgmlBackendDevice{0}
	if !onCPU {
		devs = []GgmlBackendDevice{dev, 0}
	}
	p.Devices = &devs[0]
	keep = append(keep, devs)
	if !onCPU {
		// The whole model on that device. Without a split llama.cpp divides
		// the layers by the free memory each device reports, and a GPU that
		// reports none while its total is known (CUDA under WDDM once VRAM is
		// oversubscribed) makes the split points 0/0: the load then fails
		// with "invalid vector subscript" (devices.at past the end). llama.cpp
		// copies llama_max_devices() entries.
		split := make([]float32, llamaMaxDevices())
		split[0] = 1
		p.TensorSplit = &split[0]
		keep = append(keep, split)
	}
	switch {
	case onCPU:
		p.NGpuLayers = 0
	case spec.GPULayers < 0:
		p.NGpuLayers = -1
	default:
		p.NGpuLayers = int32(spec.GPULayers)
	}
	p.SplitMode = LlamaSplitModeNone

	switch spec.LoadMode {
	case backend.LoadAuto:
		p.LoadMode = LlamaLoadModeAuto
	case backend.LoadNone:
		p.LoadMode = LlamaLoadModeNone
	case backend.LoadMMap:
		p.LoadMode = LlamaLoadModeMmap
	case backend.LoadMLock:
		p.LoadMode = LlamaLoadModeMlock
	case backend.LoadMMapMLock:
		p.LoadMode = LlamaLoadModeMmapMlock
	case backend.LoadDirectIO:
		p.LoadMode = LlamaLoadModeDirectIo
	default:
		return p, nil, fmt.Errorf("llamacpp: unknown load mode %q", spec.LoadMode)
	}
	switch spec.LazyMode {
	case "":
		// engine default (auto)
	case backend.LazyOff:
		p.LazyMode = LlamaLazyModeOff
	case backend.LazyAuto:
		p.LazyMode = LlamaLazyModeAuto
	case backend.LazyOn:
		p.LazyMode = LlamaLazyModeOn
	default:
		return p, nil, fmt.Errorf("llamacpp: unknown lazy mode %q", spec.LazyMode)
	}
	// Lazy loading reads rows through the mapping, so it needs one.
	if spec.LazyMode == backend.LazyAuto || spec.LazyMode == backend.LazyOn {
		switch spec.LoadMode {
		case backend.LoadNone, backend.LoadMLock, backend.LoadDirectIO:
			return p, nil, fmt.Errorf("llamacpp: lazy mode %s needs a mapping; load mode %s has none", spec.LazyMode, spec.LoadMode)
		}
	}
	p.UseExtraBufts = spec.Repack

	if len(spec.Overrides) > 0 {
		if limit := int(llamaMaxTensorBuftOverrides()); len(spec.Overrides) > limit {
			return p, nil, fmt.Errorf("llamacpp: %d buffer overrides, at most %d", len(spec.Overrides), limit)
		}
		ovs := make([]LlamaModelTensorBuftOverride, 0, len(spec.Overrides)+1)
		for _, o := range spec.Overrides {
			if err := checkPattern(o.Pattern); err != nil {
				return p, nil, err
			}
			var buft GgmlBackendBufferType
			if o.Device == backend.DeviceCPURepack {
				rb, err := b.cpuRepackBuft()
				if err != nil {
					return p, nil, fmt.Errorf("llamacpp: override %q: %w", o.Pattern, err)
				}
				buft = rb
			} else {
				odev, cpu, err := b.device(o.Device)
				if err != nil {
					return p, nil, fmt.Errorf("llamacpp: override %q: %w", o.Pattern, err)
				}
				if cpu {
					buft = ggmlBackendCpuBufferType()
				} else {
					buft = ggmlBackendDevBufferType(odev)
				}
			}
			pat := append([]byte(o.Pattern), 0)
			keep = append(keep, pat)
			ovs = append(ovs, LlamaModelTensorBuftOverride{Pattern: &pat[0], Buft: buft})
		}
		ovs = append(ovs, LlamaModelTensorBuftOverride{})
		keep = append(keep, ovs)
		p.TensorBuftOverrides = &ovs[0]
	}
	return p, keep, nil
}

// checkPattern accepts the override patterns llama.cpp compiles with
// std::regex (ECMAScript), whose constructor throws — and so aborts the
// process — on syntax it does not know. RE2 accepts a superset (inline flags,
// named groups, \z, \p{…}), so those are refused here even when Go parses
// them.
func checkPattern(p string) error {
	if p == "" {
		return errors.New("llamacpp: empty buffer override pattern")
	}
	if _, err := regexp.Compile(p); err != nil {
		return fmt.Errorf("llamacpp: buffer override pattern %q: %w", p, err)
	}
	if strings.Contains(p, "(?") || strings.ContainsRune(p, 0) {
		return fmt.Errorf("llamacpp: buffer override pattern %q uses a group syntax std::regex does not accept", p)
	}
	for i := 0; i < len(p)-1; i++ {
		if p[i] != '\\' {
			continue
		}
		i++
		switch p[i] {
		case 'd', 'D', 'w', 'W', 's', 'S', 'b', 'B', '.', '\\', '[', ']', '(', ')', '{', '}', '*', '+', '?', '|', '^', '$', '-', '/':
		default:
			return fmt.Errorf("llamacpp: buffer override pattern %q: escape \\%c is not portable to std::regex", p, p[i])
		}
	}
	return nil
}

// contextParams turns the context part of a spec into llama_context_params.
// NCtx is per slot (Capabilities.NCtx); llama.cpp takes the total.
func (b *llamaBackend) contextParams(spec backend.LoadSpec) (LlamaContextParams, error) {
	p := llamaContextDefaultParams()
	slots := max(spec.Slots, 1)
	if maxSeq := int(llamaMaxParallelSequences()); slots > maxSeq {
		return p, fmt.Errorf("llamacpp: %d slots, at most %d", slots, maxSeq)
	}
	if spec.NCtx < 0 || spec.NBatch < 0 || spec.NUBatch < 0 || spec.Threads < 0 || spec.ThreadsBatch < 0 {
		return p, errors.New("llamacpp: negative context size, batch or thread count")
	}
	nctx := spec.NCtx
	if nctx == 0 {
		nctx = 4096
	}
	if int64(nctx)*int64(slots) > math.MaxInt32 {
		return p, fmt.Errorf("llamacpp: %d tokens × %d slots overflows the context size", nctx, slots)
	}
	p.NCtx = uint32(nctx * slots)
	p.NSeqMax = uint32(slots)
	if spec.NBatch > 0 {
		p.NBatch = uint32(spec.NBatch)
	}
	if spec.NUBatch > 0 {
		p.NUbatch = uint32(spec.NUBatch)
	}
	if p.NUbatch > p.NBatch {
		return p, fmt.Errorf("llamacpp: n_ubatch %d is larger than n_batch %d", p.NUbatch, p.NBatch)
	}
	if t := spec.WithDefaultThreads(); t.Threads > 0 {
		p.NThreads = int32(t.Threads)
		p.NThreadsBatch = int32(max(t.ThreadsBatch, 1))
	}
	// The planner sizes n_outputs_max to slots × (1 + DraftMax): the default
	// (n_batch) reserves n_vocab logits per batch token in the compute
	// buffer. llama.cpp needs at least one row per sequence.
	if spec.NOutputsMax < 0 {
		return p, errors.New("llamacpp: negative n_outputs_max")
	}
	if spec.NOutputsMax > 0 {
		if spec.NOutputsMax < slots {
			return p, fmt.Errorf("llamacpp: n_outputs_max %d is below the %d slots", spec.NOutputsMax, slots)
		}
		p.NOutputsMax = uint32(min(spec.NOutputsMax, int(p.NBatch)))
	}
	// One output row per slot and step: the sampled position. MTP
	// verification asks for DraftMax+1.
	p.NOutputsMaxPerSeq = uint32(max(1, spec.DraftMax+1))

	k, err := b.kvType(spec.KVTypeK)
	if err != nil {
		return p, err
	}
	v, err := b.kvType(spec.KVTypeV)
	if err != nil {
		return p, err
	}
	p.TypeK, p.TypeV = k, v
	if spec.FlashAttention {
		p.FlashAttnType = LlamaFlashAttnTypeEnabled
		if spec.FlashAttnSmallBatch {
			p.FlashAttnType = LlamaFlashAttnTypeSmallBatch
		}
	} else {
		p.FlashAttnType = LlamaFlashAttnTypeDisabled
		if v != GgmlTypeF16 && v != GgmlTypeF32 && v != GgmlTypeBf16 {
			return p, fmt.Errorf("llamacpp: a quantized V cache (%s) needs flash attention", spec.KVTypeV)
		}
	}
	p.SwaFull = spec.SWAFull
	p.KvUnified = spec.KVUnified
	p.NoPerf = false
	return p, nil
}

func (b *llamaBackend) kvType(name string) (GgmlType, error) {
	if name == "" {
		return GgmlTypeF16, nil
	}
	t, ok := b.lib.typesByName[strings.ToLower(name)]
	if !ok {
		return 0, fmt.Errorf("llamacpp: unknown KV cache type %q", name)
	}
	switch t {
	case GgmlTypeF32, GgmlTypeF16, GgmlTypeBf16, GgmlTypeQ8_0, GgmlTypeQ4_0, GgmlTypeQ4_1, GgmlTypeQ5_0, GgmlTypeQ5_1, GgmlTypeIq4Nl:
		return t, nil
	}
	return 0, fmt.Errorf("llamacpp: %s is not a KV cache type", name)
}

// lastErrors appends the error lines of a failed call, which is the only
// place llama.cpp says why.
func lastErrors(log []string) string {
	var errs []string
	for _, l := range log {
		low := strings.ToLower(l)
		if strings.Contains(low, "error") || strings.Contains(low, "failed") {
			errs = append(errs, l)
		}
	}
	if len(errs) > 3 {
		errs = errs[len(errs)-3:]
	}
	if len(errs) == 0 {
		return ""
	}
	return ": " + strings.Join(errs, "; ")
}

// readMetadata collects the GGUF scalar keys. Arrays are not readable
// through the C API (the vocabulary, the SWA pattern); the planner reads
// those from the file itself.
func (m *model) readMetadata() map[string]string {
	out := map[string]string{}
	n := int(llamaModelMetaCount(m.mdl))
	buf := make([]byte, 256)
	read := func(f func([]byte) int32) string {
		for {
			r := f(buf)
			if r < 0 {
				return ""
			}
			if int(r) < len(buf) {
				return string(buf[:r])
			}
			buf = make([]byte, int(r)+1)
		}
	}
	for i := 0; i < n; i++ {
		k := read(func(b []byte) int32 { return llamaModelMetaKeyByIndex(m.mdl, int32(i), &b[0], uintptr(len(b))) })
		if k == "" {
			continue
		}
		out[k] = read(func(b []byte) int32 { return llamaModelMetaValStrByIndex(m.mdl, int32(i), &b[0], uintptr(len(b))) })
	}
	out["llama.desc"] = read(func(b []byte) int32 { return llamaModelDesc(m.mdl, &b[0], uintptr(len(b))) })
	out["llama.n_params"] = strconv.FormatUint(llamaModelNParams(m.mdl), 10)
	out["llama.size"] = strconv.FormatUint(llamaModelSize(m.mdl), 10)
	return out
}

// Buffer sizes are only in the log: llama_get_memory_breakdown is C++ and
// returns a std::map.
var (
	reModelBuf   = regexp.MustCompile(`^load_tensors:\s+(\S+) model buffer size =\s+([0-9.]+) MiB`)
	reKVBuf      = regexp.MustCompile(`^llama_kv_cache:\s+(\S+) KV buffer size =\s+([0-9.]+) MiB`)
	reComputeBuf = regexp.MustCompile(`^sched_reserve:\s+(\S+) compute buffer size =\s+([0-9.]+) MiB`)
)

func memoryFromLog(log []string, b *llamaBackend) backend.MemoryReport {
	per := map[string]*backend.DeviceMemory{}
	var order []string
	add := func(buft, mib string, field func(*backend.DeviceMemory) *int64) {
		v, err := strconv.ParseFloat(mib, 64)
		if err != nil {
			return
		}
		dev := bufferDevice(buft)
		d, ok := per[dev]
		if !ok {
			d = &backend.DeviceMemory{Device: dev}
			per[dev] = d
			order = append(order, dev)
		}
		*field(d) += int64(v * (1 << 20))
	}
	for _, l := range log {
		if m := reModelBuf.FindStringSubmatch(l); m != nil {
			add(m[1], m[2], func(d *backend.DeviceMemory) *int64 { return &d.Weights })
			if v, err := strconv.ParseFloat(m[2], 64); err == nil {
				d := per[bufferDevice(m[1])]
				if d.WeightBuffers == nil {
					d.WeightBuffers = map[string]int64{}
				}
				d.WeightBuffers[m[1]] += int64(v * (1 << 20))
			}
		} else if m := reKVBuf.FindStringSubmatch(l); m != nil {
			add(m[1], m[2], func(d *backend.DeviceMemory) *int64 { return &d.KV })
		} else if m := reComputeBuf.FindStringSubmatch(l); m != nil {
			add(m[1], m[2], func(d *backend.DeviceMemory) *int64 { return &d.Compute })
		}
	}
	var r backend.MemoryReport
	for _, dev := range order {
		r.Devices = append(r.Devices, *per[dev])
	}
	return r
}

// bufferDevice maps a ggml buffer type name to a selector: host buffers
// (CPU, CPU_Mapped, CPU_REPACK, AMX, Vulkan_Host, CUDA_Host) are "cpu",
// device buffers ("Vulkan0", "CUDA0", "MTL0") are that device.
func bufferDevice(buft string) string {
	low := strings.ToLower(buft)
	switch {
	case strings.HasPrefix(low, "cpu"), low == "amx", strings.HasSuffix(low, "_host"), strings.HasPrefix(low, "blas"):
		return "cpu"
	}
	return Name + ":" + low
}

// Tokenizer: safe from any goroutine (the C API is), guarded against Close.

func (m *model) Tokenize(text string, addSpecial, parseSpecial bool) ([]backend.Token, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return nil, errClosed
	}
	if len(text) > math.MaxInt32/2 {
		return nil, fmt.Errorf("llamacpp: tokenize: %d bytes is too long", len(text))
	}
	if text == "" && !addSpecial {
		return []backend.Token{}, nil
	}
	buf := make([]int32, len(text)+8)
	for {
		n := llamaTokenize(m.vocab, text, int32(len(text)), &buf[0], int32(len(buf)), addSpecial, parseSpecial)
		if n == math.MinInt32 {
			return nil, errors.New("llamacpp: tokenize: result overflows")
		}
		if n >= 0 {
			out := make([]backend.Token, n)
			for i := range out {
				out[i] = backend.Token(buf[i])
			}
			return out, nil
		}
		buf = make([]int32, -n)
	}
}

func (m *model) checkTokens(tokens []backend.Token) error {
	for _, t := range tokens {
		if t < 0 || int(t) >= m.nVocab {
			return fmt.Errorf("llamacpp: token %d is outside the vocabulary (0..%d)", t, m.nVocab-1)
		}
	}
	return nil
}

// Detokenize renders special tokens as text: the tool-call and thinking
// markers of a chat format are special tokens, and the parser above needs
// to see them.
func (m *model) Detokenize(tokens []backend.Token) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return "", errClosed
	}
	if len(tokens) == 0 {
		return "", nil
	}
	if err := m.checkTokens(tokens); err != nil {
		return "", err
	}
	ids := make([]int32, len(tokens))
	for i, t := range tokens {
		ids[i] = int32(t)
	}
	buf := make([]byte, 8*len(ids)+16)
	for {
		n := llamaDetokenize(m.vocab, &ids[0], int32(len(ids)), &buf[0], int32(len(buf)), false, true)
		if n >= 0 {
			return string(buf[:n]), nil
		}
		buf = make([]byte, -n)
	}
}

func (m *model) TokenPiece(t backend.Token) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 {
		return "", errClosed
	}
	if err := m.checkTokens([]backend.Token{t}); err != nil {
		return "", err
	}
	buf := make([]byte, 64)
	for {
		n := llamaTokenToPiece(m.vocab, int32(t), &buf[0], int32(len(buf)), 0, true)
		if n >= 0 {
			return string(buf[:n]), nil
		}
		buf = make([]byte, -n)
	}
}

func (m *model) IsEOG(t backend.Token) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.mdl == 0 || t < 0 || int(t) >= m.nVocab {
		return false
	}
	return llamaVocabIsEog(m.vocab, int32(t))
}

func (m *model) Metadata() map[string]string {
	out := make(map[string]string, len(m.meta))
	for k, v := range m.meta {
		out[k] = v
	}
	return out
}

func (m *model) Capabilities() backend.Capabilities { return m.caps }
func (m *model) MemoryReport() backend.MemoryReport { return m.memory }

// Perf is llama.cpp's own prompt and generation timing for the context.
func (m *model) Perf() LlamaPerfContextData {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.ctx == 0 {
		return LlamaPerfContextData{}
	}
	return llamaPerfContext(m.ctx)
}

func (m *model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.mdl == 0 {
		return nil
	}
	for i := range m.slots {
		m.freeSampler(i)
	}
	// The drafter's context reads the target's cache: it goes first.
	m.dft.close()
	m.dft = nil
	if m.batch != 0 {
		llamaBatchExtFree(m.batch)
	}
	m.detachStream()
	llamaFree(m.ctx)
	llamaModelFree(m.mdl)
	m.batch, m.ctx, m.mdl, m.vocab, m.mem = 0, 0, 0, 0, 0
	return nil
}

var errClosed = errors.New("llamacpp: model is closed")

// logitsRow views the logits of one output as a Go slice; valid until the
// next evaluation.
func (m *model) logitsRow(idx int32) ([]float32, error) {
	p := llamaGetLogitsIth(m.ctx, idx)
	if p == nil {
		return nil, fmt.Errorf("llamacpp: no logits for batch entry %d", idx)
	}
	return unsafe.Slice(p, m.nVocab), nil
}
