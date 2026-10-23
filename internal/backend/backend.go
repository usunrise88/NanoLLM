// Package backend is the boundary between NanoLLM and the inference engines
// (SPEC §2, §3).
//
// The core never calls llama.cpp or onnxruntime-genai directly. It talks to a
// Backend, which probes devices and loads models, and to a Model, which
// tokenizes and runs batched steps. Sampling, MTP drafting and verification
// happen inside a Step, so the scheduler sees only accepted tokens and the
// per-token round trips stay inside the engine.
// Everything above this package — the pool, the scheduler, the prompt cache —
// is engine-agnostic, so a backend is one package registering itself from
// init(), the same way an API dialect is.
package backend

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/usunrise88/nanollm/internal/hwinfo"
)

// DeviceType is the class of a device, as device selectors name it.
type DeviceType string

const (
	CPU DeviceType = "cpu"
	GPU DeviceType = "gpu"
	NPU DeviceType = "npu"
)

// Device is one place a model can run.
type Device struct {
	// Backend is the registered backend name, e.g. "llamacpp".
	Backend string `json:"backend"`
	// ID is the backend's own name for the device, e.g. "cpu", "vulkan0",
	// "cuda1". Backend and ID together form the selector "llamacpp:vulkan0".
	ID   string     `json:"id"`
	Type DeviceType `json:"type"`
	// Name is the human-readable device name the driver reports.
	Name string `json:"name"`
	// MemoryTotal and MemoryFree are in bytes; zero when the driver does not
	// say. On unified memory both describe the shared pool.
	MemoryTotal int64 `json:"memory_total"`
	MemoryFree  int64 `json:"memory_free"`
	Unified     bool  `json:"unified,omitempty"`
	// Features names what the loaded build can do that changes how to run it
	// (the Feature constants); empty when it says nothing.
	Features []string `json:"features,omitempty"`
}

// Has reports whether the device lists the feature.
func (d Device) Has(feature string) bool { return slices.Contains(d.Features, feature) }

// Device features.
const (
	// FeatureMultiRowQ4_0 means the CPU kernels for Q4_0 take several rows of
	// the activation in one matrix-vector call, so checking the 2-3 tokens of
	// a draft costs about one token (patches/next/0002, the feature
	// REPACK_Q4_0_ROWS of the CPU backend). Without it speculative decoding
	// on the CPU does not pay (docs/AUTOTUNE.md, the cost of a batch).
	FeatureMultiRowQ4_0 = "multirow-q4_0"
	// FeatureOutputShortlist means the library can compute the logits of a
	// subset of the vocabulary and spread them over a full row
	// (patches/llama.cpp/0009; LoadSpec.HeadShortlist, OPT-4): the head reads
	// fewer rows of the 262144-row matrix on every token.
	FeatureOutputShortlist = "output-shortlist"
	// FeatureFlashAttnSmallBatch means flash attention on this GPU is slow for
	// long batches: the matmul attention prefills faster as the context grows
	// (Iris Xe, Vulkan, Windows driver: pp512 at 4096 tokens of context 385
	// against 315 tok/s, at 8192 346 against 241), so the planner asks for
	// flash attention on batches of up to 8 tokens and on sliding-window layers
	// only (patches/llama.cpp/0007, LLAMA_FLASH_ATTN_TYPE_SMALL_BATCH).
	FeatureFlashAttnSmallBatch = "fa-small-batch"
	// FeatureBackendSampling means sampling pays in the device's graph: the
	// sampled token (and the drafter's ten candidates) leave the GPU instead
	// of a 1 MiB row of logits per output, which the host then scans (the
	// drafter's top-10 alone costs 0.23 ms per token in Go). Listed for CUDA
	// (LoadSpec.BackendSampling).
	FeatureBackendSampling = "backend-sampling"
	// FeatureCudaNano means the CUDA backend is NanoLLM's own ggml-cuda
	// (patches/llama.cpp, the "cuda:" subjects), not upstream's: it keeps a
	// CUDA graph per batch shape and has switches for the kernels it changed
	// (GGML_CUDA_NANO_*, the NANO_* entries of llama_print_system_info).
	FeatureCudaNano = "cuda-nano"
)

// Selector is how configuration and requests name this device.
func (d Device) Selector() string { return d.Backend + ":" + d.ID }

// Backend is one inference engine.
type Backend interface {
	Name() string
	// Version is the engine's build identifier, e.g. the llama.cpp tag and
	// commit, for /healthz and `nanollm version`.
	Version() string
	Devices(ctx context.Context) ([]Device, error)
	Load(ctx context.Context, spec LoadSpec) (Model, error)
	Close() error
}

// LoadMode is how weights are read from the file (llama_load_mode).
type LoadMode string

const (
	LoadAuto      LoadMode = ""           // engine default
	LoadNone      LoadMode = "none"       // plain reads into anonymous memory
	LoadMMap      LoadMode = "mmap"       // file-backed mapping
	LoadMLock     LoadMode = "mlock"      // reads, then locked
	LoadMMapMLock LoadMode = "mmap+mlock" // mapping, locked
	LoadDirectIO  LoadMode = "dio"        // O_DIRECT reads, bypassing page cache
)

// LazyMode is whether tensors are read on first use (llama_lazy_mode). Lazy
// loading needs a mapping, so it never combines with LoadDirectIO.
type LazyMode string

const (
	LazyOff  LazyMode = "off"
	LazyAuto LazyMode = "auto" // only tensors larger than 4 GiB
	LazyOn   LazyMode = "on"
)

// DeviceCPURepack is the BufferOverride device that puts tensors in the CPU
// backend's repacked buffer: the weights are copied into the layout its
// kernels read fastest (anonymous memory), for those tensors alone, with
// LoadSpec.Repack off for the rest.
const DeviceCPURepack = "cpu:repack"

// BufferOverride places tensors whose name matches Pattern (a regular
// expression, first match wins) on Device instead of where their layer goes.
// Device is a selector, or "cpu" for host memory.
type BufferOverride struct {
	Pattern string `json:"pattern"`
	Device  string `json:"device"`
}

// StreamRange is a file range one layer reads on every token and that the
// plan leaves file-backed (M8's cold FFN tensors, M9's weights).
type StreamRange struct {
	Layer  int
	Offset int64
	Length int64
}

// StreamSpec is how a backend feeds file-backed layers (OPT-8): after layer
// L is computed it asks the kernel to read the next Ahead layers, and, with
// Evict, to drop L's pages unless L is one of the first Keep layers. Without
// the page cache's LRU in the way, a budget smaller than the layers' total
// keeps Keep layers resident for good instead of cycling every page through
// the cache without a hit. Zero Ahead and no Evict leave the loader alone.
type StreamSpec struct {
	// Ranges are the plan's file-backed layer ranges, in file offsets.
	Ranges []StreamRange
	Ahead  int
	Keep   int
	Evict  bool
	// Method is how the layers are read ahead: "" or "advise" (the system's
	// readahead: madvise WILLNEED, PrefetchVirtualMemory), "pread" (reads
	// of the file, Depth at a time, into the page cache; OPT-9) or "direct"
	// (O_DIRECT reads into a ring of buffers the tensors are pointed at, no
	// page cache: Linux, libraries with patches/next/0010). Depth is zero for four.
	Method string
	Depth  int
	// Probe installs the layer callback and does nothing in it, to measure
	// what the callback alone costs.
	Probe bool
}

// Active is whether the backend has anything to do with the spec.
func (s *StreamSpec) Active() bool {
	return s != nil && len(s.Ranges) > 0 && (s.Ahead > 0 || s.Evict || s.Probe || s.Method == "direct")
}

// WithEnv applies the NANOLLM_STREAM_AHEAD, _KEEP, _EVICT, _METHOD, _DEPTH and _PROBE
// variables over the spec, for the benchmark sweeps and the autotuner: the
// configuration of a served model sets the fields directly.
func (s *StreamSpec) WithEnv() *StreamSpec {
	if s == nil {
		return nil
	}
	o := *s
	if v, err := strconv.Atoi(os.Getenv("NANOLLM_STREAM_AHEAD")); err == nil && v >= 0 {
		o.Ahead = v
	}
	if v, err := strconv.Atoi(os.Getenv("NANOLLM_STREAM_KEEP")); err == nil && v >= 0 {
		o.Keep = v
	}
	switch os.Getenv("NANOLLM_STREAM_EVICT") {
	case "1":
		o.Evict = true
	case "0":
		o.Evict = false
	}
	if v := os.Getenv("NANOLLM_STREAM_METHOD"); v != "" {
		o.Method = v
	}
	if v, err := strconv.Atoi(os.Getenv("NANOLLM_STREAM_DEPTH")); err == nil && v >= 0 {
		o.Depth = v
	}
	if os.Getenv("NANOLLM_STREAM_PROBE") == "1" {
		o.Probe = true
	}
	return &o
}

// LoadSpec is a resolved load: the planner has already turned a preset into
// concrete placement, so a backend never sees a budget it has to interpret.
type LoadSpec struct {
	Path string
	// DrafterPath is the MTP drafter, or empty. It shares the target's KV
	// cache and reads its activations, so it gets the same context shape.
	DrafterPath string
	Device      string // selector
	// GPULayers is the number of layers offloaded; -1 means all.
	GPULayers int
	// Overrides keep tensors off the device their layer is on (PLE placement
	// in modes M2 and M3 when the engine does not already do it).
	Overrides []BufferOverride
	LoadMode  LoadMode
	LazyMode  LazyMode
	// Repack enables the CPU repacked kernels (use_extra_bufts), which copy
	// weights into anonymous memory and so defeat file-backed modes.
	Repack bool
	// NCtx is tokens per slot, as Capabilities reports it; the engine's
	// context holds NCtx·Slots. Zero means 4096.
	NCtx    int
	Slots   int
	NBatch  int
	NUBatch int
	// NOutputsMax bounds the logits rows one Step can return (n_outputs_max):
	// slots × (1 + DraftMax). Left at the engine default (n_batch), the
	// compute buffer reserves n_vocab × n_ubatch logits — 512 MiB for E4B —
	// that the planner did not budget.
	NOutputsMax int
	// Threads is the CPU threads for decoding and ThreadsBatch for prompt
	// processing (llama.cpp's n_threads and n_threads_batch); zero is one
	// per physical core, and a zero ThreadsBatch follows Threads. Decoding
	// is limited by memory bandwidth and can peak at fewer threads than
	// prefill, which is limited by arithmetic.
	Threads      int
	ThreadsBatch int
	// KVTypeK and KVTypeV are ggml type names (f16, q8_0). A quantized V
	// needs flash attention.
	KVTypeK string
	KVTypeV string
	// FlashAttention is on unless the device cannot do it.
	FlashAttention bool
	// FlashAttnSmallBatch, with FlashAttention, keeps flash attention for
	// batches of up to 8 tokens and on sliding-window layers and runs longer
	// batches on the other layers through the matmul attention
	// (FeatureFlashAttnSmallBatch). The KV cache layout is that of flash
	// attention either way.
	FlashAttnSmallBatch bool
	// BackendSampling runs a slot's sampler chain in the device graph when the
	// chain has no grammar and no penalties and the request reads no logits
	// (SamplerSpec.RawLogits), and the MTP drafter's top-10 too
	// (FeatureBackendSampling). Other chains sample on the host as before.
	BackendSampling bool
	// CudaNano sets switches of NanoLLM's ggml-cuda before the model loads
	// (FeatureCudaNano; GGML_CUDA_NANO_<NAME>): the setting is the process's,
	// and takes effect for the graphs computed afterwards.
	CudaNano map[string]int
	// SWAFull keeps the full context in the sliding-window layers too, so the
	// cache can be trimmed to any position; without it a trim further back
	// than the window forces a fresh prefill.
	SWAFull bool
	// KVUnified shares one KV buffer across slots.
	KVUnified bool
	// DraftMax and DraftPMin bound MTP drafting: at most DraftMax tokens per
	// step, stopping when the drafter's top probability drops below DraftPMin.
	DraftMax  int
	DraftPMin float64
	// HeadShortlist are the vocabulary ids (ascending, unique, a multiple of 8
	// of them) whose logits the output head computes while no request needs
	// the whole vocabulary; empty keeps the whole head. The model holds a
	// second, repacked copy of those rows (FeatureOutputShortlist).
	HeadShortlist []int32
	// VerifyRows bounds the rows of a step spent on drafts and their checks,
	// summed over the decoding slots (Σ 1 + drafts); zero is no bound. The
	// scheduler enforces it (sched.Loop.SetVerifyRows).
	VerifyRows int
	// NGram is the match length of prompt-lookup drafting (internal/ngram),
	// which drafts up to DraftMax tokens from the sequence itself; zero is
	// off. It is used when no MTP drafter is loaded.
	NGram int
	// Stream is the file-backed layers of the plan and how to feed them; nil
	// when the plan has none (OPT-8).
	Stream *StreamSpec
}

// WithDefaultThreads fills a zero thread count with the physical cores this
// process may use (hwinfo.DefaultThreads), so a plan records the number it
// ran with. A zero ThreadsBatch follows Threads, except when neither was set
// and the CPU is hybrid (hwinfo.DefaultBatchThreads). A model fully offloaded to a GPU keeps zero: it does no CPU
// matrix work, and polling threads would only take cores from the host.
func (s LoadSpec) WithDefaultThreads() LoadSpec {
	auto := s.Threads == 0 && s.ThreadsBatch == 0 && !s.FullyOnGPU()
	if s.Threads == 0 && !s.FullyOnGPU() {
		s.Threads = hwinfo.DefaultThreads()
	}
	if s.ThreadsBatch == 0 {
		s.ThreadsBatch = s.Threads
		// Neither count was asked for: prompt chunks use every logical CPU of a
		// hybrid machine, decoding keeps the performance cores.
		if auto {
			s.ThreadsBatch = max(s.Threads, defaultBatchThreads())
		}
	}
	return s
}

// defaultBatchThreads is a variable for the tests.
var defaultBatchThreads = hwinfo.DefaultBatchThreads

// FullyOnGPU is a spec that names a GPU device and offloads every layer.
func (s LoadSpec) FullyOnGPU() bool {
	id := s.Device
	if i := strings.LastIndex(id, ":"); i >= 0 {
		id = id[i+1:]
	}
	return s.GPULayers == -1 && id != "" && id != "cpu"
}

// Token is a vocabulary id.
type Token int32

// SamplerSpec is the per-slot sampling chain, built once per request.
type SamplerSpec struct {
	Temperature      float64
	TopP             float64
	TopK             int
	MinP             float64
	RepeatPenalty    float64
	PresencePenalty  float64
	FrequencyPenalty float64
	LogitBias        map[Token]float64
	// Seed seeds the sampler; a negative seed is random. Zero is a seed like
	// any other, so a zero SamplerSpec is reproducible.
	Seed int64
	// RawLogits says the caller reads the logits rows (logprobs), so the
	// chain samples on the host even where the model samples on the device.
	RawLogits bool
	// Grammar is GBNF; empty means unconstrained. GrammarRoot defaults to
	// "root". With GrammarTriggers set the grammar is lazy: unconstrained
	// until the output contains one of the trigger strings (tool_choice auto).
	Grammar         string
	GrammarRoot     string
	GrammarTriggers []string
}

// SeqStep is one slot's share of a Step.
//
// Tokens are evaluated at positions Pos, Pos+1, … — a prefill chunk, or for a
// decode step the single token the previous Step returned last. If Sample is
// set, the backend samples from the logits after the last token, drafts up to
// MaxDraft more with MTP, verifies them and returns what was accepted.
type SeqStep struct {
	Seq    int
	Tokens []Token
	Pos    int
	Sample bool
	// MaxDraft caps drafting for this slot in this step: 0 turns it off (the
	// batch is full, or the slot is under a grammar that drafting would
	// break). It is clamped to Capabilities().MaxDraft.
	MaxDraft int
	// Logprobs is the number of alternatives to return per output token: -1
	// for none, 0 for the chosen token only, up to 20.
	Logprobs int
}

// SeqOut is what one sampling slot produced.
//
// Tokens holds 1..MaxDraft+1 tokens in order. All but the last are already
// in the KV cache (they were verified as part of the batch); the last is not,
// and the caller feeds it in the next Step at the position after them. The
// grammar and penalties have seen exactly these tokens. Generation stops at
// an end-of-generation token: it is the last element when EOG is set, and
// nothing after it is returned or kept in the cache.
type SeqOut struct {
	Seq      int
	Tokens   []Token
	Logprobs []TokenLogprobs // one per token, when requested
	EOG      bool
	// Drafted and Accepted count MTP draft tokens in this step.
	Drafted  int
	Accepted int
}

// TokenLogprobs is the log-softmax of the raw logits (before the sampler) at
// one output position: the chosen token and the top alternatives.
type TokenLogprobs struct {
	Token   Token
	Logprob float64
	Top     []TokenLogprob
}

type TokenLogprob struct {
	Token   Token
	Logprob float64
}

// Capabilities is the shape of a loaded model's context.
type Capabilities struct {
	NCtx   int // tokens per slot
	Slots  int
	NBatch int // tokens per Step across all slots, drafts included
	// MaxDraft is 0 when no drafter is loaded.
	MaxDraft int
	// VerifyRows is the spec's bound on the rows spent on drafts in a step;
	// zero is none. The scheduler starts from it.
	VerifyRows int
	SWAFull    bool
	// SWAWindow is how far back a trim may reach without a fresh prefill when
	// SWAFull is off; 0 when the model has no sliding-window layers.
	SWAWindow int
	NVocab    int
}

// MemoryReport is what a loaded model holds, in bytes, as the engine reports
// it. The planner's estimate is checked against it.
type MemoryReport struct {
	Devices []DeviceMemory `json:"devices"`
}

type DeviceMemory struct {
	Device  string `json:"device"` // selector, or "cpu"
	Weights int64  `json:"weights"`
	KV      int64  `json:"kv"`
	Compute int64  `json:"compute"`
	// WeightBuffers splits Weights by the name of the ggml buffer type that
	// holds them (CPU_Mapped, CPU_REPACK, Vulkan0, …): how much of the model
	// was repacked, and how much stayed in the file's mapping.
	WeightBuffers map[string]int64 `json:"weight_buffers,omitempty"`
}

// Model is one loaded model with its context and slots.
//
// Step and the methods that change a slot's state are called from one
// goroutine, the model's scheduling loop; the tokenizer methods are safe from
// any goroutine.
type Model interface {
	Tokenize(text string, addSpecial, parseSpecial bool) ([]Token, error)
	// Detokenize and TokenPiece render special tokens as their text: the
	// turn, thinking and tool-call markers of a chat format are special
	// tokens, and the parser above the backend needs to see them.
	Detokenize(tokens []Token) (string, error)
	// TokenPiece is the text of one token, for streaming.
	TokenPiece(t Token) (string, error)
	IsEOG(t Token) bool

	// Metadata exposes GGUF key/values: architecture, chat template, context
	// length. Templates are read from here unless the manifest overrides them.
	Metadata() map[string]string
	Capabilities() Capabilities
	MemoryReport() MemoryReport

	// SetSampler installs the sampling chain for a slot. History is the
	// slot's tokens already in the cache (prompt, and a reused prefix after a
	// prompt-cache hit): penalties see them, the grammar does not.
	SetSampler(seq int, spec SamplerSpec, history []Token) error
	// Step evaluates one batch across slots: prefill chunks and decode steps
	// together, at most Capabilities().NBatch tokens including drafts. It
	// returns one SeqOut per SeqStep that asked to Sample. Cancellation is
	// checked before the batch is submitted; a batch in flight finishes.
	Step(ctx context.Context, steps []SeqStep) ([]SeqOut, error)

	// TrimSeq drops a slot's cached state from position pos on. It reports
	// false when the cache cannot be trimmed that far back (sliding-window
	// layers without SWAFull, recurrent layers), and the caller clears the
	// slot and prefills again. Any MTP carry-over is dropped.
	TrimSeq(seq int, pos int) bool
	// ClearSeq empties a slot.
	ClearSeq(seq int)
	// SaveSeq and RestoreSeq move a slot's full state in and out, for the
	// state round-trip test and future snapshot-based prompt caching.
	SaveSeq(seq int) ([]byte, error)
	RestoreSeq(seq int, state []byte) error

	Close() error
}

// Warner is implemented by a backend that loaded with problems worth showing
// but not fatal: a GPU library in lib/ that did not load, and why (SPEC §3:
// a missing backend is never fatal). `nanollm devices` prints them.
type Warner interface {
	Warnings() []string
}

// ErrNotImplemented marks a backend whose engine binding has not landed yet.
var ErrNotImplemented = errors.New("not implemented")

// Factory opens a backend. libDir is absolute; a backend whose libraries are
// absent returns an error saying which file it looked for, and the server
// skips it rather than failing (SPEC §3: a missing backend is never fatal).
type Factory func(libDir string) (Backend, error)

var (
	mu        sync.RWMutex
	factories = map[string]Factory{}
)

// Register makes a backend available under name. Called from init(); a
// duplicate is a programming error.
func Register(name string, f Factory) {
	mu.Lock()
	defer mu.Unlock()
	if _, dup := factories[name]; dup {
		panic(fmt.Sprintf("backend %q registered twice", name))
	}
	factories[name] = f
}

// Open opens a registered backend.
func Open(name, libDir string) (Backend, error) {
	mu.RLock()
	f, ok := factories[name]
	mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("backend %q is not built into this binary (have %v)", name, Registered())
	}
	return f(libDir)
}

// Registered lists the backends built into this binary, sorted.
func Registered() []string {
	mu.RLock()
	defer mu.RUnlock()
	out := make([]string, 0, len(factories))
	for n := range factories {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
