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
	"sort"
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
}

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

// BufferOverride places tensors whose name matches Pattern (a regular
// expression, first match wins) on Device instead of where their layer goes.
// Device is a selector, or "cpu" for host memory.
type BufferOverride struct {
	Pattern string `json:"pattern"`
	Device  string `json:"device"`
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
	Threads     int
	// KVTypeK and KVTypeV are ggml type names (f16, q8_0). A quantized V
	// needs flash attention.
	KVTypeK string
	KVTypeV string
	// FlashAttention is on unless the device cannot do it.
	FlashAttention bool
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
}

// WithDefaultThreads fills a zero thread count with the physical cores this
// process may use (hwinfo.DefaultThreads), so a plan records the number it
// ran with. A model fully offloaded to a GPU keeps zero: it does no CPU
// matrix work, and polling threads would only take cores from the host.
func (s LoadSpec) WithDefaultThreads() LoadSpec {
	if s.Threads == 0 && !s.fullyOnGPU() {
		s.Threads = hwinfo.DefaultThreads()
	}
	return s
}

// fullyOnGPU is a spec that names a GPU device and offloads every layer.
func (s LoadSpec) fullyOnGPU() bool {
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
	SWAFull  bool
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
