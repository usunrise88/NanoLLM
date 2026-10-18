// Package backend is the boundary between NanoLLM and the inference engines
// (SPEC §2, §3).
//
// The core never calls llama.cpp or onnxruntime-genai directly. It talks to a
// Backend, which probes devices and loads models, and to a Model, which
// tokenizes, decodes batches, samples and saves or restores sequence state.
// Everything above this package — the pool, the scheduler, the prompt cache —
// is engine-agnostic, so a backend is one package registering itself from
// init(), the same way an API dialect is.
package backend

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
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
	// Version is the engine's build identifier, e.g. the llama.cpp build
	// number, for /healthz and `nanollm version`.
	Version() string
	Devices(ctx context.Context) ([]Device, error)
	Load(ctx context.Context, spec LoadSpec) (Model, error)
	Close() error
}

// LoadSpec is a resolved load: the planner has already turned a preset into
// concrete placement, so a backend never sees a budget it has to interpret.
type LoadSpec struct {
	Path string
	// DrafterPath is the MTP drafter, or empty. It is placed on the device
	// holding the target's final layers, since it reads their activations.
	DrafterPath string
	Device      string // selector
	// GPULayers is the number of layers offloaded; -1 means all.
	GPULayers int
	// HostTensors are tensor-name patterns kept in host memory even when
	// their layer is offloaded (PLE placement, modes M2 and M3).
	HostTensors []string
	// MMap and DirectIO choose how weights are read; MLock locks them.
	MMap     bool
	DirectIO bool
	MLock    bool
	// Repack enables the CPU repacked kernels, which copy weights into
	// anonymous memory and so defeat file-backed modes.
	Repack  bool
	NCtx    int
	Slots   int
	NBatch  int
	NUBatch int
	Threads int
	KVType  string
	// FlashAttention is on unless the device cannot do it.
	FlashAttention bool
}

// Token is a vocabulary id.
type Token int32

// Batch is one decode step across slots: every token belongs to a sequence
// (slot) at a position, and Logits marks the ones whose distribution is needed
// for sampling. Continuous batching puts tokens of several slots in one Batch.
type Batch struct {
	Tokens []Token
	Pos    []int32
	Seq    []int32
	Logits []bool
}

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
	Seed             int64
	// Grammar is GBNF; empty means unconstrained.
	Grammar string
}

// Model is one loaded model with its context and slots.
type Model interface {
	Tokenize(text string, addSpecial, parseSpecial bool) ([]Token, error)
	Detokenize(tokens []Token) (string, error)
	// TokenPiece is the text of one token, for streaming.
	TokenPiece(t Token) (string, error)

	// Metadata exposes GGUF key/values: architecture, chat template, context
	// length. Templates are read from here unless the manifest overrides them.
	Metadata() map[string]string

	// SetSampler installs the sampling chain for a slot.
	SetSampler(seq int, spec SamplerSpec) error
	// Decode runs one batch.
	Decode(ctx context.Context, b Batch) error
	// Sample draws the next token for a slot from the last decoded logits and
	// accepts it into the sampler (penalties, grammar state).
	Sample(seq int) (Token, error)
	IsEOG(t Token) bool

	// TrimSeq drops a slot's cached state from position pos on: prefix reuse
	// for models whose KV cache trims (Gemma 4). It reports false when the
	// model's state cannot be trimmed (recurrent layers), and the caller
	// restores a snapshot instead.
	TrimSeq(seq int, pos int) bool
	// SaveSeq and RestoreSeq move a slot's full state in and out — snapshots
	// for the Qwen3.5 fallback, and the golden state round-trip test.
	SaveSeq(seq int) ([]byte, error)
	RestoreSeq(seq int, state []byte) error

	Close() error
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
