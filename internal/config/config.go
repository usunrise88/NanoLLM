// Package config defines every knob NanoLLM has (SPEC §8).
//
// Precedence: flags > NANOLLM_* env > YAML file > defaults. The server starts
// with no configuration file at all, bound to loopback with authentication off;
// `nanollm init` writes the file a real deployment runs with.
package config

import (
	"path/filepath"
	"strings"
)

type Config struct {
	Server     Server     `yaml:"server"`
	Auth       Auth       `yaml:"auth"`
	API        API        `yaml:"api"`
	Backends   Backends   `yaml:"backends"`
	Devices    Devices    `yaml:"devices"`
	Models     Models     `yaml:"models"`
	Pool       Pool       `yaml:"pool"`
	Cache      Cache      `yaml:"cache"`
	Generation Generation `yaml:"generation"`
	Queue      Queue      `yaml:"queue"`
	Registry   Registry   `yaml:"registry"`
	Bench      Bench      `yaml:"bench"`
	UI         UI         `yaml:"ui"`
	Log        Log        `yaml:"log"`
}

type Server struct {
	Addr string `yaml:"addr"`
	// DataDir holds downloaded weights, the job database and the spool.
	DataDir string `yaml:"data_dir"`
	// MaxBody caps a request body in bytes. A long-context chat is a few
	// megabytes of JSON; anything far beyond that is a mistake or an attack.
	MaxBody int64 `yaml:"max_body"`
}

// Auth in mode "open" is accepted only when Server.Addr binds a loopback
// address; otherwise the server refuses to start.
type Auth struct {
	Mode string   `yaml:"mode"` // apikey | open
	Keys []APIKey `yaml:"keys"`
}

// APIKey is one credential.
//
//	keys:
//	  - {name: admin, secret: sk-nanollm-..., admin: true, interactive: true}
//	  - {name: user,  secret: sk-nanollm-..., rps: 10}
type APIKey struct {
	Name string `yaml:"name"`
	// Secret is the key itself, or "sha256:<hex>" to keep the plaintext out of
	// the configuration file.
	Secret string `yaml:"secret"`
	// Admin unlocks model, device and configuration management.
	Admin bool `yaml:"admin"`
	// Interactive work overtakes a batch backlog. It belongs to the key rather
	// than to the request because a request field would let every client call
	// itself urgent.
	Interactive bool `yaml:"interactive"`
	// RPS caps this key's request rate. Zero means unlimited.
	RPS float64 `yaml:"rps"`
}

// Dialect names api.dialects accepts. Declared here rather than only in the
// packages that register them so validation can name one without importing
// the HTTP layer.
const (
	DialectOpenAI = "openai"
	DialectNative = "native"
	DialectOllama = "ollama" // opt-in: not in the default dialects
)

type API struct {
	Dialects []string `yaml:"dialects"`
}

// Backend names backends.enabled accepts.
const (
	BackendLlamaCpp = "llamacpp"
	BackendORTGenAI = "ortgenai" // v2, NPU only
	BackendMLX      = "mlx"      // v3, macOS only
	// BackendFake is the scripted development backend. It is never enabled by
	// default and serves no real model.
	BackendFake = "fake"
)

type Backends struct {
	// LibDir holds libllama, the ggml backends and, optionally,
	// onnxruntime-genai. A relative path is resolved against the directory of
	// the binary, never the working directory: libraries are always loaded by
	// absolute path (SPEC §10).
	LibDir  string   `yaml:"lib_dir"`
	Enabled []string `yaml:"enabled"`
}

// Device classes devices.default and a request's device field accept, besides
// an explicit "<backend>:<device>" such as "llamacpp:vulkan0".
const (
	DeviceAuto = "auto"
	DeviceCPU  = "cpu"
	DeviceGPU  = "gpu"
	DeviceNPU  = "npu"
)

type Devices struct {
	Default string `yaml:"default"`
	// Fallback lets a request that cannot run on its device run on the next
	// one, with a device_fallback warning, instead of failing with 409.
	Fallback bool `yaml:"fallback"`
	// CPUThreads is the CPU thread budget. Zero means one per physical core.
	CPUThreads int `yaml:"cpu_threads"`
	// CPUThreadsBatch is the thread count for prompt processing; zero is
	// the same as CPUThreads. Prefill is limited by arithmetic and takes
	// more threads than decoding, which memory bandwidth limits.
	CPUThreadsBatch int `yaml:"cpu_threads_batch"`
	// NUMA is the memory placement on a machine with several NUMA nodes:
	// auto (interleave when there are several nodes), off (the kernel's
	// default), interleave, distribute (ggml pins its threads to the nodes in
	// turn and the weights are repacked by those threads, so each reads its own
	// node: 30.2 against 24.6 tok/s of interleave on a 4-node EPYC at 16
	// threads, docs/AUTOTUNE.md; wants decode threads at most the physical
	// cores), or node:N (memory and threads on one node). A process already
	// started under numactl keeps its policy.
	NUMA string `yaml:"numa"`
}

type Models struct {
	Default   string                   `yaml:"default"`
	Preload   []string                 `yaml:"preload"`
	Overrides map[string]ModelOverride `yaml:"overrides"`
}

// KV cache types a model override accepts.
const (
	KVF16  = "f16"
	KVQ8_0 = "q8_0"
)

// Tensor orders of the model file a model override accepts (OPT-7).
const (
	LayoutOriginal = "original"
	LayoutNano     = "nano"
)

// ModelOverride is how one model is loaded. Every field is optional; an empty
// one means the planner or the manifest decides.
type ModelOverride struct {
	// Preset is a hard memory budget from SPEC §4 (plan.Presets).
	Preset string `yaml:"preset"`
	// Mode is auto (the planner picks within the preset), a mode name or its
	// short ID, M1…M10.
	Mode   string `yaml:"mode"`
	KVType string `yaml:"kv_type"`
	// Repack is "" or "auto" (the planner decides: the CPU's repacked layout
	// wherever the mode has kernels for it), "off" (the weights stay as the file
	// has them and the generic kernels run: the fallback for a CPU without
	// repack kernels, and a knob for the autotuner; docs/QUALITY-STAGE2.md: KLD
	// within the noise of another kernel variant) or "on" (the repacked layout
	// even where the planner keeps weights in the file: every tensor is loaded
	// into memory, so the memory budget is not kept; for a machine with room
	// to spare and for experiments, the ram-2g / 1.5g runs of docs/AUTOTUNE.md
	// show what happens under a tight budget).
	Repack string `yaml:"repack"`
	// FlashAttention is "" or "auto" (on; on a GPU that lists fa-small-batch,
	// on batches of up to 8 tokens only), "on" (on for every batch), "off" (the
	// matmul attention everywhere; a q8_0 cache needs flash attention) or
	// "small" (on for small batches only, the matmul attention for long ones on
	// the full-attention layers; needs libraries with patches/llama.cpp/0007).
	FlashAttention string `yaml:"flash_attention"`
	// Sampling is "" or "auto" (in the GPU's graph where the device lists
	// backend-sampling, CUDA; on the host elsewhere), "host" or "device".
	// Requests with a grammar, penalties or logprobs sample on the host
	// either way.
	Sampling string `yaml:"sampling"`
	// CudaNano sets switches of NanoLLM's own ggml-cuda for this model, by the
	// names it lists (the NANO_* entries of llama_print_system_info; the same as
	// GGML_CUDA_NANO_<NAME> in the environment): the block shape a measurement
	// of this card chose (MMVQ_NWARPS: 12), or a change turned off to compare
	// (NORM_LEAN: 0). Ignored where the device does not list cuda-nano.
	CudaNano map[string]int `yaml:"cuda_nano"`
	// NCtx and Slots take "auto": what the preset leaves after weights, down
	// to a floor of 4096 tokens and one slot.
	NCtx        Auto        `yaml:"n_ctx"`
	Slots       Auto        `yaml:"slots"`
	Speculative Speculative `yaml:"speculative"`
	// Stream tunes the file-backed layers of the modes that leave weights
	// in the file (M8, M9); every field zero leaves the loader alone.
	Stream Stream `yaml:"stream"`
	// Layout is the tensor order of the model file that loads: "" or
	// "original" (the file as published) or "nano" (the copy `nanollm models
	// relayout` wrote beside it: tensors in execution order, the pinned set
	// in one range, a layer's weights in one range; the plan is made from
	// that file's own offsets). Nano needs the copy and refuses to load
	// without it; the bytes of every tensor are the same.
	Layout string `yaml:"layout"`
}

// Stream is OPT-8: the layers a plan leaves file-backed are read ahead of
// the compute and dropped behind it (docs/STAGE2.md).
type Stream struct {
	// Ahead is how many layers after the one just computed are asked for.
	Ahead int `yaml:"ahead"`
	// Keep is how many leading layers stay in the page cache.
	Keep int `yaml:"keep"`
	// Evict drops a layer's pages once it is computed.
	Evict bool `yaml:"evict"`
	// Method is how the layers are read ahead: advise (the system's
	// readahead, the default) or pread (reads of the file, Depth at a time;
	// for an NVMe disk, which needs a queue deeper than one).
	Method string `yaml:"method"`
	// Depth is the reads in flight of method pread; zero is four.
	Depth int `yaml:"depth"`
}

// Speculative decoding types.
const (
	SpecMTP  = "mtp"
	SpecNone = "none"
	// SpecNGram drafts by lookup in the sequence itself (OPT-5): no
	// drafter model, nothing to load, and exact output.
	SpecNGram = "ngram"
)

type Speculative struct {
	Type string `yaml:"type"`
	// NMax is the most tokens the drafter proposes per step.
	NMax int `yaml:"n_max"`
	// PMin is the drafter probability below which a draft stops early.
	PMin float64 `yaml:"p_min"`
	// VerifyRows bounds the rows of one scheduler step spent on drafts and
	// their checks, summed over the sequences decoding at once. Zero is
	// auto: on a CPU with the multi-row kernels 1 + n_max (drafts only while
	// one sequence decodes: with more the checks cost more than they save),
	// no bound elsewhere. A negative number removes the bound.
	VerifyRows int `yaml:"verify_rows"`
	// Match is the number of trailing tokens that must occur earlier in the
	// sequence for type ngram to draft the tokens that followed them
	// (default 3).
	Match int `yaml:"match"`
}

type Pool struct {
	MaxResidentModels int `yaml:"max_resident_models"`
	// MaxPending is how many requests may wait for a slot of one model
	// before the next is refused with 503 queue_full; 0 is no limit.
	MaxPending int `yaml:"max_pending"`
}

type Cache struct {
	Prompt bool `yaml:"prompt"`
	// MaxSnapshots bounds state snapshots per slot. Only the Qwen3.5 fallback
	// takes them: its recurrent state cannot be trimmed to a prefix.
	MaxSnapshots int `yaml:"max_snapshots"`

	// PromptDisk saves a slot's state (the KV cache of a prompt that has just
	// been prefilled) to PromptDir and restores it when a request starts with
	// that prompt as its prefix and no slot holds it: after a restart, or
	// after the slot was reused for something else (OPT-13). Off by default;
	// when off nothing is created or read. It needs Prompt: a request that
	// opts out of the prompt cache opts out of the disk too.
	PromptDisk bool `yaml:"prompt_disk"`
	// PromptDir is where the states go; empty means <data_dir>/promptcache
	// (PromptCachePath).
	PromptDir string `yaml:"prompt_dir"`
	// PromptMaxBytes caps the directory. The least recently used states are
	// deleted past it.
	PromptMaxBytes int64 `yaml:"prompt_max_bytes"`
	// PromptMinTokens is the shortest prompt worth a state on disk, and a
	// restore must save at least half of it over what a slot already holds.
	PromptMinTokens int `yaml:"prompt_min_tokens"`
}

// Context overflow policies.
const (
	OverflowError          = "error"
	OverflowTruncateMiddle = "truncate_middle"
)

type Generation struct {
	EnableThinking  bool   `yaml:"enable_thinking"`
	MaxTokensCap    int    `yaml:"max_tokens_cap"`
	ContextOverflow string `yaml:"context_overflow"`
}

type Queue struct {
	Enabled    bool `yaml:"enabled"`
	MaxPending int  `yaml:"max_pending"`
	// Concurrency is how many lines of the running job are in flight; the
	// scheduler batches them on the model's slots.
	Concurrency int `yaml:"concurrency"`
	// WebhookSecret signs webhook bodies: X-NanoLLM-Signature is
	// sha256=HMAC(secret, "<X-NanoLLM-Timestamp>.<body>"). Empty sends them
	// unsigned.
	WebhookSecret string `yaml:"webhook_secret"`
	// RetentionDays deletes finished jobs after this many days; 0 keeps them.
	RetentionDays int `yaml:"retention_days"`
}

type Registry struct {
	// ModelsDir holds downloaded weights; empty means <data_dir>/models. It
	// is separate so the weights can live on a different disk than the job
	// database.
	ModelsDir string `yaml:"models_dir"`
	// StrictLicense refuses models whose commercial use is unconfirmed.
	StrictLicense bool   `yaml:"strict_license"`
	CatalogURL    string `yaml:"catalog_url"`
	// Mirrors are Hugging Face compatible base URLs tried after
	// huggingface.co: <mirror>/<repo>/resolve/<revision>/<path>.
	Mirrors []string `yaml:"mirrors"`
}

// ModelsPath is where downloaded weights live.
func (c Config) ModelsPath() string {
	if c.Registry.ModelsDir != "" {
		return c.Registry.ModelsDir
	}
	return filepath.Join(c.Server.DataDir, "models")
}

// PromptCachePath is where the prompt cache on disk keeps its states.
func (c Config) PromptCachePath() string {
	if c.Cache.PromptDir != "" {
		return c.Cache.PromptDir
	}
	return filepath.Join(c.Server.DataDir, "promptcache")
}

// Bench is the public benchmark client (docs/BENCHMARKS.md). Nothing is
// ever sent without the person seeing it first, unless they pass --submit.
type Bench struct {
	// Submit is ask (offer after a run) or never.
	Submit    string `yaml:"submit"`
	SubmitURL string `yaml:"submit_url"`
	// Anonymous sends with a throwaway key: not withdrawable, not linked to
	// this installation's other results.
	Anonymous bool `yaml:"anonymous"`
}

// UI has no path knob: the SPA lives at /ui/ (SPEC §11), and a configurable
// unauthenticated prefix is one setting away from exempting the API.
type UI struct {
	Enabled bool `yaml:"enabled"`
}

type Log struct {
	Level  string `yaml:"level"`  // debug | info | warn | error
	Format string `yaml:"format"` // json | text
	// File appends the log to this file, rolled over at 64 MiB with one
	// previous generation kept, instead of writing it to stderr. A Windows
	// service and a launchd agent have no console to write to.
	File string `yaml:"file"`
}

// DialectEnabled reports whether this dialect is mounted.
func (a API) DialectEnabled(name string) bool {
	for _, d := range a.Dialects {
		if strings.TrimSpace(d) == name {
			return true
		}
	}
	return false
}

// Override returns how a model should be loaded, or the zero override.
func (m Models) Override(id string) ModelOverride { return m.Overrides[id] }
