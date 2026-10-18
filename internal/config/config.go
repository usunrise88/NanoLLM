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
	DialectOllama = "ollama" // v2
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

// ModelOverride is how one model is loaded. Every field is optional; an empty
// one means the planner or the manifest decides.
type ModelOverride struct {
	// Preset is a hard memory budget from SPEC §4 (plan.Presets).
	Preset string `yaml:"preset"`
	// Mode is auto (the planner picks within the preset), a mode name or its
	// short ID, M1…M10.
	Mode   string `yaml:"mode"`
	KVType string `yaml:"kv_type"`
	// NCtx and Slots take "auto": what the preset leaves after weights, down
	// to a floor of 4096 tokens and one slot.
	NCtx        Auto        `yaml:"n_ctx"`
	Slots       Auto        `yaml:"slots"`
	Speculative Speculative `yaml:"speculative"`
}

// Speculative decoding types.
const (
	SpecMTP  = "mtp"
	SpecNone = "none"
)

type Speculative struct {
	Type string `yaml:"type"`
	// NMax is the most tokens the drafter proposes per step.
	NMax int `yaml:"n_max"`
	// PMin is the drafter probability below which a draft stops early.
	PMin float64 `yaml:"p_min"`
}

type Pool struct {
	MaxResidentModels int `yaml:"max_resident_models"`
}

type Cache struct {
	Prompt bool `yaml:"prompt"`
	// MaxSnapshots bounds state snapshots per slot. Only the Qwen3.5 fallback
	// takes them: its recurrent state cannot be trimmed to a prefix.
	MaxSnapshots int `yaml:"max_snapshots"`
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
