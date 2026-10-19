package config

import (
	"os"
	"path/filepath"
	"runtime"
)

// DefaultModel is the catalog entry a fresh installation runs (SPEC §7).
const DefaultModel = "gemma-4-e4b-qat"

// Default is the configuration with no file at all: loopback only, no
// authentication, the primary model and both v1 dialects. It is what a
// developer gets from `nanollm serve` in a fresh checkout, and it validates on
// its own.
func Default() Config {
	return Config{
		Server: Server{
			Addr:    "127.0.0.1:8080",
			DataDir: DefaultDataDir(),
			MaxBody: 16 << 20,
		},
		Auth: Auth{Mode: "open"},
		API:  API{Dialects: []string{DialectOpenAI, DialectNative}},
		Backends: Backends{
			LibDir:  "./lib",
			Enabled: []string{BackendLlamaCpp},
		},
		Devices: Devices{
			Default:  DeviceAuto,
			Fallback: true,
			NUMA:     "auto",
		},
		Models: Models{
			Default: DefaultModel,
			Preload: []string{DefaultModel},
		},
		Pool:  Pool{MaxResidentModels: 1, MaxPending: 64},
		Cache: Cache{Prompt: true, MaxSnapshots: 8},
		Generation: Generation{
			// Off by default: the user waits for every token of thinking
			// before the answer starts.
			EnableThinking:  false,
			MaxTokensCap:    8192,
			ContextOverflow: OverflowError,
		},
		Queue:    Queue{Enabled: true, MaxPending: 1000, Concurrency: 4, RetentionDays: 30},
		Registry: Registry{Mirrors: []string{}},
		Bench:    Bench{Submit: "ask", SubmitURL: "https://nano.llmto.day"},
		UI:       UI{Enabled: true},
		Log:      Log{Level: "info", Format: "json"},
	}
}

// DefaultDataDir is where the installers put state on each platform (SPEC
// §10): /var/lib/nanollm, C:\ProgramData\NanoLLM, or Application Support on
// macOS.
func DefaultDataDir() string {
	switch runtime.GOOS {
	case "windows":
		if pd := os.Getenv("ProgramData"); pd != "" {
			return filepath.Join(pd, "NanoLLM")
		}
		return `C:\ProgramData\NanoLLM`
	case "darwin":
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, "Library", "Application Support", "NanoLLM")
		}
		return "/usr/local/nanollm/data"
	default:
		return "/var/lib/nanollm"
	}
}
