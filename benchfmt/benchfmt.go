// Package benchfmt is the nanollm.bench/v1 result format: what `nanollm bench`
// writes, what it submits to the public benchmark, and what the benchmark
// service stores and aggregates (docs/BENCHMARKS.md).
//
// It is its own module with no dependencies, because two programs that will
// live in different repositories share it: NanoLLM and the service. Changes to
// v1 are additive only; a field is never renamed, retyped or given a new
// meaning, and an unknown field is kept by whoever reads it.
package benchfmt

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// Schema is the format identifier every document carries.
const Schema = "nanollm.bench/v1"

// Result is one benchmark run on one machine.
type Result struct {
	Schema         string  `json:"schema"`
	ConsentVersion string  `json:"consent_version,omitempty"`
	Suite          Suite   `json:"suite"`
	Client         Client  `json:"client"`
	Model          Model   `json:"model"`
	Machine        Machine `json:"machine"`
	Runs           []Run   `json:"runs"`
	// Date is the day of the run, without the time: coarser is enough to
	// order results and says less about the person who ran it.
	Date      string  `json:"date"`
	DurationS float64 `json:"duration_s"`
	// InstallID is the public half of the installation's ed25519 key, hex;
	// Signature signs the document with it (see Sign). Both are empty in an
	// anonymous submission's key-less form and in local results.
	InstallID string `json:"install_id,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// Suite identifies what was measured. Results of different suite versions
// are never compared.
type Suite struct {
	ID      string `json:"id"`      // standard | dev
	Version string `json:"version"` // v1
	// PromptsSHA256 hashes the built-in prompts and generation parameters, so
	// a client that drifted from the suite cannot pass as running it.
	PromptsSHA256 string `json:"prompts_sha256"`
	Quick         bool   `json:"quick,omitempty"`
}

type Client struct {
	NanoLLM  string   `json:"nanollm"`
	LlamaCpp LlamaCpp `json:"llama_cpp"`
	// GGMLCPUVariant is the CPU backend library ggml chose (haswell, zen4,
	// alderlake, …): two machines on one CPU model can run different
	// kernels, and the numbers follow the kernel.
	GGMLCPUVariant string `json:"ggml_cpu_variant,omitempty"`
	OS             string `json:"os"`
	Arch           string `json:"arch"`
}

// FakeCPUVariant marks a result measured on NanoLLM's scripted fake
// backend: its numbers are not a machine's, so the public service refuses
// it outside test mode.
const FakeCPUVariant = "fake"

type LlamaCpp struct {
	Tag    string `json:"tag"`
	Commit string `json:"commit"`
}

type Model struct {
	ID            string `json:"id"`
	Revision      string `json:"revision"`
	SHA256        string `json:"sha256"`
	DrafterSHA256 string `json:"drafter_sha256,omitempty"`
}

// Machine is the hardware profile: no serial numbers, UUIDs, MAC addresses,
// hostnames or user names, ever.
type Machine struct {
	CPU    CPU     `json:"cpu"`
	Memory Memory  `json:"memory"`
	GPUs   []GPU   `json:"gpus,omitempty"`
	System System  `json:"system"`
	OS     OSInfo  `json:"os"`
	Power  Power   `json:"power"`
	Disk   Disk    `json:"disk"`
	Extra  []Field `json:"extra,omitempty"`
}

type CPU struct {
	Vendor  string   `json:"vendor"`
	Model   string   `json:"model"`
	Cores   int      `json:"cores"`
	Threads int      `json:"threads"`
	PCores  int      `json:"p_cores,omitempty"`
	ECores  int      `json:"e_cores,omitempty"`
	ISA     []string `json:"isa,omitempty"` // avx2, avx512f, avx_vnni, neon, …
}

type Memory struct {
	TotalBytes int64 `json:"total_bytes"`
	SpeedMTS   int   `json:"speed_mts,omitempty"`
}

type GPU struct {
	Name        string `json:"name"`
	Vendor      string `json:"vendor,omitempty"`
	VRAMBytes   int64  `json:"vram_bytes,omitempty"`
	DriverMajor string `json:"driver_major,omitempty"`
	PowerLimitW int    `json:"power_limit_w,omitempty"`
	Cores       int    `json:"cores,omitempty"`
	Unified     bool   `json:"unified,omitempty"`
}

// Form factors.
const (
	FormLaptop  = "laptop"
	FormDesktop = "desktop"
	FormServer  = "server"
	FormVM      = "vm"
	FormUnknown = "unknown"
)

type System struct {
	Vendor string `json:"vendor,omitempty"`
	// Family is the marketing name where the firmware has one (SMBIOS
	// SystemFamily / SystemVersion, "ThinkPad X1 Carbon Gen 11"); Product is
	// the SKU-ish product name, often a part number.
	Family     string `json:"family,omitempty"`
	Product    string `json:"product,omitempty"`
	FormFactor string `json:"form_factor"`
	Chassis    string `json:"chassis,omitempty"`
	// UserEntered marks a model name the user typed (WSL2 without interop),
	// which the service moderates before showing.
	UserEntered bool `json:"user_entered,omitempty"`
}

type OSInfo struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
	// KernelMM is the kernel's major.minor only.
	KernelMM string `json:"kernel_mm,omitempty"`
	WSL      bool   `json:"wsl,omitempty"`
}

type Power struct {
	Source  string `json:"source,omitempty"` // ac | battery
	Profile string `json:"profile,omitempty"`
}

type Disk struct {
	Type string `json:"type,omitempty"` // nvme | ssd | hdd | unknown
}

// Field is an additive escape hatch for profile details v1 did not foresee.
type Field struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// Run statuses. skipped and refused are not failures: a preset the planner
// refuses on this machine is a result.
const (
	StatusOK      = "ok"
	StatusSkipped = "skipped"
	StatusRefused = "refused"
	StatusFailed  = "failed"
)

// Run is one preset's measurements.
type Run struct {
	Preset   string  `json:"preset"`
	PlanHash string  `json:"plan_hash,omitempty"`
	Plan     Plan    `json:"plan"`
	Metrics  Metrics `json:"metrics"`
	Reps     int     `json:"reps"`
	Output   Output  `json:"output"`
	Status   string  `json:"status"`
	Reason   string  `json:"reason,omitempty"`
	// Variant names what a development sweep changed against the preset
	// ("threads=8,kv=q8_0"); empty for the preset as planned.
	Variant string `json:"variant,omitempty"`
	// SameAs names the preset whose run this one shares because their plans
	// are identical (deduplicated by plan hash).
	SameAs string `json:"same_as,omitempty"`
}

type Plan struct {
	Mode        string `json:"mode,omitempty"`
	Device      string `json:"device,omitempty"`
	GPULayers   int    `json:"gpu_layers"`
	NCtx        int    `json:"n_ctx,omitempty"`
	Slots       int    `json:"slots,omitempty"`
	KV          string `json:"kv,omitempty"`
	FA          bool   `json:"fa"`
	SWAFull     bool   `json:"swa_full"`
	MTP         bool   `json:"mtp"`
	Threads     int    `json:"threads,omitempty"`
	Enforcement string `json:"enforcement,omitempty"` // strict | advisory

	// What the run loaded with and where it ran. All of these are omitted
	// when unset, so a v1 document without them still verifies.
	ThreadsBatch int     `json:"threads_batch,omitempty"`
	NBatch       int     `json:"n_batch,omitempty"`
	NUBatch      int     `json:"n_ubatch,omitempty"`
	KVV          string  `json:"kv_v,omitempty"`
	Repack       string  `json:"repack,omitempty"`    // on | off
	LoadMode     string  `json:"load_mode,omitempty"` // none | mmap | mlock | mmap+mlock | dio
	Lazy         string  `json:"lazy,omitempty"`      // off | auto | on
	DraftMax     int     `json:"draft_max,omitempty"`
	DraftPMin    float64 `json:"draft_p_min,omitempty"`
	// NUMA is the memory policy the worker ran under (off, interleave,
	// node:N); CPUs the cpu list it was pinned to, when it was.
	NUMA string `json:"numa,omitempty"`
	CPUs string `json:"cpus,omitempty"`
	// Cold means the model file was evicted from the page cache first.
	Cold bool `json:"cold,omitempty"`
}

// Metrics are medians over Reps; StdDev holds the spread of the main ones.
type Metrics struct {
	LoadMS   float64 `json:"load_ms,omitempty"`
	PPTokS   float64 `json:"pp_tok_s,omitempty"`
	TGTokS   TG      `json:"tg_tok_s"`
	TTFTMS   float64 `json:"ttft_ms,omitempty"`
	ChatTokS float64 `json:"chat_tok_s,omitempty"`
	MTP      *MTP    `json:"mtp,omitempty"`
	// Echo is drafting on a task that repeats its input (dev suite): the
	// best case of prompt lookup, next to the chat-ru worst case.
	Echo          *MTP  `json:"echo,omitempty"`
	PeakRSS       int64 `json:"peak_rss,omitempty"`
	PeakVRAM      int64 `json:"peak_vram,omitempty"`
	PeakLocked    int64 `json:"peak_locked,omitempty"`
	PeakPageCache int64 `json:"peak_page_cache,omitempty"`
	// PeakCgroup is the memory.peak of the worker's strict cgroup: resident
	// pages plus the page cache charged to it, which is what the budget
	// limits.
	PeakCgroup int64 `json:"peak_cgroup,omitempty"`
	// Buffers is the weights by ggml buffer type (CPU_Mapped, CPU_REPACK,
	// Vulkan0, …), from the load log.
	Buffers map[string]int64   `json:"buffers,omitempty"`
	StdDev  map[string]float64 `json:"stddev,omitempty"`
}

// TG is token generation at depth 0 and at depth N (DepthN tokens of
// context already in the cache).
type TG struct {
	D0     float64 `json:"d0,omitempty"`
	DN     float64 `json:"dN,omitempty"`
	DepthN int     `json:"depth_n,omitempty"`
}

type MTP struct {
	Acceptance float64 `json:"acceptance"`
	Speedup    float64 `json:"speedup"`
	// TokS is the chat-ru decode rate with drafting on, absolute: Speedup is
	// a ratio of two noisy numbers and hides which one moved.
	TokS float64 `json:"tok_s,omitempty"`
}

// Output fingerprints what the model generated under greedy decoding: two
// runs of one cohort (backend, CPU variant, repack, tag) normally agree, and
// a disagreement is a soft signal for the service, never a rejection.
type Output struct {
	SHA256 string `json:"sha256,omitempty"`
	Tokens int    `json:"tokens,omitempty"`
	// Prefixes hash the first Tokens tokens of the output at a few
	// checkpoints (16, 32, 64, …), so the service can tell how long a prefix
	// two runs share rather than only whether they agree to the end: greedy
	// decoding on different kernels drifts late, not early. Optional.
	Prefixes []Prefix `json:"prefixes,omitempty"`
}

// Prefix is the sha256 of the output's first Tokens token ids.
type Prefix struct {
	Tokens int    `json:"tokens"`
	SHA256 string `json:"sha256"`
}

var (
	hex64    = regexp.MustCompile(`^[0-9a-f]{64}$`)
	dateRe   = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
	presetRe = regexp.MustCompile(`^[a-z0-9.+-]{1,32}$`)
)

// Limits on sizes and plausible ranges, shared by the client (which refuses
// to write nonsense) and the service (which refuses to store it).
const (
	MaxRuns      = 64
	MaxTextField = 128
	MaxTokS      = 100000
	MaxPrefixes  = 16
)

// Validate checks a document against v1: required fields, formats and ranges.
// It does not judge plausibility against the hardware; that is the service's.
func (r *Result) Validate() error {
	var errs []error
	bad := func(format string, args ...any) { errs = append(errs, fmt.Errorf(format, args...)) }
	if r.Schema != Schema {
		bad("schema is %q, want %q", r.Schema, Schema)
	}
	if r.Suite.ID == "" || r.Suite.Version == "" {
		bad("suite.id and suite.version are required")
	}
	if r.Suite.PromptsSHA256 != "" && !hex64.MatchString(r.Suite.PromptsSHA256) {
		bad("suite.prompts_sha256 is not a sha256")
	}
	if r.Client.NanoLLM == "" || r.Client.LlamaCpp.Tag == "" || r.Client.OS == "" || r.Client.Arch == "" {
		bad("client.nanollm, client.llama_cpp.tag, client.os and client.arch are required")
	}
	if r.Model.ID == "" || !hex64.MatchString(r.Model.SHA256) {
		bad("model.id and model.sha256 are required")
	}
	if r.Model.DrafterSHA256 != "" && !hex64.MatchString(r.Model.DrafterSHA256) {
		bad("model.drafter_sha256 is not a sha256")
	}
	if !dateRe.MatchString(r.Date) {
		bad("date must be YYYY-MM-DD")
	} else if _, err := time.Parse("2006-01-02", r.Date); err != nil {
		bad("date: %v", err)
	}
	if r.Machine.CPU.Model == "" || r.Machine.CPU.Threads <= 0 {
		bad("machine.cpu.model and machine.cpu.threads are required")
	}
	if r.Machine.Memory.TotalBytes <= 0 {
		bad("machine.memory.total_bytes is required")
	}
	switch r.Machine.System.FormFactor {
	case FormLaptop, FormDesktop, FormServer, FormVM, FormUnknown:
	default:
		bad("machine.system.form_factor %q is not one of laptop, desktop, server, vm, unknown", r.Machine.System.FormFactor)
	}
	for name, s := range map[string]string{
		"machine.cpu.model": r.Machine.CPU.Model, "machine.system.family": r.Machine.System.Family,
		"machine.system.product": r.Machine.System.Product, "machine.system.vendor": r.Machine.System.Vendor,
	} {
		if len(s) > MaxTextField || strings.ContainsAny(s, "\x00\n\r") {
			bad("%s is too long or contains control characters", name)
		}
	}
	if len(r.Runs) == 0 || len(r.Runs) > MaxRuns {
		bad("runs: between 1 and %d", MaxRuns)
	}
	for i, run := range r.Runs {
		if !presetRe.MatchString(run.Preset) {
			bad("runs[%d].preset %q is not a preset name", i, run.Preset)
		}
		switch run.Status {
		case StatusOK:
			if run.Reps <= 0 {
				bad("runs[%d]: an ok run has reps", i)
			}
		case StatusSkipped, StatusRefused, StatusFailed:
		default:
			bad("runs[%d].status %q is not ok, skipped, refused or failed", i, run.Status)
		}
		m := run.Metrics
		for name, v := range map[string]float64{"pp_tok_s": m.PPTokS, "tg_tok_s.d0": m.TGTokS.D0, "tg_tok_s.dN": m.TGTokS.DN, "chat_tok_s": m.ChatTokS} {
			if v < 0 || v > MaxTokS {
				bad("runs[%d].metrics.%s = %g is out of range", i, name, v)
			}
		}
		if m.MTP != nil && (m.MTP.Acceptance < 0 || m.MTP.Acceptance > 1 || m.MTP.Speedup < 0 || m.MTP.Speedup > 10) {
			bad("runs[%d].metrics.mtp is out of range", i)
		}
		if run.Output.SHA256 != "" && !hex64.MatchString(run.Output.SHA256) {
			bad("runs[%d].output.sha256 is not a sha256", i)
		}
		if len(run.Output.Prefixes) > MaxPrefixes {
			bad("runs[%d].output.prefixes: at most %d", i, MaxPrefixes)
		}
		for j, p := range run.Output.Prefixes {
			if p.Tokens <= 0 || !hex64.MatchString(p.SHA256) {
				bad("runs[%d].output.prefixes[%d] needs tokens > 0 and a sha256", i, j)
			}
		}
	}
	return errors.Join(errs...)
}

// Parse reads and validates a document.
func Parse(b []byte) (*Result, error) {
	var r Result
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("benchfmt: %w", err)
	}
	if err := r.Validate(); err != nil {
		return nil, fmt.Errorf("benchfmt: %w", err)
	}
	return &r, nil
}
