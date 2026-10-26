package plan

import (
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/gguf"
)

// Bytes is how much of something sits in each placement target (SPEC §4):
// device memory, RAM that is resident (anonymous or pinned), RAM that is
// file-backed and evictable, and disk read on demand by our own loader.
// On unified memory VRAM is the shared pool.
type Bytes struct {
	VRAM       int64 `json:"vram"`
	RAMLocked  int64 `json:"ram_locked"`
	FileBacked int64 `json:"file_backed"`
	Disk       int64 `json:"disk"`
}

func (b *Bytes) add(o Bytes) {
	b.VRAM += o.VRAM
	b.RAMLocked += o.RAMLocked
	b.FileBacked += o.FileBacked
	b.Disk += o.Disk
}

// Total is the bytes across all targets.
func (b Bytes) Total() int64 { return b.VRAM + b.RAMLocked + b.FileBacked + b.Disk }

// Usage is one budget: what the preset allows and what the plan takes.
// Budgeted is false for host RAM under a VRAM-only preset, which the spec
// leaves open: the plan measures and reports it instead of bounding it.
type Usage struct {
	Budget   int64 `json:"budget"`
	Used     int64 `json:"used"`
	Budgeted bool  `json:"budgeted"`
}

// Short is how far the plan is over budget; zero when it fits or the budget
// is not enforced.
func (u Usage) Short() int64 {
	if !u.Budgeted {
		return 0
	}
	return max(0, u.Used-u.Budget)
}

// Memory is the plan against the preset. Overheads are included in Used.
type Memory struct {
	VRAM    *Usage `json:"vram,omitempty"`
	RAM     *Usage `json:"ram,omitempty"`
	Unified *Usage `json:"unified,omitempty"`
	// Combined is the autotuner's limit on private memory plus video memory
	// together (Request.MemCap); absent without one.
	Combined *Usage `json:"combined,omitempty"`
	// FileBacked is page cache the plan relies on; it is reported, never
	// budgeted (SPEC §4, presets), except by strict benchmark runs.
	FileBacked int64 `json:"file_backed"`
	// OverheadVRAM and OverheadRAM are the runtime allowances inside Used.
	OverheadVRAM int64 `json:"overhead_vram,omitempty"`
	OverheadRAM  int64 `json:"overhead_ram"`
}

// Shortfall is how many bytes a mode lacked, per budget.
type Shortfall struct {
	VRAM     int64 `json:"vram,omitempty"`
	RAM      int64 `json:"ram,omitempty"`
	Unified  int64 `json:"unified,omitempty"`
	Combined int64 `json:"combined,omitempty"`
}

func (s Shortfall) zero() bool { return s == Shortfall{} }

// KV is the cache the plan sizes (G5).
type KV struct {
	TypeK string `json:"type_k"`
	TypeV string `json:"type_v"`
	// SWAFull and its reason are the prompt-cache decision of
	// docs/STAGE1.md 1.4.
	SWAFull       bool   `json:"swa_full"`
	SWAFullReason string `json:"swa_full_reason"`
	Unified       bool   `json:"unified"`
	CellsFull     int    `json:"cells_full"`
	CellsSWA      int    `json:"cells_swa"`
	Streams       int    `json:"streams"`
	Bytes         int64  `json:"bytes"`
}

// MTP is whether the drafter runs, and why.
type MTP struct {
	Enabled   bool    `json:"enabled"`
	Reason    string  `json:"reason"`
	DraftMax  int     `json:"draft_max,omitempty"`
	DraftPMin float64 `json:"draft_p_min,omitempty"`
	// VerifyRows bounds the rows of a scheduler step spent on drafts and
	// their checks over the decoding sequences; zero is no bound.
	VerifyRows int `json:"verify_rows,omitempty"`
}

// Head is the output shortlist (OPT-4, docs/AUTOTUNE.md): the head computes the
// logits of Rows tokens while no request needs the whole vocabulary. The model
// keeps a second, repacked copy of those rows (Bytes of memory).
type Head struct {
	Kind  string `json:"kind"`
	Rows  int    `json:"rows"`
	Bytes int64  `json:"bytes"`
}

// NGram is prompt-lookup drafting (OPT-5): drafts come from the sequence
// itself, no drafter model is loaded.
type NGram struct {
	Enabled  bool   `json:"enabled"`
	N        int    `json:"n,omitempty"`
	DraftMax int    `json:"draft_max,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

// Range is a byte range of one file to pin with range-lock (M7, M8). Offsets
// are absolute in the file; page alignment is the locker's job.
type Range struct {
	Role   string `json:"role"`
	Offset int64  `json:"offset"`
	Length int64  `json:"length"`
}

// StreamRange is one file-backed range read by a layer.
type StreamRange struct {
	Layer  int   `json:"layer"`
	Offset int64 `json:"offset"`
	Length int64 `json:"length"`
}

// Attempt records one mode the planner considered.
type Attempt struct {
	Mode Mode   `json:"mode"`
	ID   string `json:"id"`
	Fits bool   `json:"fits"`
	// Skipped is why the mode was not tried at all.
	Skipped string `json:"skipped,omitempty"`
	// Shortfall is what it lacked at the floor (for M4, with no layer on
	// the device).
	Shortfall *Shortfall `json:"shortfall,omitempty"`
	// LayersOnGPU is M4's N when it fit.
	LayersOnGPU int `json:"layers_on_gpu,omitempty"`
}

// Plan is a resolved placement of one model under one preset: what goes
// where, the context it leaves, and how to load it. A plan that found no
// mode carries Refused and the per-mode shortfall in Tried.
type Plan struct {
	Model  string  `json:"model"`
	Preset string  `json:"preset"`
	Device *Device `json:"device,omitempty"`

	// Layout is the tensor order of the model file the plan was made for:
	// "nano" for a file relaid by `models relayout` (OPT-7), omitted for the
	// file as published.
	Layout string `json:"layout,omitempty"`

	Mode   Mode   `json:"mode,omitempty"`
	ModeID string `json:"mode_id,omitempty"`
	// GPULayers is llama.cpp's n_gpu_layers: -1 for all, 0 for none.
	// LayersOnGPU is N, the repeating layers on the device. llama.cpp counts
	// the output layer as one more and offloads from the end, so M4 with N
	// layers is GPULayers = N+1 holding the last N.
	GPULayers   int `json:"gpu_layers"`
	LayersOnGPU int `json:"layers_on_gpu"`

	NCtx           int  `json:"n_ctx"` // per slot
	Slots          int  `json:"slots"`
	NBatch         int  `json:"n_batch"`
	NUBatch        int  `json:"n_ubatch"`
	NOutputsMax    int  `json:"n_outputs_max"`
	FlashAttention bool `json:"flash_attention"`
	// FlashAttnSmallBatch keeps flash attention for batches of up to 8 tokens
	// and runs longer ones through the matmul attention on the full-attention
	// layers (Request.FlashAttnSmallBatch).
	FlashAttnSmallBatch bool `json:"flash_attn_small_batch,omitempty"`
	// BackendSampling samples in the device graph (Request.BackendSampling).
	BackendSampling bool `json:"backend_sampling,omitempty"`
	// CudaNano are the switches of ggml-cuda set for this load (Request.CudaNano).
	CudaNano map[string]int `json:"cuda_nano,omitempty"`
	KV       KV             `json:"kv"`

	LoadMode   backend.LoadMode         `json:"load_mode,omitempty"`
	LazyMode   backend.LazyMode         `json:"lazy_mode,omitempty"`
	Repack     bool                     `json:"repack"`
	Overrides  []backend.BufferOverride `json:"overrides,omitempty"`
	LockRanges []Range                  `json:"lock_ranges,omitempty"`
	// RandomRanges are the file-backed ranges read at random (the PLE
	// table): the loader tells the kernel not to read ahead in them.
	RandomRanges []Range `json:"random_ranges,omitempty"`
	// StreamRanges are the file-backed ranges of the layers, which every
	// token reads front to back: with them the loader can read ahead and
	// evict layer by layer (OPT-8).
	StreamRanges []StreamRange `json:"stream_ranges,omitempty"`
	MTP          MTP           `json:"mtp"`
	// NGram is set only when the request asked for it.
	NGram *NGram `json:"ngram,omitempty"`
	// Head is the output shortlist the plan runs, nil for the whole head.
	Head *Head `json:"head,omitempty"`
	// headIDs are the ids of the shortlist, for LoadSpec; not part of the plan's JSON.
	headIDs []int32

	// Groups is bytes per placement group and target; G5 is the KV cache,
	// G7 the compute and output buffers of both contexts.
	Groups map[gguf.Group]Bytes `json:"groups,omitempty"`
	Memory Memory               `json:"memory"`
	// ComputeCalibrated says whether G7 comes from coefficients fitted to
	// this backend (docs/STAGE1.md 1.4).
	ComputeCalibrated bool `json:"compute_calibrated"`

	Target   Target   `json:"target"`
	Warnings []string `json:"warnings,omitempty"`
	// Sources says where each setting came from when the autotuner's mode or
	// profile had a say: config, profile, mode or rule (docs/OPTIMIZE.md, §3.7).
	Sources map[string]string `json:"sources,omitempty"`
	Tried   []Attempt         `json:"tried"`
	Refused string            `json:"refused,omitempty"`
}

// OK reports whether the plan found a mode.
func (p *Plan) OK() bool { return p.Refused == "" }
