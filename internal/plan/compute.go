package plan

import (
	"bufio"
	"io"
	"regexp"
	"strconv"

	"github.com/usunrise88/nanollm/internal/gguf"
)

// ComputeCoeffs estimate a backend's compute buffer (G7) from the shape of a
// batch. llama.cpp sizes the buffer by reserving a worst-case graph, and the
// allocator reuses memory between tensors that are not alive together, so
// the buffer is the larger of two peaks plus what lives through both:
//
//	tail   logits + n_ubatch · 4 · (n_ff + n_embd_per_layer·n_layer + Embd·n_embd + OutEmbd·n_embd_out)
//	middle n_ubatch · 4 · MidFF · n_ff                        (an FFN block in flight)
//	       + KQ · n_head · n_kv · n_ubatch · 4                (without flash attention)
//	mask   n_kv · n_ubatch · MaskFA                           (with flash attention)
//	       n_kv · n_ubatch · MaskNoFA + MaskNoFABase          (without)
//
// where logits = n_vocab · min(n_ubatch, n_outputs_max) · 4. With the engine
// default n_outputs_max = n_batch the logits dominate (512 MiB for E4B at
// n_ubatch 512); with n_outputs_max = slots·(1+n_draft) the middle does.
type ComputeCoeffs struct {
	Embd    float64 `json:"embd"`
	OutEmbd float64 `json:"out_embd"`
	MidFF   float64 `json:"mid_ff"`
	// KQ is the share of the f32 KQ matrix that is alive at the middle peak
	// when flash attention is off.
	KQ float64 `json:"kq"`
	// MaskFA and MaskNoFA are bytes per KV cell per ubatch token.
	MaskFA       float64 `json:"mask_fa"`
	MaskNoFA     float64 `json:"mask_no_fa"`
	MaskNoFABase int64   `json:"mask_no_fa_base"`
	// Calibrated is set when the coefficients were fitted to this backend's
	// own `compute buffer size` log lines; otherwise they are the CPU ones.
	Calibrated bool `json:"calibrated"`
}

// cpuCoeffs were fitted on the stage-1 VM to b11600 with E4B and its drafter
// on the CPU backend (docs/STAGE1.md 1.3): llama-completion with the default
// n_outputs_max at n_ubatch 128/256/512, n_ctx 4096/32768, one and four
// sequences, with and without flash attention; llama-server with MTP
// (n_outputs_max 4 and 16) for the middle peak and the drafter context.
var cpuCoeffs = ComputeCoeffs{
	Embd: 1.21, OutEmbd: 3.1, MidFF: 5.4, KQ: 0.6,
	MaskFA: 2, MaskNoFA: 5.86, MaskNoFABase: 5_531_000,
	Calibrated: true,
}

// Coefficients holds the compute-buffer coefficients per backend kind. GPU
// backends start from the CPU fit with Calibrated false: their graphs differ
// in fusion and scratch, and stages 2–3 replace these from their logs with
// Calibrate.
var Coefficients = map[string]ComputeCoeffs{
	BackendCPU:    cpuCoeffs,
	BackendCUDA:   uncalibrated(cpuCoeffs),
	BackendVulkan: uncalibrated(cpuCoeffs),
	BackendMetal:  uncalibrated(cpuCoeffs),
}

func uncalibrated(c ComputeCoeffs) ComputeCoeffs { c.Calibrated = false; return c }

// coeffsFor never fails: an unknown backend plans with the CPU fit, marked
// uncalibrated.
func coeffsFor(backend string) ComputeCoeffs {
	if c, ok := Coefficients[backend]; ok {
		return c
	}
	return uncalibrated(cpuCoeffs)
}

// ComputeShape is the batch a compute buffer is reserved for.
type ComputeShape struct {
	NUBatch     int
	NOutputsMax int
	// NKV is the cells of the widest attention, the full-attention cache.
	NKV            int
	FlashAttention bool
}

// ComputeBytes estimates one context's compute buffer.
func (c ComputeCoeffs) ComputeBytes(h gguf.HParams, s ComputeShape) int64 {
	p := c.peaks(h, s)
	return max(p.tail, p.mid) + p.mask
}

type peaks struct {
	logits, tail, mid, mask int64
	// tailFixed and midFixed are the peaks without their fitted activation
	// term; perEmbd and perFF are what one unit of Embd and of MidFF add.
	tailFixed, midFixed, perEmbd, perFF float64
}

func (c ComputeCoeffs) peaks(h gguf.HParams, s ComputeShape) peaks {
	ub := float64(s.NUBatch)
	nOut := min(int64(s.NUBatch), int64(max(s.NOutputsMax, 1)))
	var p peaks
	p.logits = int64(h.VocabSize) * nOut * 4
	ff, heads := 0, 0
	for il := range h.BlockCount {
		ff = max(ff, h.FF(il))
		heads = max(heads, h.Heads(il))
	}
	p.tailFixed = float64(p.logits) + ub*4*(float64(ff)+float64(h.EmbeddingLengthPerLayerInput*h.BlockCount)+
		c.OutEmbd*float64(h.EmbeddingLengthOut))
	p.perEmbd = ub * 4 * float64(h.EmbeddingLength)
	p.perFF = ub * 4 * float64(ff)
	cells := float64(s.NKV) * ub
	if s.FlashAttention {
		p.mask = int64(c.MaskFA * cells)
	} else {
		p.midFixed = c.KQ * float64(heads) * cells * 4
		p.mask = int64(c.MaskNoFA*cells) + c.MaskNoFABase
	}
	p.tail = int64(p.tailFixed + c.Embd*p.perEmbd)
	p.mid = int64(p.midFixed + c.MidFF*p.perFF)
	return p
}

// Calibrate fits the activation coefficient of whichever peak dominates this
// shape to one measured compute buffer — the `compute buffer size` line
// libllama logs — and marks the result calibrated. It is the hook stages 2–3
// use for GPU backends (docs/STAGE1.md 1.4): measure once per backend at the
// planned n_outputs_max and once at the default, store the coefficients,
// plan from them.
func (c ComputeCoeffs) Calibrate(h gguf.HParams, s ComputeShape, measured int64) ComputeCoeffs {
	out := c
	out.Calibrated = true
	p := c.peaks(h, s)
	peak := float64(measured - p.mask)
	switch {
	case p.tail >= p.mid && p.perEmbd > 0:
		out.Embd = max((peak-p.tailFixed)/p.perEmbd, 0)
	case p.perFF > 0:
		out.MidFF = max((peak-p.midFixed)/p.perFF, 0)
	}
	return out
}

var computeLine = regexp.MustCompile(`(\S+) compute buffer size = +([0-9.]+) MiB`)

// ParseComputeBuffers reads libllama log output and returns the compute
// buffer sizes in the order they were reported, per buffer type ("CPU",
// "CUDA0", "Vulkan0"), in bytes. A process with a drafter reports the target
// context first.
func ParseComputeBuffers(r io.Reader) (map[string][]int64, error) {
	out := map[string][]int64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		m := computeLine.FindStringSubmatch(sc.Text())
		if m == nil {
			continue
		}
		mib, err := strconv.ParseFloat(m[2], 64)
		if err != nil {
			continue
		}
		out[m[1]] = append(out[m[1]], int64(mib*(1<<20)))
	}
	return out, sc.Err()
}
