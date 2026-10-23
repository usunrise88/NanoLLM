package tune

import (
	"fmt"
	"math"

	"github.com/usunrise88/nanollm/internal/plan"
)

// The reference request (docs/OPTIMIZE.md, §3.4): a prompt of 4096 tokens and
// an answer of 300, "a typical answer". Every candidate's worth is the time
// this takes.
const (
	RefPromptTokens = 4096
	RefAnswerTokens = 300
)

// AnswerSeconds is the time of the reference request at the given prefill and
// decode rates (tokens per second): 4096 / pp + 300 / tg. Load time is not
// part of it.
func AnswerSeconds(pp, tg float64) float64 {
	if pp <= 0 || tg <= 0 {
		return math.Inf(1)
	}
	return RefPromptTokens/pp + RefAnswerTokens/tg
}

// Estimate is what a candidate is expected to do before it is measured.
type Estimate struct {
	// Known says there was a baseline to estimate from; without one the
	// candidate has to be measured.
	Known bool `json:"known"`
	// PP and TG are tokens per second at the depth of the reference request:
	// the prefill rate averaged over the prompt, and decode after it.
	PP float64 `json:"pp,omitempty"`
	TG float64 `json:"tg,omitempty"`
	// AnswerS is the time of the reference request in seconds.
	AnswerS float64 `json:"answer_s,omitempty"`
	// DiskTokS is the decode rate a disk allows when the setup reads weights
	// from it on every token; 0 when the setup reads none.
	DiskTokS float64 `json:"disk_tok_s,omitempty"`
	// DiskBound says the disk, not the processor, sets TG.
	DiskBound bool `json:"disk_bound,omitempty"`
	// Basis says what the numbers were taken from.
	Basis string `json:"basis,omitempty"`
}

// DiskTokS is the decode rate when every token reads coldBytes from a disk
// that sustains diskMiBs: tg = disk speed / cold bytes per token
// (docs/AUTOTUNE.md, the disk formula). Zero when coldBytes is zero.
func DiskTokS(diskMiBs float64, coldBytes int64) float64 {
	if coldBytes <= 0 || diskMiBs <= 0 {
		return 0
	}
	return diskMiBs * float64(MiB) / float64(coldBytes)
}

// ColdBytesPerToken is the weight bytes a plan leaves in the file and reads on
// every token: the layer ranges the streamer follows.
func ColdBytesPerToken(p *plan.Plan) int64 {
	var n int64
	for _, r := range p.StreamRanges {
		n += r.Length
	}
	return n
}

// Scaling of a CPU baseline to another thread count: prefill is limited by
// arithmetic and follows the threads; decoding is limited by memory bandwidth
// and follows them weakly, by exponents each baseline carries
// (Baseline.PPThreadExp, Baseline.TGThreadExp).
const (
	// DefaultPPThreadExp and DefaultTGThreadExp are the exponents of a baseline
	// that has none.
	DefaultPPThreadExp = 1.0
	DefaultTGThreadExp = 0.4
)

// EstimateFor estimates a candidate from the matrix's baselines. mtp says the
// candidate runs the drafter, diskMiBs the sequential read rate of the model's
// disk (0 when unknown).
func (m *Matrix) EstimateFor(c Class, device string, threads int, mtp bool, p *plan.Plan, diskMiBs float64) Estimate {
	b, dist, ok := m.BaselineFor(device, c)
	if !ok {
		return Estimate{}
	}
	pp, tg := b.PP4096(), b.TG4096()
	if device == "cpu" && b.Threads > 0 && threads > 0 && threads != b.Threads {
		r := float64(threads) / float64(b.Threads)
		ppExp, tgExp := b.PPThreadExp, b.TGThreadExp
		if ppExp == 0 {
			ppExp = DefaultPPThreadExp
		}
		if tgExp == 0 {
			tgExp = DefaultTGThreadExp
		}
		pp *= math.Pow(r, ppExp)
		tg *= math.Pow(r, tgExp)
	}
	if mtp && b.MTPGain > 1 {
		tg *= b.MTPGain
	}
	e := Estimate{Known: true, PP: pp, TG: tg}
	e.Basis = fmt.Sprintf("%s baseline (distance %d)", device, dist)
	if b.Estimated {
		e.Basis += ", depth factors estimated"
	}
	if cold := ColdBytesPerToken(p); cold > 0 {
		e.DiskTokS = DiskTokS(diskMiBs, cold)
		if e.DiskTokS > 0 && e.DiskTokS < e.TG {
			e.TG, e.DiskBound = e.DiskTokS, true
		}
	}
	e.AnswerS = AnswerSeconds(e.PP, e.TG)
	return e
}
