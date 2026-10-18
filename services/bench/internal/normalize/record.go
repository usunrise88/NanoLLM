package normalize

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"

	"github.com/usunrise88/nanollm/benchfmt"
)

// Record is one submission as the derived tables hold it.
type Record struct {
	Normalizer   int
	Date         string
	Suite        string // "standard/v1": results of different suites never mix
	ModelID      string
	ModelSHA256  string
	NanoLLM      string
	LlamaTag     string
	CPUVariant   string
	Repack       string
	CPU, CPUSlug string
	GPU, GPUSlug string
	System       string
	SystemSlug   string
	SystemVendor string
	FormFactor   string
	OS           string
	InstallID    string
	// Fingerprint is the coarse hardware fingerprint: an aggregate counts
	// one per fingerprint as well as one per install, so a fresh key per
	// run does not multiply one machine.
	Fingerprint string
	UserEntered bool
	Runs        []Run
}

// Run is one preset's row.
type Run struct {
	Preset        string
	Status        string
	Reason        string
	Mode          string
	Device        string
	Backend       string
	GPULayers     int
	MTP           bool
	KV            string
	LoadMS        float64
	PPTokS        float64
	TGD0          float64
	TGDN          float64
	TTFTMS        float64
	ChatTokS      float64
	MTPSpeedup    float64
	MTPAcceptance float64
	OutputSHA256  string
	OutputTokens  int
	Prefixes      []benchfmt.Prefix
}

// Normalize derives the record from a parsed result.
func Normalize(r *benchfmt.Result, al Aliases) Record {
	m := r.Machine
	rec := Record{
		Normalizer:  Version,
		Date:        r.Date,
		Suite:       r.Suite.ID + "/" + r.Suite.Version,
		ModelID:     r.Model.ID,
		ModelSHA256: r.Model.SHA256,
		NanoLLM:     r.Client.NanoLLM,
		LlamaTag:    r.Client.LlamaCpp.Tag,
		CPUVariant:  r.Client.GGMLCPUVariant,
		FormFactor:  m.System.FormFactor,
		OS:          OS(m.OS.Name, m.OS.Version, m.OS.WSL),
		InstallID:   r.InstallID,
		UserEntered: m.System.UserEntered,
	}
	for _, f := range m.Extra {
		if f.Key == "repack" {
			rec.Repack = f.Value
		}
	}
	rec.CPU = al.apply(KindCPU, CPU(m.CPU.Vendor, m.CPU.Model))
	rec.GPU = "none"
	if g := primaryGPU(m.GPUs); g != nil {
		rec.GPU = al.apply(KindGPU, GPU(g.Vendor, g.Name, g.Cores))
	}
	rec.SystemVendor = al.apply(KindVendor, Vendor(m.System.Vendor))
	cores := 0
	if g := primaryGPU(m.GPUs); g != nil && GPUVendor(g.Vendor, g.Name) == "Apple" {
		cores = g.Cores
	}
	sys := System(rec.SystemVendor, m.System.Family, m.System.Product, cores)
	if strings.TrimSpace(sys) == "" {
		sys = "unknown"
	}
	rec.System = al.apply(KindSystem, sys)
	rec.CPUSlug, rec.GPUSlug, rec.SystemSlug = Slug(rec.CPU), Slug(rec.GPU), Slug(rec.System)
	rec.Fingerprint = fingerprint(r, rec)
	for _, run := range r.Runs {
		nr := Run{
			Preset: run.Preset, Status: run.Status, Reason: run.Reason,
			Mode: run.Plan.Mode, Device: run.Plan.Device, Backend: Backend(run.Plan.Device),
			GPULayers: run.Plan.GPULayers, MTP: run.Plan.MTP, KV: run.Plan.KV,
			LoadMS: run.Metrics.LoadMS, PPTokS: run.Metrics.PPTokS,
			TGD0: run.Metrics.TGTokS.D0, TGDN: run.Metrics.TGTokS.DN,
			TTFTMS: run.Metrics.TTFTMS, ChatTokS: run.Metrics.ChatTokS,
			OutputSHA256: run.Output.SHA256, OutputTokens: run.Output.Tokens, Prefixes: run.Output.Prefixes,
		}
		if run.Metrics.MTP != nil {
			nr.MTPSpeedup, nr.MTPAcceptance = run.Metrics.MTP.Speedup, run.Metrics.MTP.Acceptance
		}
		rec.Runs = append(rec.Runs, nr)
	}
	return rec
}

// primaryGPU is the discrete GPU if there is one, else the first.
func primaryGPU(gs []benchfmt.GPU) *benchfmt.GPU {
	for i := range gs {
		if !gs[i].Unified {
			return &gs[i]
		}
	}
	if len(gs) > 0 {
		return &gs[0]
	}
	return nil
}

// fingerprint is deliberately coarse: what the hardware is, not which copy.
// No versions (a driver update is the same machine), memory to the GiB.
func fingerprint(r *benchfmt.Result, rec Record) string {
	m := r.Machine
	var gpus []string
	for _, g := range m.GPUs {
		gpus = append(gpus, fmt.Sprintf("%s/%d", strings.ToLower(GPU(g.Vendor, g.Name, g.Cores)), (g.VRAMBytes+1<<29)>>30))
	}
	sort.Strings(gpus)
	parts := []string{
		strings.ToLower(rec.CPU), fmt.Sprint(m.CPU.Threads),
		fmt.Sprint((m.Memory.TotalBytes + 1<<29) >> 30), fmt.Sprint(m.Memory.SpeedMTS),
		strings.Join(gpus, ","), strings.ToLower(rec.SystemVendor), strings.ToLower(System(m.System.Vendor, m.System.Family, m.System.Product, 0)),
		m.System.FormFactor, strings.ToLower(m.OS.Name),
	}
	sum := sha256.Sum256([]byte(strings.Join(parts, "|")))
	return hex.EncodeToString(sum[:8])
}
