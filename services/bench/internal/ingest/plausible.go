package ingest

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/services/bench/internal/normalize"
	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

// The plausibility ceiling bounds token generation without MTP: every
// decoded token reads the weights it touches once, so tokens per second
// cannot exceed memory bandwidth over bytes read per token. Bytes come from
// the plan's placement: G1 (the PLE table, of which one row per token is
// read) and G6 (the drafter, not used without MTP) are excluded, and a
// partial offload reads its GPU layers at GPU bandwidth and the rest at host
// bandwidth. KV reads are left out, which only raises the ceiling.
//
// Unknown hardware or an unknown model gives no ceiling, and the run is
// marked "unknown", never refused. The tables are deliberately generous:
// the ceiling exists to catch fabrications, not to grade a machine.

// ModelCost is what one decoded token reads.
type ModelCost struct {
	Layers int
	// LayerBytes is the repeating layers' weights (G2, G3, G4 and the PLE
	// projections), OutputBytes the output projection (G0; tied to
	// token_embd in Gemma 4, read in full per token).
	LayerBytes, OutputBytes int64
}

// Models are the catalog models the ceiling knows, by model id. Measured
// from the GGUF header (docs/STAGE1.md 1.3).
var Models = map[string]ModelCost{
	"gemma-4-e4b-qat": {Layers: 42, LayerBytes: 330301440 + 46448640 + 1857945600 + 2241704, OutputBytes: 377487360},
}

// bw is a bandwidth table row: a pattern over the lower-case normalized
// name and the peak in GB/s.
type bw struct {
	re  *regexp.Regexp
	gbs float64
}

func table(rows ...any) []bw {
	var out []bw
	for i := 0; i < len(rows); i += 2 {
		out = append(out, bw{regexp.MustCompile(rows[i].(string)), rows[i+1].(float64)})
	}
	return out
}

// hostBandwidth is the memory bandwidth of SoCs and server CPUs whose bus
// width a memory speed does not tell, first match wins.
var hostBandwidth = table(
	`^apple m1$`, 68.25, `^apple m1 pro$`, 200.0, `^apple m1 max$`, 400.0, `^apple m1 ultra$`, 800.0,
	`^apple m2$`, 100.0, `^apple m2 pro$`, 200.0, `^apple m2 max$`, 400.0, `^apple m2 ultra$`, 800.0,
	`^apple m3$`, 100.0, `^apple m3 pro$`, 150.0, `^apple m3 max$`, 400.0, `^apple m3 ultra$`, 819.0,
	`^apple m4$`, 120.0, `^apple m4 pro$`, 273.0, `^apple m4 max$`, 546.0, `^apple m5$`, 153.0,
	`ryzen ai max`, 256.0,
	`ryzen ai (9|7|5)`, 120.0,
	`core ultra \d (2|3)\d\dv`, 136.5, // Lunar Lake, on-package LPDDR5X-8533
	`epyc 9\d\d5`, 614.4*2, // Turin, 12 × DDR5-6400 per socket, two sockets
	`epyc 9\d\d4`, 460.8*2, // Genoa, 12 × DDR5-4800
	`epyc 8\d\d4`, 230.4*2, // Siena, 6 × DDR5-4800
	`epyc 7\d\d3`, 204.8*2, // Milan, 8 × DDR4-3200
	`epyc 7\d\d2`, 204.8*2,
	// Naples (7001), 8 × DDR4-2666 per socket. The name does not tell the
	// sockets, so all but the P parts, which are single-socket, count two.
	// This is the peak of the whole socket, not what is reached: a 7351P has
	// 4 NUMA nodes at 36.7 GB/s each, and llama.cpp, which does not place
	// weights per node, decodes E4B at about 20 tok/s on it.
	`epyc 7\d\d1p`, 170.7,
	`epyc 7\d\d1`, 170.7*2,
	`threadripper pro`, 409.6, // 8 × DDR5-6400
	`threadripper`, 204.8,
	`xeon`, 614.4*2,
)

// gpuBandwidth is VRAM bandwidth by normalized GPU name.
var gpuBandwidth = table(
	`rtx 5090 laptop`, 896.0, `rtx 5080 laptop`, 896.0, `rtx 5070 ti laptop`, 672.0, `rtx 5070 laptop`, 384.0, `rtx 5060 laptop`, 448.0,
	`rtx 5090`, 1792.0, `rtx 5080`, 960.0, `rtx 5070 ti`, 896.0, `rtx 5070`, 672.0, `rtx 5060 ti`, 448.0, `rtx 5060`, 448.0,
	`rtx 4090 laptop`, 576.0, `rtx 4080 laptop`, 432.0, `rtx 4070 laptop`, 256.0, `rtx 4060 laptop`, 256.0, `rtx 4050 laptop`, 192.0,
	`rtx 4090`, 1008.0, `rtx 4080 super`, 736.0, `rtx 4080`, 717.0, `rtx 4070 ti super`, 672.0, `rtx 4070 ti`, 504.0,
	`rtx 4070 super`, 504.0, `rtx 4070`, 504.0, `rtx 4060 ti`, 288.0, `rtx 4060`, 272.0,
	`rtx 3080 ti laptop`, 512.0, `rtx 3080 laptop`, 448.0, `rtx 3070 ti laptop`, 448.0, `rtx 3070 laptop`, 448.0,
	`rtx 3060 laptop`, 336.0, `rtx 3050 ti laptop`, 192.0, `rtx 3050 laptop`, 192.0,
	`rtx 3090 ti`, 1008.0, `rtx 3090`, 936.0, `rtx 3080 ti`, 912.0, `rtx 3080`, 912.0, `rtx 3070 ti`, 608.0,
	`rtx 3070`, 448.0, `rtx 3060 ti`, 448.0, `rtx 3060`, 360.0, `rtx 3050`, 224.0,
	`rtx 2080 ti`, 616.0, `rtx 2080`, 448.0, `rtx 2070`, 448.0, `rtx 2060`, 336.0, `gtx 1080 ti`, 484.0, `gtx 1660`, 336.0,
	`h200`, 4800.0, `h100`, 3350.0, `a100`, 2039.0, `l40`, 864.0, `a10\b`, 600.0, `l4\b`, 300.0, `t4\b`, 320.0, `a6000`, 960.0,
	`rx 9070 xt`, 640.0, `rx 9070`, 640.0, `rx 7900 xtx`, 960.0, `rx 7900 xt`, 800.0, `rx 7900 gre`, 576.0, `rx 7800 xt`, 624.0,
	`rx 7700 xt`, 432.0, `rx 7600`, 288.0, `rx 6900 xt`, 512.0, `rx 6800`, 512.0, `rx 6700 xt`, 384.0, `rx 6600`, 224.0,
	`arc a770`, 560.0, `arc a750`, 512.0, `arc b580`, 456.0,
)

func lookup(t []bw, name string) float64 {
	n := strings.ToLower(name)
	for _, r := range t {
		if r.re.MatchString(n) {
			return r.gbs
		}
	}
	return 0
}

var appleChip = regexp.MustCompile(`^(apple m\d( pro| max| ultra)?)`)

// HostBandwidth is the machine's RAM bandwidth in GB/s, 0 when unknown: the
// table for SoCs and servers, else the memory speed over a bus as wide as
// the form factor allows (laptop 256-bit, desktop 512-bit). A VM or an
// unknown form factor with no table entry is unknown.
func HostBandwidth(m benchfmt.Machine, cpu string) float64 {
	if v := lookup(hostBandwidth, cpu); v > 0 {
		return v
	}
	if m.Memory.SpeedMTS <= 0 {
		return 0
	}
	var bytesWide float64
	switch m.System.FormFactor {
	case benchfmt.FormLaptop:
		bytesWide = 32
	case benchfmt.FormDesktop:
		bytesWide = 64
	default:
		return 0
	}
	return float64(m.Memory.SpeedMTS) * bytesWide / 1000
}

// DeviceBandwidth is the bandwidth of the plan's device in GB/s, 0 when
// unknown. A unified GPU reads host memory.
func DeviceBandwidth(m benchfmt.Machine, device, cpu string) float64 {
	g := deviceGPU(m.GPUs, device)
	if g == nil {
		return 0
	}
	name := normalize.GPU(g.Vendor, g.Name, 0)
	if g.Unified || normalize.GPUVendor(g.Vendor, g.Name) == "Apple" {
		if c := appleChip.FindString(strings.ToLower(name)); c != "" {
			return lookup(hostBandwidth, c)
		}
		return HostBandwidth(m, cpu)
	}
	return lookup(gpuBandwidth, name)
}

func deviceGPU(gs []benchfmt.GPU, device string) *benchfmt.GPU {
	if len(gs) == 0 {
		return nil
	}
	d := strings.TrimSpace(device)
	i := len(d)
	for i > 0 && d[i-1] >= '0' && d[i-1] <= '9' {
		i--
	}
	if n, err := strconv.Atoi(d[i:]); err == nil && n < len(gs) {
		return &gs[n]
	}
	return &gs[0]
}

// gpuModes place every repeating layer on the device; hostModes none.
var (
	gpuModes  = map[string]bool{"vram-full": true, "vram-ple-host": true, "vram-ple-disk": true}
	hostModes = map[string]bool{"ram-full": true, "ram-ple-disk": true, "ram-hybrid": true, "mmap": true, "stream": true}
)

// Ceiling is the highest plausible tg without MTP for a run, in tok/s, or
// 0 when the model, the placement or the hardware is unknown.
func Ceiling(r *benchfmt.Result, run benchfmt.Run, cpu string, slack float64) float64 {
	cost, ok := Models[r.Model.ID]
	if !ok || cost.Layers <= 0 {
		return 0
	}
	mode := run.Plan.Mode
	onGPU := normalize.Backend(run.Plan.Device) != "CPU"
	total := float64(cost.LayerBytes + cost.OutputBytes)
	var seconds float64
	host := HostBandwidth(r.Machine, cpu) * 1e9
	switch {
	case hostModes[mode] || (!onGPU && mode != ""):
		if host == 0 {
			return 0
		}
		seconds = total / host
	case gpuModes[mode] || (mode == "vram-partial" && (run.Plan.GPULayers < 0 || run.Plan.GPULayers >= cost.Layers)):
		dev := DeviceBandwidth(r.Machine, run.Plan.Device, cpu) * 1e9
		if dev == 0 {
			return 0
		}
		seconds = total / dev
	case mode == "vram-partial" || mode == "vram-split":
		dev := DeviceBandwidth(r.Machine, run.Plan.Device, cpu) * 1e9
		if dev == 0 || host == 0 {
			return 0
		}
		n := run.Plan.GPULayers
		if n < 0 {
			n = 0
		}
		per := float64(cost.LayerBytes) / float64(cost.Layers)
		seconds = float64(n)*per/dev + (float64(cost.Layers-n)*per+float64(cost.OutputBytes))/host
	default:
		return 0
	}
	if slack < 1 {
		slack = 1
	}
	return slack / seconds
}

// Plausibility judges a run's tg (both depths) and plain chat decode, all
// measured without MTP, against the ceiling.
func Plausibility(r *benchfmt.Result, run benchfmt.Run, cpu string, slack float64) (string, float64) {
	if run.Status != benchfmt.StatusOK {
		return store.PlausibleUnknown, 0
	}
	c := Ceiling(r, run, cpu, slack)
	if c == 0 {
		return store.PlausibleUnknown, 0
	}
	m := run.Metrics
	for _, v := range []float64{m.TGTokS.D0, m.TGTokS.DN, m.ChatTokS} {
		if v > c {
			return store.PlausibleNo, c
		}
	}
	return store.PlausibleOK, c
}
