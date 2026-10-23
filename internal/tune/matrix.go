package tune

import (
	_ "embed"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// The matrix of known optimizations and how they did on which hardware
// (docs/OPTIMIZE.md, §3.8 and appendix A). An optimization that was never
// measured on this exact machine is not switched off for that: the matrix
// finds the nearest class that was measured, by the attributes that matter
// to that lever.

//go:embed matrix.yaml
var matrixYAML []byte

// Value sources, strongest first.
const (
	SourceMeasured  = "measured"  // measured on this machine
	SourceClass     = "class"     // a cell with the same significant attributes
	SourceInherited = "inherited" // the nearest cell
	SourceMechanism = "mechanism" // a short probe of the mechanism
	SourcePosture   = "posture"   // the rule for the type of lever
	SourceRule      = "rule"      // the static rule of the planner or the mode
	SourceConfig    = "config"    // set in the configuration
	SourceProfile   = "profile"   // chosen by an earlier optimize run
	SourceMode      = "mode"      // the limits of the mode
)

// Attributes of a hardware class. All values are strings so that a class is a
// plain map: "86" for a compute capability, "true" for hybrid.
const (
	AttrCPUArch    = "cpu_arch"    // x86_64, arm64
	AttrCPUVariant = "cpu_variant" // the ggml CPU backend variant: haswell, alderlake, zen4, …
	AttrHybrid     = "hybrid"      // true: performance and efficiency cores
	AttrNUMA       = "numa_nodes"  // 1, 2, 4, …
	AttrGPUBackend = "gpu_backend" // cuda, vulkan, metal
	AttrGPUVendor  = "gpu_vendor"  // nvidia, intel, amd, apple
	AttrGPUMemory  = "gpu_memory"  // own, shared
	AttrGPUCC      = "gpu_cc"      // NVIDIA compute capability: 86
	AttrOS         = "os"          // windows, linux, macos
	AttrDisk       = "disk"        // hdd, ssd, nvme
)

// Class is a set of hardware attributes. A cell's class lists only the
// attributes its result depends on.
type Class map[string]string

// Verdicts of a cell.
type Verdict string

const (
	Win     Verdict = "win"
	Loss    Verdict = "loss"
	Neutral Verdict = "neutral"
	Depends Verdict = "depends"
)

// Effect is the size of a result.
type Effect struct {
	Metric string  `yaml:"metric,omitempty" json:"metric,omitempty"`
	Ratio  float64 `yaml:"ratio,omitempty" json:"ratio,omitempty"`
	Note   string  `yaml:"note,omitempty" json:"note,omitempty"`
}

// Cell is one measured fact: this lever, in this variant, on this class of
// hardware, did this.
type Cell struct {
	Lever      string         `yaml:"lever" json:"lever"`
	Variant    map[string]any `yaml:"variant,omitempty" json:"variant,omitempty"`
	Class      Class          `yaml:"class" json:"class"`
	Verdict    Verdict        `yaml:"verdict" json:"verdict"`
	Effect     *Effect        `yaml:"effect,omitempty" json:"effect,omitempty"`
	Conditions map[string]any `yaml:"conditions,omitempty" json:"conditions,omitempty"`
	Machines   []string       `yaml:"machines,omitempty" json:"machines,omitempty"`
	Source     string         `yaml:"source" json:"source"`
	Why        string         `yaml:"why,omitempty" json:"why,omitempty"`
}

// Baseline is how fast a class of hardware runs the reference model, for
// estimating a candidate's answer time before measuring it.
type Baseline struct {
	// Device is cpu, cuda, vulkan or metal.
	Device string `yaml:"device" json:"device"`
	Class  Class  `yaml:"class" json:"class"`
	// Threads is the CPU team the numbers were taken with; 0 for a GPU.
	Threads int `yaml:"threads,omitempty" json:"threads,omitempty"`
	// PP512 is prefill tokens per second at the start of a prompt and TG0 decode
	// tokens per second at an empty context. DepthPP is the average prefill
	// rate of a 4096-token prompt divided by PP512, and DepthTG the decode rate
	// after 4096 tokens divided by TG0: the factors that carry a probe at 512
	// tokens over to the reference request. MTPGain is the decode speedup of
	// the best drafting setting on chat; 1 when there is none.
	PP512   float64 `yaml:"pp512" json:"pp512"`
	TG0     float64 `yaml:"tg0" json:"tg0"`
	DepthPP float64 `yaml:"depth_pp" json:"depth_pp"`
	DepthTG float64 `yaml:"depth_tg" json:"depth_tg"`
	MTPGain float64 `yaml:"mtp_gain,omitempty" json:"mtp_gain,omitempty"`
	// PPThreadExp and TGThreadExp are how prefill and decode speed follow the
	// thread count of a CPU: rate ~ threads^exp. Decoding waits for memory: 0.57
	// on the EPYC, whose cores each reach little of the bandwidth, 0.1 on the
	// i7-12700H, where two threads already reach it (11.3 tok/s at 2 threads,
	// 11.1 at 6). Zero is DefaultPPThreadExp and DefaultTGThreadExp.
	PPThreadExp float64 `yaml:"pp_thread_exp,omitempty" json:"pp_thread_exp,omitempty"`
	TGThreadExp float64 `yaml:"tg_thread_exp,omitempty" json:"tg_thread_exp,omitempty"`
	// Estimated says DepthPP and DepthTG are not measured on this class.
	Estimated bool   `yaml:"estimated,omitempty" json:"estimated,omitempty"`
	Source    string `yaml:"source" json:"source"`
}

// PP4096 and TG4096 are the rates at the depth of the reference request.
func (b Baseline) PP4096() float64 { return b.PP512 * b.DepthPP }
func (b Baseline) TG4096() float64 { return b.TG0 * b.DepthTG }

// Matrix is the table of cells and baselines.
type Matrix struct {
	Cells     []Cell     `yaml:"cells"`
	Baselines []Baseline `yaml:"baselines"`
}

// LoadMatrix is the matrix built into the binary.
func LoadMatrix() (*Matrix, error) {
	var m Matrix
	if err := yaml.Unmarshal(matrixYAML, &m); err != nil {
		return nil, fmt.Errorf("tune: matrix.yaml: %w", err)
	}
	return &m, m.validate()
}

// Default is the built-in matrix; a broken one is a programming error found
// by the tests.
var Default = func() *Matrix {
	m, err := LoadMatrix()
	if err != nil {
		panic(err)
	}
	return m
}()

func (m *Matrix) validate() error {
	for i, c := range m.Cells {
		switch c.Verdict {
		case Win, Loss, Neutral, Depends:
		default:
			return fmt.Errorf("tune: matrix cell %d (%s): unknown verdict %q", i, c.Lever, c.Verdict)
		}
		if _, ok := Significant[c.Lever]; !ok {
			return fmt.Errorf("tune: matrix cell %d: unknown lever %q", i, c.Lever)
		}
		if c.Source == "" {
			return fmt.Errorf("tune: matrix cell %d (%s) has no source", i, c.Lever)
		}
	}
	for i, b := range m.Baselines {
		if b.Source == "" || b.PP512 <= 0 || b.TG0 <= 0 || b.DepthPP <= 0 || b.DepthTG <= 0 {
			return fmt.Errorf("tune: matrix baseline %d (%s) is incomplete", i, b.Device)
		}
	}
	return nil
}

// Significant are the attributes each lever depends on (docs/OPTIMIZE.md,
// §3.8). A lever absent from the table is not known to the matrix.
var Significant = map[string][]string{
	"repack":          {AttrCPUVariant},
	"mtp_cpu":         {AttrCPUVariant},
	"mtp_gpu":         {AttrGPUBackend, AttrGPUVendor, AttrGPUMemory},
	"ngram_cpu":       {AttrCPUVariant},
	"ngram_gpu":       {AttrGPUBackend},
	"threads_decode":  {AttrHybrid},
	"threads_prefill": {AttrHybrid},
	"numa":            {AttrNUMA},
	"device_sampling": {AttrGPUBackend, AttrGPUMemory},
	"fa_small":        {AttrGPUBackend, AttrGPUMemory},
	"fa_off_cpu":      {AttrCPUVariant},
	"fa_off_gpu":      {AttrGPUBackend, AttrGPUMemory},
	"cuda_graph":      {AttrGPUBackend},
	"cuda_norm":       {AttrGPUBackend},
	"mmvq":            {AttrGPUCC},
	"fa_vec":          {AttrGPUCC},
	"fuse_batch":      {AttrGPUCC},
	"vk_matvec":       {AttrGPUVendor, AttrGPUMemory, AttrOS},
	"ubatch":          {AttrGPUBackend, AttrGPUMemory},
	"cuda13":          {AttrGPUCC},
	"kv_q8":           {},
	"thp":             {AttrCPUVariant},
	"stream":          {AttrDisk},
	"stream_ahead":    {AttrDisk},
	"stream_evict":    {AttrDisk},
	"madv_random":     {AttrDisk},
}

// MaxInheritDistance is the distance beyond which a cell is too far to lend
// its verdict to another class.
const MaxInheritDistance = 60

// Match is the cell a lookup found, where its value comes from and how far the
// cell's class is from the machine's.
type Match struct {
	Cell     Cell   `json:"cell"`
	Source   string `json:"source"` // SourceClass or SourceInherited
	Distance int    `json:"distance"`
}

// Lookup finds the cell of a lever nearest to a machine's class, by the
// lever's significant attributes only. Of several at the same distance the
// one measured on more machines wins. ok is false when the lever has no cell
// within MaxInheritDistance.
func (m *Matrix) Lookup(lever string, c Class) (Match, bool) {
	all := m.LookupAll(lever, c)
	if len(all) == 0 {
		return Match{}, false
	}
	return all[0], true
}

// LookupAll returns the nearest cell of each variant of a lever, nearest
// first. A lever whose cells have no variants yields at most one entry.
func (m *Matrix) LookupAll(lever string, c Class) []Match {
	best := map[string]Match{}
	for _, cell := range m.Cells {
		if cell.Lever != lever {
			continue
		}
		d := distance(lever, c, cell.Class)
		if d > MaxInheritDistance {
			continue
		}
		key := variantKey(cell.Variant)
		cur, ok := best[key]
		if ok && (cur.Distance < d || cur.Distance == d && len(cur.Cell.Machines) >= len(cell.Machines)) {
			continue
		}
		src := SourceInherited
		if d == 0 {
			src = SourceClass
		}
		best[key] = Match{Cell: cell, Source: src, Distance: d}
	}
	out := make([]Match, 0, len(best))
	for _, v := range best {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Distance != out[j].Distance {
			return out[i].Distance < out[j].Distance
		}
		return variantKey(out[i].Cell.Variant) < variantKey(out[j].Cell.Variant)
	})
	return out
}

func variantKey(v map[string]any) string {
	keys := make([]string, 0, len(v))
	for k := range v {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&b, "%s=%v;", k, v[k])
	}
	return b.String()
}

// distance is how far a cell's class is from the machine's, over the lever's
// significant attributes that the cell names. The attributes carry different
// weights: the same backend and kind of memory matter more than the vendor.
func distance(lever string, machine, cell Class) int {
	d := 0
	for _, attr := range Significant[lever] {
		want, named := cell[attr]
		if !named {
			continue // the cell covers every value of this attribute
		}
		have, known := machine[attr]
		if !known {
			d += 5
			continue
		}
		d += attrDistance(attr, have, want)
	}
	return d
}

func attrDistance(attr, have, want string) int {
	if have == want {
		return 0
	}
	switch attr {
	case AttrCPUVariant:
		return isaDistance(have, want)
	case AttrGPUCC:
		a, _ := strconv.Atoi(have)
		b, _ := strconv.Atoi(want)
		d := a - b
		if d < 0 {
			d = -d
		}
		if a/10 != b/10 {
			d += 20
		}
		return d
	case AttrGPUBackend:
		return 100 // a kernel tuned for one backend says nothing about another
	case AttrGPUMemory:
		return 30
	case AttrGPUVendor:
		return 10
	case AttrOS:
		return 5
	case AttrHybrid:
		return 20
	case AttrCPUArch:
		return 100
	case AttrNUMA:
		a, _ := strconv.Atoi(have)
		b, _ := strconv.Atoi(want)
		if (a > 1) != (b > 1) {
			return 50
		}
		d := a - b
		if d < 0 {
			d = -d
		}
		return d
	case AttrDisk:
		return 10 * abs(diskRank(have)-diskRank(want))
	}
	return 40
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

func diskRank(t string) int {
	switch t {
	case "hdd":
		return 0
	case "ssd":
		return 1
	case "nvme":
		return 2
	}
	return 1
}

// variantISA lists the vector extensions each ggml CPU variant needs, so that
// a processor that has the superset of a measured variant's extensions is near
// it: an AVX-VNNI part is closer to the AVX2 class than to "x86".
var variantISA = map[string][]string{
	"x64":             {},
	"sse42":           {"sse42"},
	"sandybridge":     {"sse42", "avx"},
	"ivybridge":       {"sse42", "avx", "f16c"},
	"piledriver":      {"sse42", "avx", "f16c", "fma"},
	"haswell":         {"sse42", "avx", "f16c", "fma", "avx2"},
	"alderlake":       {"sse42", "avx", "f16c", "fma", "avx2", "avx_vnni"},
	"skylakex":        {"sse42", "avx", "f16c", "fma", "avx2", "avx512"},
	"cannonlake":      {"sse42", "avx", "f16c", "fma", "avx2", "avx512"},
	"cascadelake":     {"sse42", "avx", "f16c", "fma", "avx2", "avx512", "avx512_vnni"},
	"icelake":         {"sse42", "avx", "f16c", "fma", "avx2", "avx512", "avx512_vnni"},
	"cooperlake":      {"sse42", "avx", "f16c", "fma", "avx2", "avx512", "avx512_vnni", "avx512_bf16"},
	"zen4":            {"sse42", "avx", "f16c", "fma", "avx2", "avx512", "avx512_vnni", "avx512_bf16"},
	"sapphirerapids":  {"sse42", "avx", "f16c", "fma", "avx2", "avx512", "avx512_vnni", "avx512_bf16", "amx"},
	"armv8":           {"neon"},
	"armv8.2-dotprod": {"neon", "dotprod"},
	"armv8.6-i8mm":    {"neon", "dotprod", "i8mm"},
}

// isaDistance is how far a measured CPU variant is from the machine's: a
// variant whose extensions the machine has all of is near (one point for each
// extension the machine has beyond it); one that needs extensions the machine
// lacks is far (four points for each, since its kernels cannot run).
func isaDistance(have, want string) int {
	h, okH := variantISA[have]
	w, okW := variantISA[want]
	if !okH || !okW {
		return 40
	}
	hs := map[string]bool{}
	for _, x := range h {
		hs[x] = true
	}
	ws := map[string]bool{}
	for _, x := range w {
		ws[x] = true
	}
	d := 0
	for x := range ws {
		if !hs[x] {
			d += 4
		}
	}
	for x := range hs {
		if !ws[x] {
			d++
		}
	}
	return d
}

// Without returns a matrix with the cells and baselines of the given classes
// taken out: a cell goes when every attribute it names has the same value in
// one of the classes. It is how the blind test hides what is known about the
// machine under test.
func (m *Matrix) Without(classes ...Class) *Matrix {
	covered := func(cell Class) bool {
		for _, c := range classes {
			all := len(cell) > 0
			for k, v := range cell {
				if c[k] != v {
					all = false
					break
				}
			}
			if all {
				return true
			}
		}
		return false
	}
	out := &Matrix{}
	for _, c := range m.Cells {
		if !covered(c.Class) {
			out.Cells = append(out.Cells, c)
		}
	}
	for _, b := range m.Baselines {
		if !covered(b.Class) {
			out.Baselines = append(out.Baselines, b)
		}
	}
	return out
}

// BaselineFor is the baseline of a device kind nearest to the machine's class,
// with the distance as for cells; the CPU baseline nearest in kernels, or a GPU
// baseline with the same backend. ok is false when there is none.
func (m *Matrix) BaselineFor(device string, c Class) (Baseline, int, bool) {
	var best Baseline
	bestD, found := 1<<30, false
	for _, b := range m.Baselines {
		if b.Device != device {
			continue
		}
		d := 0
		for attr, want := range b.Class {
			have, known := c[attr]
			if !known {
				d += 5
				continue
			}
			if attr == AttrGPUMemory && have != want {
				// A card with memory of its own and an iGPU on shared memory differ
				// in speed by an order of magnitude: a lever may pass from one to the
				// other, a speed may not.
				d += MaxInheritDistance + 1
			}
			d += attrDistance(attr, have, want)
		}
		if d < bestD || d == bestD && b.Threads > best.Threads {
			best, bestD, found = b, d, true
		}
	}
	if !found || bestD > MaxInheritDistance {
		return Baseline{}, 0, false
	}
	return best, bestD, true
}
