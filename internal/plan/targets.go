package plan

// Target is the context a preset aims for once weights are placed (SPEC §15,
// "Ctx target"). The planner grows toward it from the floor and settles for
// less if the budget runs out; it never goes past it, because a preset is
// benchmarked as a whole and a bigger context than the row says is a
// different benchmark.
type Target struct {
	NCtx  int `json:"n_ctx"` // per slot
	Slots int `json:"slots"`
}

// Targets is the provisional preset map of SPEC §15 for Gemma 4 E4B. Where
// the spec gives a range ("4k–8k") the upper end is the target, the floor
// being the lower; where it gives no slot count, one slot.
var Targets = map[string]Target{
	"vram-8g":        {32768, 4},
	"vram-6g":        {16384, 2},
	"vram-4g":        {8192, 1},
	"vram-3g":        {8192, 1},
	"vram-2g":        {4096, 1},
	"vram-6g+ram-1g": {16384, 1},
	"vram-4g+ram-1g": {8192, 1},
	"vram-3g+ram-1g": {8192, 1},
	"vram-2g+ram-2g": {4096, 1},
	"ram-8g":         {32768, 1},
	"ram-6g":         {16384, 1},
	"ram-4g":         {8192, 1},
	"ram-3g":         {4096, 1},
	"ram-2g":         {4096, 1},
	"ram-1.5g":       {4096, 1},
	"ram-1g":         {4096, 1},
	"uma-8g":         {32768, 1},
	"uma-6g":         {16384, 1},
	"uma-4g":         {8192, 1},
}

// TargetFor returns the preset's target, or the floor for a budget that is
// not one of the defined presets.
func TargetFor(preset string) Target {
	if t, ok := Targets[preset]; ok {
		return t
	}
	return Target{FloorNCtx, FloorSlots}
}

// ProvenModes says which modes the planner may pick: a mode whose mechanism
// has not been shown to work on this tag is skipped with its reason rather
// than planned and found out at load time. A mode absent from the table is
// not proven.
type ProvenModes map[Mode]bool

// DefaultProven is the state of stage 1 (docs/STAGE1.md 1.4). M2 is
// allowed: it is what upstream does by default — per_layer_token_embd is an
// input-layer tensor and stays in host memory — and 1.5 confirms the
// placement. M7 and M8 rest on range-lock (1.6), proven by residency on
// Linux and by a working-set test on Windows; macOS only measures it, so
// ProvenFor turns them off there.
var DefaultProven = ProvenModes{
	VRAMFull:    true,
	VRAMPLEHost: true,
	VRAMPLEDisk: true,
	VRAMPartial: true,
	RAMFull:     true,
	RAMPLEDisk:  true,
	RAMHybrid:   true,
	MMap:        true,
}

// ProvenFor is DefaultProven on one OS.
func ProvenFor(goos string) ProvenModes {
	out := ProvenModes{}
	for m, ok := range DefaultProven {
		out[m] = ok
	}
	if goos == "darwin" {
		out[RAMPLEDisk], out[RAMHybrid] = false, false
	}
	return out
}

var unprovenWhy = map[Mode]string{
	RAMPLEDisk: "needs range-lock (docs/STAGE1.md 1.6), proven on Linux and Windows only",
	RAMHybrid:  "needs range-lock (docs/STAGE1.md 1.6), proven on Linux and Windows only",
}

func (p ProvenModes) why(m Mode) string {
	if p[m] {
		return ""
	}
	if w, ok := unprovenWhy[m]; ok {
		return w
	}
	return "not proven on this llama.cpp tag"
}
