// Package plan holds the vocabulary of memory placement: the memory modes a
// model can be loaded in and the fixed-budget presets that bound them (SPEC §4,
// §15).
//
// The planner itself — measured tensor inventory in, mode, layer split and
// context out — lands with v0.1. What is here is the part everything else
// already has to agree on: the configuration validates against these names,
// `nanollm plan` and `bench` print them, and the benchmark matrix is keyed by
// them. Defining them once keeps a typo in a preset from becoming a separate
// row in a report.
package plan

import (
	"fmt"
	"strings"
)

// Mode is one placement strategy, named as the configuration spells it.
type Mode string

// ModeAuto lets the planner pick the first mode that fits the preset.
const ModeAuto Mode = "auto"

const (
	VRAMFull    Mode = "vram-full"     // M1: every tensor group in VRAM
	VRAMPLEHost Mode = "vram-ple-host" // M2: PLE tables in locked host RAM
	VRAMPLEDisk Mode = "vram-ple-disk" // M3: PLE tables file-backed
	VRAMPartial Mode = "vram-partial"  // M4: first N layers in VRAM
	VRAMSplit   Mode = "vram-split"    // M5: attention and FFN on different devices
	RAMFull     Mode = "ram-full"      // M6: everything locked in RAM
	RAMPLEDisk  Mode = "ram-ple-disk"  // M7: core locked, PLE file-backed
	RAMHybrid   Mode = "ram-hybrid"    // M8: hot groups locked, the rest file-backed
	MMap        Mode = "mmap"          // M9: the OS decides
	Stream      Mode = "stream"        // M10: layer streaming, batch only
)

// Stage says when a mode becomes available. Research modes are named so the
// configuration can refer to them in a benchmark run, but the planner never
// picks one on its own.
type Stage string

const (
	StageV01      Stage = "v0.1"
	StageV10      Stage = "v1.0"
	StageResearch Stage = "research"
)

// ModeInfo is one row of the mode table in SPEC §4.
type ModeInfo struct {
	// ID is the short name the spec and the benchmark reports use (M1…M10).
	ID    string
	Mode  Mode
	Stage Stage
}

// Modes lists every mode in spec order.
var Modes = []ModeInfo{
	{"M1", VRAMFull, StageV01},
	{"M2", VRAMPLEHost, StageV01},
	{"M3", VRAMPLEDisk, StageV10},
	{"M4", VRAMPartial, StageV01},
	{"M5", VRAMSplit, StageResearch},
	{"M6", RAMFull, StageV01},
	{"M7", RAMPLEDisk, StageV10},
	{"M8", RAMHybrid, StageV10},
	{"M9", MMap, StageV01},
	{"M10", Stream, StageResearch},
}

// PlannerOrder is the order in which the planner tries modes against a preset:
// the first that fits wins, and M4 is tried with the largest layer count first.
// Research modes are absent on purpose.
var PlannerOrder = []Mode{
	VRAMFull, VRAMPLEHost, VRAMPLEDisk, VRAMPartial,
	RAMFull, RAMPLEDisk, RAMHybrid, MMap,
}

// ParseMode accepts a mode by name ("vram-full"), by its short ID ("M1", any
// case) or "auto". The short form is what people copy out of a benchmark
// report, so refusing it would only produce a support question.
func ParseMode(s string) (Mode, error) {
	s = strings.TrimSpace(s)
	if s == "" || s == string(ModeAuto) {
		return ModeAuto, nil
	}
	for _, m := range Modes {
		if s == string(m.Mode) || strings.EqualFold(s, m.ID) {
			return m.Mode, nil
		}
	}
	return "", fmt.Errorf("unknown memory mode %q; use auto, M1…M10 or one of %s", s, modeNames())
}

// Info returns the table row of m. ModeAuto has none.
func Info(m Mode) (ModeInfo, bool) {
	for _, info := range Modes {
		if info.Mode == m {
			return info, true
		}
	}
	return ModeInfo{}, false
}

func modeNames() string {
	names := make([]string, len(Modes))
	for i, m := range Modes {
		names[i] = string(m.Mode)
	}
	return strings.Join(names, ", ")
}
