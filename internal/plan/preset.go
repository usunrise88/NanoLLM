package plan

import (
	"fmt"
	"strconv"
	"strings"
)

// GiB is the unit presets are written in. "ram-4g" means 4 GiB: the budget is
// compared against byte counts the OS and the GPU driver report, which are
// binary.
const GiB = int64(1) << 30

// Budget is the memory a preset allows, in bytes. A zero field means the preset
// grants none of that kind, not that it is unlimited.
//
// Unified is set only for Apple Silicon presets, where Metal maps host memory
// and VRAM and RAM are one pool; it is never combined with the other two.
type Budget struct {
	VRAM    int64
	RAM     int64
	Unified int64
}

// Total is everything the preset allows across all memory kinds.
func (b Budget) Total() int64 { return b.VRAM + b.RAM + b.Unified }

// Preset is a hard memory budget with the name the configuration uses.
type Preset struct {
	Name   string
	Budget Budget
}

// Presets are the fixed budgets defined in SPEC §4, in table order. They are
// the unit of optimization and benchmarking, so the set is closed: a budget
// outside it has no measured plan behind it, and accepting one would make the
// benchmark matrix grow a row per typo.
var Presets = []string{
	"ram-1g", "ram-1.5g", "ram-2g", "ram-3g", "ram-4g", "ram-6g", "ram-8g",
	"vram-2g", "vram-3g", "vram-4g", "vram-6g", "vram-8g",
	"vram-2g+ram-2g", "vram-3g+ram-1g", "vram-4g+ram-1g", "vram-6g+ram-1g",
	"uma-4g", "uma-6g", "uma-8g",
}

// LookupPreset returns one of the defined presets.
func LookupPreset(name string) (Preset, error) {
	name = strings.TrimSpace(name)
	for _, p := range Presets {
		if p == name {
			return ParsePreset(name)
		}
	}
	return Preset{}, fmt.Errorf("unknown preset %q; defined presets are %s",
		name, strings.Join(Presets, ", "))
}

// ParsePreset reads the budget out of a preset name: one or more
// "<kind>-<size>g" terms joined by "+", where kind is vram, ram or uma and size
// may be fractional ("ram-1.5g").
//
// It parses any well-formed name, defined or not, so tests and the benchmark
// harness can describe a budget before it is promoted to a preset. Use
// LookupPreset for anything a user typed.
func ParsePreset(name string) (Preset, error) {
	p := Preset{Name: name}
	if name == "" {
		return p, fmt.Errorf("empty preset name")
	}
	seen := map[string]bool{}
	for _, term := range strings.Split(name, "+") {
		kind, size, ok := strings.Cut(term, "-")
		if !ok || !strings.HasSuffix(size, "g") {
			return p, fmt.Errorf("preset %q: term %q is not <vram|ram|uma>-<size>g", name, term)
		}
		gb, err := strconv.ParseFloat(strings.TrimSuffix(size, "g"), 64)
		if err != nil || gb <= 0 {
			return p, fmt.Errorf("preset %q: %q is not a positive size in GiB", name, size)
		}
		if seen[kind] {
			return p, fmt.Errorf("preset %q names %s twice", name, kind)
		}
		seen[kind] = true

		bytes := int64(gb * float64(GiB))
		switch kind {
		case "vram":
			p.Budget.VRAM = bytes
		case "ram":
			p.Budget.RAM = bytes
		case "uma":
			p.Budget.Unified = bytes
		default:
			return p, fmt.Errorf("preset %q: unknown memory kind %q", name, kind)
		}
	}
	// On unified memory there is no separate VRAM to give or withhold, so a
	// combined budget would count the same bytes twice.
	if p.Budget.Unified > 0 && (p.Budget.VRAM > 0 || p.Budget.RAM > 0) {
		return p, fmt.Errorf("preset %q: uma cannot be combined with vram or ram", name)
	}
	return p, nil
}

// FloorNCtx and FloorSlots are the smallest context the planner will settle
// for: if weights leave less than this, the load is refused with the shortfall
// rather than started with a context nobody can use.
const (
	FloorNCtx  = 4096
	FloorSlots = 1
)
