// Package tune is the autotuner behind `nanollm optimize` (docs/OPTIMIZE.md):
// the user picks one of three modes, and the module looks the machine over,
// drops the options the mode rules out, measures the rest briefly and writes
// a profile the server runs by. It needs no administrator rights.
package tune

import (
	"fmt"
	"strings"
)

// Mode is what the user asked of the machine.
type Mode string

const (
	// Silent loads the hardware least: CPU only, two threads, low priority.
	Silent Mode = "silent"
	// Optimal is the fastest answer within 4 GiB of memory and half the cores.
	Optimal Mode = "optimal"
	// Powerful is the fastest answer with everything the machine has free.
	Powerful Mode = "powerful"
)

// Modes lists the three, quietest first.
var Modes = []Mode{Silent, Optimal, Powerful}

// ParseMode reads a mode name, ignoring case and surrounding space.
func ParseMode(s string) (Mode, error) {
	m := Mode(strings.ToLower(strings.TrimSpace(s)))
	if !m.Valid() {
		return "", fmt.Errorf("unknown mode %q: use silent, optimal or powerful", s)
	}
	return m, nil
}

// Valid says whether m is one of the three modes.
func (m Mode) Valid() bool {
	for _, k := range Modes {
		if m == k {
			return true
		}
	}
	return false
}

// Summary is the one-line meaning of a mode, for menus.
func (m Mode) Summary() string {
	switch m {
	case Silent:
		return "quiet: CPU only, 2 threads, low priority; slowest answers"
	case Optimal:
		return "balanced: up to 4 GiB of memory and half of the cores; the fastest answer within that"
	case Powerful:
		return "fastest: all the free memory and all the cores"
	}
	return ""
}

// Fixed numbers of the modes (docs/OPTIMIZE.md, §2).
const (
	MiB = int64(1) << 20
	GiB = int64(1) << 30

	// OptimalMemLimit is the plan's limit of private memory plus video memory
	// in the optimal mode, and MemTolerance how far a measurement may go past
	// it.
	OptimalMemLimit = 4 * GiB
	MemTolerance    = 200 * MiB
	// PowerfulMinFree is the free memory below which the powerful mode falls
	// back to the optimal mode's memory limit.
	PowerfulMinFree = 6 * GiB
	// MinTokS is the decode rate below which a setup is not worth running.
	MinTokS = 1.0
	// MinNCtx is the smallest context per slot the modes accept: the reference
	// request (a 4096-token prompt, 300 tokens of answer) plus the chat markup.
	MinNCtx = 4608
	// SilentThreads is the thread count of the silent mode.
	SilentThreads = 2
)

// Goal is what a mode optimizes among the setups that meet its limits.
type Goal string

const (
	// GoalMinMemory picks the setup that holds the least memory and still
	// answers at MinTokS; the silent mode.
	GoalMinMemory Goal = "min_memory"
	// GoalMinTime picks the setup with the shortest typical answer.
	GoalMinTime Goal = "min_time"
)

// Cores is the CPU as the modes count it.
type Cores struct {
	// Physical counts every physical core, efficiency cores included.
	Physical int
	Logical  int
}

// Constraints are the limits and the goal of a mode, resolved for this
// machine. The server applies them without any measurement; a profile only
// picks the best setup inside them.
type Constraints struct {
	Mode Mode `json:"mode"`
	// CPUOnly rules out every GPU.
	CPUOnly bool `json:"cpu_only,omitempty"`
	// MemCap is the limit of private plus video memory the plan may take, and
	// MemTolerance how far the measured memory may exceed it; zero is no limit.
	MemCap       int64 `json:"mem_cap,omitempty"`
	MemTolerance int64 `json:"mem_tolerance,omitempty"`
	// Threads and ThreadsBatch are the CPU teams for decoding and for prompt
	// processing; zero leaves the usual defaults (every physical core, every
	// logical CPU of a hybrid machine for prompts).
	Threads      int `json:"threads,omitempty"`
	ThreadsBatch int `json:"threads_batch,omitempty"`
	// LowPriority puts the process below normal priority, and on Windows in
	// the efficiency (EcoQoS) class.
	LowPriority bool    `json:"low_priority,omitempty"`
	MinNCtx     int     `json:"min_n_ctx"`
	MinTokS     float64 `json:"min_tok_s"`
	Goal        Goal    `json:"goal"`
}

// ConstraintsFor resolves a mode for a machine with the given cores.
// freeBudget is the largest memory budget any device can offer right now
// (what plan.DerivedPreset would grant); the powerful mode with less than
// PowerfulMinFree falls back to the optimal mode's memory and says so in the
// second result.
func ConstraintsFor(m Mode, c Cores, freeBudget int64) (Constraints, *Problem) {
	k := Constraints{Mode: m, MinNCtx: MinNCtx, MinTokS: MinTokS, Goal: GoalMinTime}
	phys := max(c.Physical, 1)
	switch m {
	case Silent:
		k.CPUOnly, k.Goal, k.LowPriority = true, GoalMinMemory, true
		k.Threads = min(SilentThreads, phys)
		k.ThreadsBatch = k.Threads
	case Optimal:
		k.MemCap, k.MemTolerance = OptimalMemLimit, MemTolerance
		k.Threads = max(phys/2, 1)
		k.ThreadsBatch = k.Threads
	case Powerful:
		if freeBudget > 0 && freeBudget < PowerfulMinFree {
			k.MemCap, k.MemTolerance = OptimalMemLimit, MemTolerance
			p := degraded(freeBudget)
			return k, &p
		}
	}
	return k, nil
}
