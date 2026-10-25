// Package posture sets how politely the process behaves toward the machine:
// the quiet mode of the autotuner (docs/OPTIMIZE.md, §2) runs below normal
// priority and, on Windows, in the efficiency class (EcoQoS), where the
// scheduler keeps the process on the efficiency cores at low clocks. Lowering
// priority needs no privilege.
package posture

import "errors"

// ErrRestart means the change cannot be made on a running process: Linux does
// not let a process raise its own priority again without CAP_SYS_NICE, so a
// server that was quiet stays quiet until it is restarted.
var ErrRestart = errors.New("posture: the priority cannot be raised again without a restart")

// Nice is the priority of the quiet mode on the systems that have niceness.
const Nice = 10

// Level is how far the process steps back from the machine.
type Level int

const (
	// Normal is the priority every process starts with.
	Normal Level = iota
	// Below is below normal priority alone: a process an application runs in
	// the background (nanollm managed) in a mode that is not the quiet one.
	Below
	// Quiet is the quiet mode: below normal priority and, on Windows, the
	// efficiency class as well. Elsewhere it is the same as Below.
	Quiet
)

// Apply is ApplyLevel(Quiet) when low is set and ApplyLevel(Normal) when it is
// not.
func Apply(low bool) error {
	if low {
		return ApplyLevel(Quiet)
	}
	return ApplyLevel(Normal)
}

// State is the process's posture as the OS reports it.
type State struct {
	// Low is set when the priority is below normal.
	Low bool `json:"low"`
	// Eco is set when the process is in the efficiency class (Windows).
	Eco bool `json:"eco,omitempty"`
}
