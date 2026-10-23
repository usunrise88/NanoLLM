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

// State is the process's posture as the OS reports it.
type State struct {
	// Low is set when the priority is below normal.
	Low bool `json:"low"`
	// Eco is set when the process is in the efficiency class (Windows).
	Eco bool `json:"eco,omitempty"`
}
