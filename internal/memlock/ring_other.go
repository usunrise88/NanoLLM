//go:build !linux

package memlock

import (
	"errors"
	"time"
)

// RingTensor is a tensor of a managed layer: where its bytes lie in the file.
type RingTensor struct {
	Layer          int
	Offset, Length int64
}

// RingOptions is how a Ring runs.
type RingOptions struct {
	Ahead, Depth, Keep, Layers int
	Retarget                   func(i int, addr uintptr)
}

// RingStats are the ring's counters.
type RingStats struct {
	Reads, Bytes, Stalls, Skipped int64
	StallTime                     time.Duration
	Slots                         int
	Err                           error
}

// Ring is Linux-only for now: the direct reads of Windows (unbuffered
// overlapped I/O) and macOS (F_NOCACHE) come with their measurements.
type Ring struct{}

func NewRing(string, []RingTensor, RingOptions) (*Ring, error) {
	return nil, errors.New("memlock: the layer ring (direct reads) is not supported on this platform")
}

func (r *Ring) Warm()           {}
func (r *Ring) Layer(int)       {}
func (r *Ring) Layers() int     { return 0 }
func (r *Ring) Stats() RingStats { return RingStats{} }
func (r *Ring) Close()          {}
