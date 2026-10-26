// Package procmem measures what a process holds, the way the autotuner counts
// it (docs/OPTIMIZE.md, §2.1): private bytes (memory the OS cannot take back
// without a swap file) plus the video memory the process uses. The pages of
// the mapped model file are left out: the OS gives them to other programs
// whenever they ask.
package procmem

import (
	"errors"
	"sync"
	"time"
)

// ErrUnavailable means the platform has no way to read the number, or this
// process may not read it.
var ErrUnavailable = errors.New("procmem: not available")

// Sources of a VRAM reading.
const (
	SourcePDH       = "pdh"
	SourceNvidiaSMI = "nvidia-smi"
)

// VRAM is the GPU memory one process holds: Dedicated in the card's own memory
// and Shared in the system memory the driver lends to the GPU (on a discrete
// card a nonzero Shared beyond the host memory the plan pinned means the driver
// spilled the model out of VRAM).
type VRAM struct {
	Dedicated int64  `json:"dedicated"`
	Shared    int64  `json:"shared"`
	Source    string `json:"source"`
	// Adapters is the same split by graphics adapter where the source has one
	// (PDH: a machine with an iGPU and a discrete card has two), so a reading
	// can be attributed to the device the model ran on.
	Adapters []Adapter `json:"adapters,omitempty"`
}

// Adapter is one graphics adapter's share of a process's GPU memory; Name is
// the adapter instance as the source names it (PDH: luid_0x…_0x…_phys_0).
type Adapter struct {
	Name      string `json:"name"`
	Dedicated int64  `json:"dedicated"`
	Shared    int64  `json:"shared"`
}

// Meter follows the peak of the calling process's private memory by sampling
// it: no OS keeps a peak of the private working set.
type Meter struct {
	stop chan struct{}
	done chan struct{}
	mu   sync.Mutex
	peak int64
}

// SampleInterval is how often a Meter reads the process.
const SampleInterval = 100 * time.Millisecond

// Start begins following the calling process.
func Start() *Meter {
	m := &Meter{stop: make(chan struct{}), done: make(chan struct{}), peak: Private()}
	go m.sample()
	return m
}

func (m *Meter) sample() {
	defer close(m.done)
	t := time.NewTicker(SampleInterval)
	defer t.Stop()
	for {
		select {
		case <-m.stop:
			return
		case <-t.C:
			m.note()
		}
	}
}

func (m *Meter) note() {
	if cur := Private(); cur > 0 {
		m.mu.Lock()
		m.peak = max(m.peak, cur)
		m.mu.Unlock()
	}
}

// Peak is the highest private memory seen so far, the present reading included.
func (m *Meter) Peak() int64 {
	m.note()
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.peak
}

// Stop ends sampling and returns the peak.
func (m *Meter) Stop() int64 {
	select {
	case <-m.stop:
	default:
		close(m.stop)
	}
	<-m.done
	return m.Peak()
}
