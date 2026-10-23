//go:build !linux && !windows

package memlock

import "errors"

// StreamOptions is how a Streamer feeds the layers.
type StreamOptions struct {
	Ahead  int
	Keep   int
	Evict  bool
	Method string
	Depth  int
}

// The methods a Streamer reads ahead by.
const (
	MethodAdvise = "advise"
	MethodPread  = "pread"
)

// Streamer is Linux-only for now: the other systems have no madvise with
// MADV_PAGEOUT. A platform port (PrefetchVirtualMemory, F_RDADVISE) comes
// with its measurements.
type Streamer struct{}

type StreamStats struct {
	Calls, Dropped, Errors int64
	WillNeed, PageOut      int64
	FirstError             error
}

func NewStreamer(string, map[int][]Range, StreamOptions) (*Streamer, error) {
	return nil, errors.New("memlock: layer streaming is not supported on this platform")
}

func (s *Streamer) Layers() int        { return 0 }
func (s *Streamer) Warm()              {}
func (s *Streamer) Layer(int)          {}
func (s *Streamer) Stats() StreamStats { return StreamStats{} }
func (s *Streamer) Close()             {}
