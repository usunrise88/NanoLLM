//go:build linux

package procmem

import (
	"os"
	"strconv"
	"strings"
)

func status() map[string]int64 {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return nil
	}
	out := map[string]int64{}
	for _, line := range strings.Split(string(b), "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		if kb, err := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(v), " kB"), 10, 64); err == nil {
			out[k] = kb << 10
		}
	}
	return out
}

// Private is the anonymous and shared-anonymous resident memory
// (RssAnon + RssShmem): the part of the resident set that is not a file the
// OS could re-read.
func Private() int64 {
	s := status()
	return s["RssAnon"] + s["RssShmem"]
}

// Commit is the virtual size, the nearest thing to a commit charge.
func Commit() int64 { return status()["VmSize"] }

func PeakCommit() int64 { return status()["VmPeak"] }

// Resident is the peak resident set (VmHWM), file pages included.
func Resident() int64 { return status()["VmHWM"] }
