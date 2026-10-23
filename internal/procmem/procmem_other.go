//go:build !windows && !linux

package procmem

import (
	"runtime"

	"golang.org/x/sys/unix"
)

// Private is the resident set where the OS offers nothing narrower.
func Private() int64 { return Resident() }

func Commit() int64 { return Resident() }

func PeakCommit() int64 { return Resident() }

// Resident is the peak resident set from getrusage.
func Resident() int64 {
	var ru unix.Rusage
	if unix.Getrusage(unix.RUSAGE_SELF, &ru) != nil {
		return 0
	}
	if runtime.GOOS == "darwin" { // bytes there, kilobytes elsewhere
		return ru.Maxrss
	}
	return ru.Maxrss << 10
}
