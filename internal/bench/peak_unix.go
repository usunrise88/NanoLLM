//go:build !windows

package bench

import (
	"os"
	"runtime"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

// peakMemory is the process's peak resident set and its locked memory.
func peakMemory() (rss, locked int64) {
	if b, err := os.ReadFile("/proc/self/status"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			k, v, ok := strings.Cut(line, ":")
			if !ok {
				continue
			}
			kb, _ := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(v), " kB"), 10, 64)
			switch k {
			case "VmHWM":
				rss = kb << 10
			case "VmLck":
				locked = kb << 10
			}
		}
		return rss, locked
	}
	var ru unix.Rusage
	if unix.Getrusage(unix.RUSAGE_SELF, &ru) == nil {
		rss = ru.Maxrss
		if runtime.GOOS != "darwin" { // kilobytes everywhere but macOS
			rss <<= 10
		}
	}
	return rss, 0
}
