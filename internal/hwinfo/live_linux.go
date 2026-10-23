//go:build linux

package hwinfo

import (
	"context"
	"os"
	"strconv"
	"strings"
	"time"
)

func memoryNow() (total, available int64) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	return parseMeminfo(string(b))
}

// parseMeminfo reads MemTotal and MemAvailable (kernel 3.14 and later; without
// it MemFree plus the caches stands in).
func parseMeminfo(s string) (total, available int64) {
	var free, cached, buffers int64
	haveAvail := false
	for _, line := range strings.Split(s, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		kb, err := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(v), " kB"), 10, 64)
		if err != nil {
			continue
		}
		switch k {
		case "MemTotal":
			total = kb << 10
		case "MemAvailable":
			available, haveAvail = kb<<10, true
		case "MemFree":
			free = kb << 10
		case "Cached":
			cached = kb << 10
		case "Buffers":
			buffers = kb << 10
		}
	}
	if !haveAvail {
		available = free + cached + buffers
	}
	return total, available
}

func cpuTimes() (idle, total uint64, ok bool) {
	b, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, 0, false
	}
	return parseProcStat(string(b))
}

// parseProcStat sums the first "cpu" line: user nice system idle iowait irq
// softirq steal. Idle counts iowait.
func parseProcStat(s string) (idle, total uint64, ok bool) {
	line, _, _ := strings.Cut(s, "\n")
	f := strings.Fields(line)
	if len(f) < 5 || f[0] != "cpu" {
		return 0, 0, false
	}
	for i, x := range f[1:] {
		if i >= 8 {
			break
		}
		v, err := strconv.ParseUint(x, 10, 64)
		if err != nil {
			return 0, 0, false
		}
		total += v
		if i == 3 || i == 4 {
			idle += v
		}
	}
	return idle, total, true
}

func cpuBusy(ctx context.Context, window time.Duration) float64 {
	i0, t0, ok := cpuTimes()
	if !ok {
		return -1
	}
	select {
	case <-time.After(window):
	case <-ctx.Done():
		return -1
	}
	i1, t1, ok := cpuTimes()
	if !ok || t1 == t0 {
		return -1
	}
	return 100 * float64((t1-t0)-(i1-i0)) / float64(t1-t0)
}
