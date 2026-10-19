package stats

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

var started = time.Now()

// Process is the server process's own footprint: what a soak test watches
// for growth.
type Process struct {
	UptimeS    float64 `json:"uptime_s"`
	Goroutines int     `json:"goroutines"`
	Threads    int     `json:"threads,omitempty"`
	// RSS, RSSPeak and Locked are VmRSS, VmHWM and VmLck of /proc/self/status
	// (Linux), in bytes. Locked is memory pinned with mlock: the locked
	// weights of the ram-* modes.
	RSS     int64 `json:"rss"`
	RSSPeak int64 `json:"rss_peak"`
	Locked  int64 `json:"locked"`
	// GoHeap and GoSys are the Go runtime's own: the C++ heap of libllama is
	// in RSS but in neither.
	GoHeap uint64 `json:"go_heap"`
	GoSys  uint64 `json:"go_sys"`
	GCs    uint32 `json:"gc_cycles"`
}

// ReadProcess samples the process.
func ReadProcess() Process {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	p := Process{
		UptimeS: time.Since(started).Seconds(), Goroutines: runtime.NumGoroutine(),
		GoHeap: ms.HeapAlloc, GoSys: ms.Sys, GCs: ms.NumGC,
	}
	if b, err := os.ReadFile("/proc/self/status"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			k, v, ok := strings.Cut(line, ":")
			if !ok {
				continue
			}
			v = strings.TrimSpace(v)
			kb := func() int64 { n, _ := strconv.ParseInt(strings.TrimSuffix(v, " kB"), 10, 64); return n << 10 }
			switch k {
			case "VmRSS":
				p.RSS = kb()
			case "VmHWM":
				p.RSSPeak = kb()
			case "VmLck":
				p.Locked = kb()
			case "Threads":
				p.Threads, _ = strconv.Atoi(v)
			}
		}
	}
	return p
}
