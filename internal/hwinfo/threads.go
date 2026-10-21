package hwinfo

import (
	"context"
	"io/fs"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// DefaultThreads is the number of CPU threads inference should use when the
// configuration says 0: the physical cores this process may run on. llama.cpp's
// own default is 4, which leaves most of a server idle, and SMT siblings only
// compete for the same vector units and memory bandwidth (docs/STAGE2.md:
// 16 threads on 16 cores of the EPYC is 4x faster than 32 on 32).
//
// On hybrid Intel and Apple CPUs only the performance cores count: a thread on
// an efficiency core holds back every matrix product it shares. A process
// pinned with taskset or numactl, or inside a cpuset, gets the cores it is
// allowed on, not the machine's.
func DefaultThreads() int {
	defaultThreadsOnce.Do(func() { defaultThreads = max(1, detectThreads()) })
	return defaultThreads
}

var (
	defaultThreadsOnce sync.Once
	defaultThreads     int
)

// DefaultBatchThreads is the size of the team that should process prompt
// chunks when neither the configuration nor the request says: every logical CPU
// this process may run on, on hybrid x86 CPUs (Intel P+E), and 0 (follow
// DefaultThreads) elsewhere. Decoding is limited by memory bandwidth, which
// six performance cores of an i7-12700H already saturate, but prefill is
// limited by arithmetic and gains from the efficiency cores and the second
// threads of the performance cores: 6 threads 125-128 tok/s, 14 138, 20 155-172
// (bench/results/rusunrisebook/stage0-2026-10-21). Apple's hybrid parts are not
// measured and keep following DefaultThreads.
func DefaultBatchThreads() int {
	defaultBatchOnce.Do(func() {
		m := collect(context.Background())
		if runtime.GOARCH == "amd64" && m.CPU.PCores > 0 && m.CPU.ECores > 0 {
			defaultBatch = runtime.NumCPU()
		}
	})
	return defaultBatch
}

var (
	defaultBatchOnce sync.Once
	defaultBatch     int
)

// detectThreads is DefaultThreads without the cache.
func detectThreads() int {
	if n := allowedPhysicalCores(); n > 0 {
		return n
	}
	m := collect(context.Background())
	switch {
	case m.CPU.PCores > 0:
		return m.CPU.PCores
	case m.CPU.Cores > 0:
		return m.CPU.Cores
	}
	return runtime.NumCPU()
}

// physicalCores counts the distinct (package, core) pairs among the given
// logical CPUs, reading the topology from fsys. It returns 0 when sysfs does
// not describe a CPU, so the caller can fall back.
func physicalCores(fsys fs.FS, cpus []int) int {
	seen := map[string]bool{}
	for _, cpu := range cpus {
		base := "sys/devices/system/cpu/cpu" + strconv.Itoa(cpu) + "/topology/"
		core, err := fs.ReadFile(fsys, base+"core_id")
		if err != nil {
			return 0
		}
		pkg, _ := fs.ReadFile(fsys, base+"physical_package_id")
		seen[strings.TrimSpace(string(pkg))+"/"+strings.TrimSpace(string(core))] = true
	}
	return len(seen)
}
