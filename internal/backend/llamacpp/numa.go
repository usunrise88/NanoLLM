package llamacpp

import (
	"log/slog"
	"os"
	"strings"

	"github.com/ebitengine/purego"
)

// EnvNUMA selects ggml's own NUMA strategy, which pins its compute threads:
//
//	distribute  thread i runs on node i mod n, so every node's cores work
//	isolate     all threads run on the node the process started on
//	numactl     all threads keep the cpuset numactl gave the process
//
// Under any of them ggml also splits the matrix rows statically, thread k
// reading the k-th slab. Memory is placed by whoever first touches it, so with
// the patched repack (patches/next/0003, GGML_REPACK_THREADS equal to the
// decode thread count) every thread reads weights on its own node. Without
// the variable the memory policy of internal/numa decides (interleave).
// Experimental: docs/AUTOTUNE.md has the measurements.
const EnvNUMA = "NANOLLM_GGML_NUMA"

// ggml_numa_strategy.
var numaStrategies = map[string]uintptr{"distribute": 1, "isolate": 2, "numactl": 3}

// initNUMA applies EnvNUMA through the CPU backend's ggml_backend_cpu_numa_init,
// which is only reachable through the registry's proc address table.
func initNUMA() {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(EnvNUMA)))
	if v == "" || v == "off" {
		return
	}
	strategy, ok := numaStrategies[v]
	if !ok {
		slog.Warn("llama.cpp NUMA strategy unknown", "env", EnvNUMA, "value", v)
		return
	}
	for i := uintptr(0); i < ggmlBackendRegCount(); i++ {
		reg := ggmlBackendRegGet(i)
		if !strings.EqualFold(ggmlBackendRegName(reg), "cpu") {
			continue
		}
		p := ggmlBackendRegGetProcAddress(reg, "ggml_backend_cpu_numa_init")
		if p == 0 {
			slog.Warn("llama.cpp NUMA strategy not applied", "reason", "the CPU backend has no ggml_backend_cpu_numa_init")
			return
		}
		purego.SyscallN(p, strategy)
		slog.Info("llama.cpp NUMA strategy", "strategy", v)
		return
	}
	slog.Warn("llama.cpp NUMA strategy not applied", "reason", "no CPU backend")
}
