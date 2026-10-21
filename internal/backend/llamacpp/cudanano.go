package llamacpp

import (
	"log/slog"
	"runtime"
	"slices"
	"strings"
	"unsafe"

	"github.com/ebitengine/purego"
)

// setCudaNano sets one switch of NanoLLM's ggml-cuda (patches/llama.cpp/0008, nano.cuh) through the CUDA backend's
// ggml_backend_cuda_nano_set, which only the registry's proc address table reaches. The switches are what
// GGML_CUDA_NANO_<NAME> sets in the environment; an environment variable set after the process started does not reach the
// C runtime of a Windows DLL, so the host program uses this. It reports whether the backend has such a switch; upstream's
// ggml-cuda has none.
func setCudaNano(name string, value int) bool {
	for i := uintptr(0); i < ggmlBackendRegCount(); i++ {
		reg := ggmlBackendRegGet(i)
		if !strings.EqualFold(ggmlBackendRegName(reg), "cuda") {
			continue
		}
		p := ggmlBackendRegGetProcAddress(reg, "ggml_backend_cuda_nano_set")
		if p == 0 {
			return false
		}
		cname := append([]byte(name), 0)
		r, _, _ := purego.SyscallN(p, uintptr(unsafe.Pointer(&cname[0])), uintptr(int64(value)))
		runtime.KeepAlive(cname)
		return r&0xff != 0
	}
	return false
}

// applyCudaNano sets a plan's switches before its model loads, in name order; a name the backend does not know is
// logged and skipped.
func applyCudaNano(settings map[string]int) {
	names := make([]string, 0, len(settings))
	for name := range settings {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		if setCudaNano(name, settings[name]) {
			slog.Info("ggml-cuda switch set", "name", name, "value", settings[name])
		} else {
			slog.Warn("ggml-cuda has no such switch (or is upstream's)", "name", name)
		}
	}
}
