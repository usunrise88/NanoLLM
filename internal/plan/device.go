package plan

import (
	"fmt"
	"strings"
)

// Backend kinds the planner distinguishes. They decide the compute-buffer
// coefficients, the runtime overhead, and whether memory is unified.
const (
	BackendCPU    = "cpu"
	BackendCUDA   = "cuda"
	BackendVulkan = "vulkan"
	BackendMetal  = "metal"
)

// Device is the accelerator a plan places layers on. RAM presets have none.
type Device struct {
	// Selector is how the backend names it, e.g. "llamacpp:cuda0".
	Selector string `json:"selector"`
	// Backend is the kind: cuda, vulkan, metal (or anything else, which plans
	// as an uncalibrated GPU).
	Backend string `json:"backend"`
	// Memory is the device's memory in bytes; 0 when unknown, in which case
	// the preset's budget is taken as what the device has.
	Memory int64 `json:"memory,omitempty"`
	// Unified is set for Apple Silicon, where the GPU maps host memory.
	Unified bool `json:"unified,omitempty"`
}

// Kind is the backend kind of a ggml device id: the id without its trailing
// index, with ggml's name for the Apple GPU ("mtl0") folded into BackendMetal.
func Kind(id string) string {
	kind := strings.TrimRight(id, "0123456789")
	if kind == "mtl" {
		return BackendMetal
	}
	return kind
}

// ParseDevice reads a selector ("llamacpp:vulkan0", "cuda0", "mtl0") into a
// device. The backend kind is the device id without its trailing index (Kind).
func ParseDevice(selector string) (Device, error) {
	s := strings.TrimSpace(selector)
	if s == "" {
		return Device{}, fmt.Errorf("empty device selector")
	}
	if !strings.Contains(s, ":") {
		s = "llamacpp:" + s
	}
	_, id, _ := strings.Cut(s, ":")
	kind := Kind(id)
	if kind == "" {
		return Device{}, fmt.Errorf("device selector %q names no device", selector)
	}
	if kind == BackendCPU {
		return Device{}, fmt.Errorf("%q is the host; RAM presets plan for it without a device", selector)
	}
	return Device{Selector: s, Backend: kind, Unified: kind == BackendMetal}, nil
}

// SyntheticDevice is the device a preset is planned for when none is given:
// devices are not probed yet, so a VRAM preset gets a CUDA GPU of exactly its
// budget and a unified preset a Metal device of its pool. RAM presets need
// none.
func SyntheticDevice(p Preset) *Device {
	switch {
	case p.Budget.Unified > 0:
		return &Device{Selector: "llamacpp:mtl0", Backend: BackendMetal, Memory: p.Budget.Unified, Unified: true}
	case p.Budget.VRAM > 0:
		return &Device{Selector: "llamacpp:cuda0", Backend: BackendCUDA, Memory: p.Budget.VRAM}
	}
	return nil
}

// Runtime overhead the budgets carry besides tensors, KV and compute buffers.
// None of these is calibrated yet; they are deliberately round and generous,
// and the bench (1.11) measures them per machine.
const (
	// HostOverhead is the process itself: the Go server, libllama and the
	// ggml backends' code, the vocabulary tables and graph metadata.
	HostOverhead = 192 << 20
	// GPUOverheadCUDA and GPUOverheadVulkan are device memory the driver
	// takes before the first buffer: context, cuBLAS workspace, pipelines.
	GPUOverheadCUDA   = 320 << 20
	GPUOverheadVulkan = 160 << 20
)

func gpuOverhead(d *Device) int64 {
	switch {
	case d == nil || d.Unified:
		return 0
	case d.Backend == BackendVulkan:
		return GPUOverheadVulkan
	default:
		return GPUOverheadCUDA
	}
}
