//go:build !windows

package hwinfo

// GPUDrivers: only Windows keeps driver versions where a user can read them.
// NVIDIA's version comes from nvidia-smi on every system (enrichGPUs), and
// Linux's Mesa version is part of the Vulkan device description.
func GPUDrivers() []GPUDriver { return nil }
