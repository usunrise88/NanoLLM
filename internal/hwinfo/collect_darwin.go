//go:build darwin

package hwinfo

import (
	"context"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sys/unix"

	"github.com/usunrise88/nanollm/benchfmt"
)

func collect(ctx context.Context) benchfmt.Machine {
	var m benchfmt.Machine
	m.CPU.Model, _ = unix.Sysctl("machdep.cpu.brand_string")
	if n, err := unix.SysctlUint32("hw.physicalcpu"); err == nil {
		m.CPU.Cores = int(n)
	}
	if n, err := unix.SysctlUint32("hw.logicalcpu"); err == nil {
		m.CPU.Threads = int(n)
	}
	if n, err := unix.SysctlUint32("hw.perflevel0.physicalcpu"); err == nil {
		m.CPU.PCores = int(n)
		if e, err := unix.SysctlUint32("hw.perflevel1.physicalcpu"); err == nil {
			m.CPU.ECores = int(e)
		}
	}
	if b, err := unix.SysctlUint64("hw.memsize"); err == nil {
		m.Memory.TotalBytes = int64(b)
	}
	if strings.Contains(m.CPU.Model, "Apple") {
		m.CPU.Vendor = "Apple"
		m.CPU.ISA = []string{"neon", "dotprod"}
	}

	m.System.Vendor = "Apple"
	m.System.Product, _ = unix.Sysctl("hw.model")
	name := systemProfiler(ctx)
	m.System.Family = name
	m.System.FormFactor = benchfmt.FormDesktop
	if strings.Contains(name, "MacBook") || strings.HasPrefix(m.System.Product, "MacBook") {
		m.System.FormFactor = benchfmt.FormLaptop
	}
	if strings.Contains(strings.ToLower(m.System.Product), "virtual") {
		m.System.FormFactor = benchfmt.FormVM
	}

	m.OS.Name = "macos"
	m.OS.Version, _ = unix.Sysctl("kern.osproductversion")
	if rel, err := unix.Sysctl("kern.osrelease"); err == nil {
		m.OS.KernelMM = majorMinor(rel)
	}
	m.Power.Source = macPower(ctx)
	m.Disk.Type = "ssd"
	if cores := gpuCores(ctx); cores > 0 {
		m.GPUs = append(m.GPUs, benchfmt.GPU{Name: m.CPU.Model + " GPU", Vendor: "Apple", Cores: cores, Unified: true})
	}
	return m
}

func run(ctx context.Context, name string, args ...string) string {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return ""
	}
	return string(out)
}

var modelName = regexp.MustCompile(`(?m)^\s*Model Name:\s*(.+)$`)

func systemProfiler(ctx context.Context) string {
	if m := modelName.FindStringSubmatch(run(ctx, "system_profiler", "SPHardwareDataType")); m != nil {
		return strings.TrimSpace(m[1])
	}
	return ""
}

var gpuCoreCount = regexp.MustCompile(`"gpu-core-count"\s*=\s*(\d+)`)

// gpuCores is the Apple GPU's core count: an M3 with 8 and one with 10 are
// different machines for a benchmark, and the name does not say which.
func gpuCores(ctx context.Context) int {
	if m := gpuCoreCount.FindStringSubmatch(run(ctx, "ioreg", "-rc", "AGXAccelerator", "-d", "1")); m != nil {
		n, _ := strconv.Atoi(m[1])
		return n
	}
	return 0
}

func macPower(ctx context.Context) string {
	out := run(ctx, "pmset", "-g", "batt")
	switch {
	case strings.Contains(out, "AC Power"):
		return "ac"
	case strings.Contains(out, "Battery Power"):
		return "battery"
	}
	return ""
}
