// Package hwinfo describes the machine for a benchmark result: CPU, memory,
// GPUs, the system's maker and model (the laptop model, where the firmware
// says), OS, power state and disk type (docs/BENCHMARKS.md).
//
// It collects nothing that identifies a machine or a person: no serial
// numbers, UUIDs, MAC addresses, hostnames or user names are read, so none can
// be sent by mistake. Everything is gathered without cgo — sysfs and procfs on
// Linux, SMBIOS and Win32 calls on Windows, sysctl on macOS — and a source that
// fails leaves its fields empty rather than failing the profile.
package hwinfo

import (
	"context"
	"regexp"
	"runtime"
	"strings"

	"github.com/usunrise88/nanollm/benchfmt"
)

// Collect profiles this machine. GPUs come from the backends (they know the
// devices they will run on) and are passed in; Collect adds what the OS
// knows about them, such as the NVIDIA power limit.
func Collect(ctx context.Context, gpus []benchfmt.GPU) benchfmt.Machine {
	m := collect(ctx)
	m.GPUs = append(m.GPUs, gpus...)
	enrichGPUs(ctx, &m)
	if m.System.FormFactor == "" {
		m.System.FormFactor = benchfmt.FormUnknown
	}
	if m.OS.Name == "" {
		m.OS.Name = runtime.GOOS
	}
	m.CPU.Model = CleanCPUName(m.CPU.Model)
	if m.CPU.Vendor == "" {
		m.CPU.Vendor = cpuVendor(m.CPU.Model)
	}
	// Dedicated servers often report the chassis of a tower or nothing at
	// all; an EPYC or a Xeon is a server whatever the firmware says.
	if serverCPU.MatchString(m.CPU.Model) && (m.System.FormFactor == benchfmt.FormDesktop || m.System.FormFactor == benchfmt.FormUnknown) {
		m.System.FormFactor = benchfmt.FormServer
	}
	return m
}

var serverCPU = regexp.MustCompile(`(?i)\b(EPYC|Xeon)\b`)

// FormFactorOfChassis maps an SMBIOS chassis type (DMI chassis_type) to a
// form factor.
func FormFactorOfChassis(t int) string {
	switch t {
	case 8, 9, 10, 11, 14, 30, 31, 32:
		return benchfmt.FormLaptop
	case 3, 4, 5, 6, 7, 13, 15, 16, 24, 34, 35, 36:
		return benchfmt.FormDesktop
	case 17, 23, 25, 28, 29:
		return benchfmt.FormServer
	case 1:
		// "Other": virtual machines report it.
		return benchfmt.FormUnknown
	}
	return benchfmt.FormUnknown
}

// ChassisName is the SMBIOS name of a chassis type, for display.
func ChassisName(t int) string {
	names := map[int]string{
		3: "desktop", 4: "low-profile desktop", 6: "mini tower", 7: "tower", 8: "portable", 9: "laptop",
		10: "notebook", 11: "hand held", 13: "all in one", 14: "sub notebook", 15: "space-saving",
		17: "main server chassis", 23: "rack mount", 30: "tablet", 31: "convertible", 32: "detachable",
		35: "mini pc", 36: "stick pc",
	}
	return names[t]
}

// virtualVendors are system makers that mean "a VM", whatever the chassis says.
var virtualVendors = []string{"qemu", "kvm", "vmware", "innotek", "virtualbox", "xen", "microsoft corporation virtual", "amazon ec2", "google compute", "firecracker", "bochs", "parallels"}

func looksVirtual(vendor, product string) bool {
	s := strings.ToLower(vendor + " " + product)
	for _, v := range virtualVendors {
		if strings.Contains(s, v) {
			return true
		}
	}
	return strings.Contains(s, "virtual machine")
}

// CleanCPUName drops the decoration CPU brand strings carry: "(R)", "(TM)",
// "CPU", "@ 2.30GHz", "12th Gen", trailing core counts and runs of spaces.
// The service normalises further; this keeps what is sent readable.
func CleanCPUName(s string) string {
	for _, junk := range []string{"(R)", "(r)", "(TM)", "(tm)", "®", "™"} {
		s = strings.ReplaceAll(s, junk, "")
	}
	if i := strings.Index(s, "@"); i > 0 {
		s = s[:i]
	}
	fields := strings.Fields(s)
	var out []string
	for i := 0; i < len(fields); i++ {
		f := fields[i]
		switch {
		case f == "CPU" || f == "Processor" || f == "processor":
			continue
		case strings.HasSuffix(f, "th") && i+1 < len(fields) && fields[i+1] == "Gen":
			i++ // "12th Gen"
			continue
		case coreCount.MatchString(f):
			continue // "16-Core"
		}
		out = append(out, f)
	}
	return strings.Join(out, " ")
}

var coreCount = regexp.MustCompile(`^\d+-[Cc]ore$`)

func cpuVendor(model string) string {
	m := strings.ToLower(model)
	switch {
	case strings.Contains(m, "intel"):
		return "Intel"
	case strings.Contains(m, "amd") || strings.Contains(m, "ryzen") || strings.Contains(m, "epyc"):
		return "AMD"
	case strings.Contains(m, "apple"):
		return "Apple"
	case strings.Contains(m, "snapdragon") || strings.Contains(m, "qualcomm"):
		return "Qualcomm"
	}
	return ""
}

// majorOf keeps the leading component of a version: "560.94" → "560".
func majorOf(v string) string {
	v = strings.TrimSpace(v)
	if i := strings.IndexAny(v, ". "); i > 0 {
		return v[:i]
	}
	return v
}

// majorMinor keeps two components: "6.8.0-45-generic" → "6.8".
func majorMinor(v string) string {
	parts := strings.SplitN(strings.TrimSpace(v), ".", 3)
	if len(parts) < 2 {
		return v
	}
	minor := parts[1]
	if i := strings.IndexFunc(minor, func(r rune) bool { return r < '0' || r > '9' }); i >= 0 {
		minor = minor[:i]
	}
	return parts[0] + "." + minor
}
