//go:build linux

package hwinfo

import (
	"context"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
)

func collect(ctx context.Context) benchfmt.Machine {
	m := linuxProfile(os.DirFS("/"))
	if m.OS.WSL {
		wslHost(ctx, &m)
	}
	return m
}

// wslHost asks Windows, through WSL interop, what machine this is: inside
// WSL2 the Linux DMI files describe a Hyper-V VM, not the laptop.
func wslHost(ctx context.Context, m *benchfmt.Machine) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
		"$c=Get-CimInstance Win32_ComputerSystem; $p=Get-CimInstance Win32_ComputerSystemProduct; $e=Get-CimInstance Win32_SystemEnclosure;"+
			"Write-Output ('{0}|{1}|{2}|{3}|{4}' -f $c.Manufacturer,$c.SystemFamily,$p.Version,$c.Model,($e.ChassisTypes -join ','))").Output()
	if err != nil {
		return
	}
	f := strings.Split(strings.TrimSpace(string(out)), "|")
	if len(f) < 5 {
		return
	}
	m.System.Vendor = strings.TrimSpace(f[0])
	m.System.Product = strings.TrimSpace(f[3])
	m.System.Family = marketingName(strings.TrimSpace(f[1]), strings.TrimSpace(f[2]), m.System.Product)
	m.System.FormFactor = benchfmt.FormUnknown
	if t, err := strconv.Atoi(strings.Split(strings.TrimSpace(f[4]), ",")[0]); err == nil {
		m.System.FormFactor = FormFactorOfChassis(t)
		m.System.Chassis = ChassisName(t)
	}
}

func cpuNameRaw() string {
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		if k, v, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(k) == "model name" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}
