//go:build linux

package hwinfo

import (
	"os"
	"strings"
)

// ReadPowerMode reads the ACPI platform profile and the CPU governor.
func ReadPowerMode() PowerMode {
	read := func(p string) string {
		b, _ := os.ReadFile(p)
		return strings.TrimSpace(string(b))
	}
	var pm PowerMode
	if p := read("/sys/firmware/acpi/platform_profile"); p != "" {
		pm.Name = p
		pm.Saver = p == "low-power" || p == "quiet"
	}
	if g := read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"); g != "" {
		if pm.Name == "" {
			pm.Name = g
		}
		if g == "powersave" && read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver") != "intel_pstate" {
			// intel_pstate's powersave is its normal, frequency-scaling mode.
			pm.Saver = true
		}
	}
	return pm
}
