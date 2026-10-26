//go:build windows

package hwinfo

import (
	"strconv"

	"golang.org/x/sys/windows/registry"
)

// GPUDrivers lists the display drivers from the display adapter class in
// HKLM (the class GUID {4d36e968-…}); the keys are readable by any user.
func GPUDrivers() []GPUDriver {
	class, err := registry.OpenKey(registry.LOCAL_MACHINE, `SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}`, registry.ENUMERATE_SUB_KEYS)
	if err != nil {
		return nil
	}
	defer class.Close()
	names, err := class.ReadSubKeyNames(-1)
	if err != nil {
		return nil
	}
	var out []GPUDriver
	for _, n := range names {
		if _, err := strconv.Atoi(n); err != nil {
			continue // "Configuration", "Properties"
		}
		k, err := registry.OpenKey(class, n, registry.QUERY_VALUE)
		if err != nil {
			continue
		}
		desc, _, _ := k.GetStringValue("DriverDesc")
		ver, _, _ := k.GetStringValue("DriverVersion")
		date, _, _ := k.GetStringValue("DriverDate")
		k.Close()
		if desc == "" {
			continue
		}
		out = append(out, GPUDriver{Name: desc, Vendor: vendorOf(desc), Version: ver, Date: parseDriverDate(date)})
	}
	return out
}
