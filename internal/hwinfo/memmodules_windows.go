//go:build windows

package hwinfo

// MemoryModules is the number of populated memory slots from SMBIOS, 0 when
// the firmware does not say. One module on a laptop means single-channel
// memory, which halves the bandwidth decoding is limited by.
func MemoryModules() int {
	if info, ok := smbios(); ok {
		return info.memModules
	}
	return 0
}
