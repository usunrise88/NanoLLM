//go:build !windows

package hwinfo

// MemoryModules: SMBIOS is readable without privileges only on Windows.
func MemoryModules() int { return 0 }
