//go:build !windows && !linux

package hwinfo

// DiskOf: macOS machines have SSDs (collect_darwin says so); elsewhere the
// kind is not read.
func DiskOf(string) DiskKind { return DiskKind{Type: DiskUnknown} }
