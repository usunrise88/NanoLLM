//go:build !windows && !linux

package hwinfo

// ReadPowerMode: the Low Power Mode of macOS is read from pmset by
// collect_darwin's Power.Source only.
func ReadPowerMode() PowerMode { return PowerMode{} }
