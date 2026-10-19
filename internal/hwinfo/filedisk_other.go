//go:build !linux

package hwinfo

import "github.com/usunrise88/nanollm/benchfmt"

// FileDisk is Linux-only: elsewhere the device's queue is not read.
func FileDisk(string) []benchfmt.Field { return nil }
