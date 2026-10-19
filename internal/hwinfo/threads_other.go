//go:build !linux

package hwinfo

func allowedPhysicalCores() int { return 0 }
