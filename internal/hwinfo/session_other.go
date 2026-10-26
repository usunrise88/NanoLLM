//go:build !windows

package hwinfo

func inServiceSession(int) bool { return false }
