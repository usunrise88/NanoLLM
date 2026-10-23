//go:build windows

package hwinfo

import "golang.org/x/sys/windows"

func inServiceSession(pid int) bool {
	var id uint32
	if err := windows.ProcessIdToSessionId(uint32(pid), &id); err != nil {
		return false
	}
	return id == 0
}
