//go:build windows

package quietexec

import (
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

var procGetConsoleWindow = windows.NewLazySystemDLL("kernel32.dll").NewProc("GetConsoleWindow")

// hide gives the child no console window when this process has none. Under a
// console the child shares it, as before: no window appears, and Ctrl+C
// reaches it.
func hide(cmd *exec.Cmd) {
	if HasConsole() {
		return
	}
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= windows.CREATE_NO_WINDOW
}

// HasConsole reports whether this process has a console window, visible or
// not.
func HasConsole() bool {
	if procGetConsoleWindow.Find() != nil {
		return true
	}
	h, _, _ := procGetConsoleWindow.Call()
	return h != 0
}
