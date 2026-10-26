package main

import (
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

// asApplication starts the child as a GUI application starts a background
// process: without a console window, so that its own children must not open
// one either.
func asApplication(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windows.CREATE_NO_WINDOW}
}
