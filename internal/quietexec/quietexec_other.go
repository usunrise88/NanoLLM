//go:build !windows

package quietexec

import "os/exec"

// hide has nothing to do: no other system opens a window for a child.
func hide(*exec.Cmd) {}

// HasConsole is true: only Windows gives a process a console window or not.
func HasConsole() bool { return true }
