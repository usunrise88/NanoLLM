// Package quietexec starts the helper programs nanollm runs (its own probe and
// benchmark workers, nvidia-smi, powercfg, PowerShell) so that none of them
// shows a window. On Windows a console program started by a process that has
// no console of its own (a service, or nanollm managed, which an application
// starts with CREATE_NO_WINDOW) gets a new console window, and the user sees
// it flash for every probe. Elsewhere, and under a console, a command is an
// ordinary exec.CommandContext.
package quietexec

import (
	"context"
	"os/exec"
)

// Command is exec.CommandContext that starts the program without a console
// window when this process has none.
func Command(ctx context.Context, name string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, args...)
	hide(cmd)
	return cmd
}
