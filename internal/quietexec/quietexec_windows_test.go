//go:build windows

package quietexec

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"

	"golang.org/x/sys/windows"
)

const helperEnv = "NANOLLM_QUIETEXEC_HELPER"

var procIsWindowVisible = windows.NewLazySystemDLL("user32.dll").NewProc("IsWindowVisible")

// TestHelperProcess is the child and the grandchild of the test below, not a
// test: as the child it starts the grandchild through Command and passes on
// what it says; as the grandchild it says whether it has a visible console
// window.
func TestHelperProcess(t *testing.T) {
	switch os.Getenv(helperEnv) {
	case "child":
		cmd := Command(context.Background(), os.Args[0], "-test.run=^TestHelperProcess$")
		cmd.Env = append(os.Environ(), helperEnv+"=grandchild")
		out, err := cmd.Output()
		if err != nil {
			os.Stdout.WriteString("error: " + err.Error())
			os.Exit(1)
		}
		os.Stdout.Write(out)
		os.Exit(0)
	case "grandchild":
		h, _, _ := procGetConsoleWindow.Call()
		visible := false
		if h != 0 {
			v, _, _ := procIsWindowVisible.Call(h)
			visible = v != 0
		}
		if visible {
			os.Stdout.WriteString("visible")
		} else {
			os.Stdout.WriteString("hidden")
		}
		os.Exit(0)
	}
}

// A process with no console (an application starts nanollm managed so, and a
// service has none) starts its helpers without a window.
func TestNoWindowForTheChildOfAProcessWithoutAConsole(t *testing.T) {
	for name, flags := range map[string]uint32{
		"detached":  windows.DETACHED_PROCESS,
		"no window": windows.CREATE_NO_WINDOW,
	} {
		cmd := exec.Command(os.Args[0], "-test.run=^TestHelperProcess$")
		cmd.Env = append(os.Environ(), helperEnv+"=child")
		cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: flags}
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("%s: %v: %s", name, err, out)
		}
		if got := strings.TrimSpace(string(out)); got != "hidden" {
			t.Errorf("%s: the helper's console is %q", name, got)
		}
	}
}
