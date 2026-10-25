//go:build !windows

package main

import "os/exec"

// asApplication has nothing to set outside Windows.
func asApplication(*exec.Cmd) {}
