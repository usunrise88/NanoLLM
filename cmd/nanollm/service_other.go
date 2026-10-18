//go:build !windows && !linux && !darwin

package main

import (
	"fmt"
	"runtime"
)

// runAsService: only Windows has a service entry point.
func runAsService() (bool, error) { return false, nil }

// serviceCommand has no service manager to talk to on this platform.
func serviceCommand([]string) error {
	return fmt.Errorf("`nanollm service` knows systemd, launchd and the Windows service control manager, and this is %s; "+
		"run `nanollm serve` under whatever supervises processes here", runtime.GOOS)
}
