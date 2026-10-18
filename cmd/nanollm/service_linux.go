//go:build linux

package main

import (
	"errors"
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const systemdUnitDir = "/etc/systemd/system"

// systemdRunning distinguishes "systemctl is installed" from "systemd is PID
// 1", which a container often gets to disagree about: a systemctl call on a
// machine that is not running systemd fails with "Failed to connect to bus".
func systemdRunning() bool {
	st, err := os.Stat("/run/systemd/system")
	return err == nil && st.IsDir()
}

func platformService(verb string, o serviceOptions) error {
	if o.system {
		return errors.New("-system is for launchd; a systemd unit always runs for the whole machine")
	}
	name := o.name
	if name == "" {
		name = serviceName
	}
	unitDir := o.unitDir
	if unitDir == "" {
		unitDir = systemdUnitDir
	}
	unit := filepath.Join(unitDir, name+".service")

	if verb == "unit" {
		p, err := resolveUnitPreview(o)
		if err != nil {
			return err
		}
		if p.User == "" {
			p.User = serviceName
		}
		text, err := renderSystemd(p)
		if err != nil {
			return err
		}
		fmt.Print(text)
		return nil
	}

	if !systemdRunning() {
		exe, _ := exePath()
		return fmt.Errorf("systemd is not running here (a container, or another init), so there is no service to %s;\n"+
			"run the server under whatever supervises processes here:\n  %s serve -config FILE", verb, exe)
	}
	switch verb {
	case "install":
		return installSystemd(name, unit, o)
	case "uninstall":
		return uninstallSystemd(name, unit)
	case "start", "stop", "restart":
		if err := requireRoot("service " + verb); err != nil {
			return err
		}
		if !fileExists(unit) {
			return fmt.Errorf("the %s service is not installed; register it with `sudo nanollm service install`", name)
		}
		if err := run("systemctl", verb, name); err != nil {
			return fmt.Errorf("%w\nthe reason is in: journalctl -u %s -n 50", err, name)
		}
		return statusSystemd(name, unit)
	default:
		return statusSystemd(name, unit)
	}
}

func installSystemd(name, unit string, o serviceOptions) error {
	if err := requireRoot("service install"); err != nil {
		return err
	}
	p, err := resolveInstall(o)
	if err != nil {
		return err
	}
	if p.User == "" {
		p.User = serviceName
	}
	u, err := user.Lookup(p.User)
	if err != nil {
		return fmt.Errorf("there is no account %q to run the service as: %w\n"+
			"create one (install.sh does): sudo useradd --system --home-dir %s --shell /usr/sbin/nologin %s\n"+
			"or pass -user", p.User, err, p.DataDir, p.User)
	}
	if g, err := user.LookupGroupId(u.Gid); err == nil {
		p.Group = g.Name
	}

	// ReadWritePaths must exist when the unit starts, or systemd fails to set
	// up the mount namespace with a message that names no path.
	if err := ensureOwnedDir(p.DataDir, u); err != nil {
		return err
	}

	text, err := renderSystemd(p)
	if err != nil {
		return err
	}
	// The binary must be readable by the account: an archive unpacked by
	// root under a umask of 077 is a service that cannot exec.
	if st, err := os.Stat(p.Exe); err == nil && st.Mode().Perm()&0o005 != 0o005 && u.Uid != "0" {
		fmt.Fprintf(os.Stderr, "warning: %s is not world-readable and executable; %s may not be able to run it\n", p.Exe, p.User)
	}
	if err := writeFileAtomic(unit, []byte(text), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", unit, err)
	}
	if err := run("systemctl", "daemon-reload"); err != nil {
		return err
	}
	if !o.manual {
		if err := run("systemctl", "enable", "--quiet", name); err != nil {
			return err
		}
	}

	fmt.Printf("service     %s (%s)\n", name, serviceDisplayName)
	fmt.Printf("unit        %s\n", unit)
	fmt.Printf("command     %s serve -config %s\n", p.Exe, p.Config)
	fmt.Printf("account     %s\n", p.User)
	fmt.Printf("data        %s\n", p.DataDir)
	fmt.Printf("start       %s\n", map[bool]string{true: "manual", false: "at boot"}[o.manual])
	if o.noStart {
		fmt.Printf("\nStart it with:\n  sudo %s service start\n", p.Exe)
		return nil
	}
	// restart rather than start: on an upgrade the old binary is running.
	if err := run("systemctl", "restart", name); err != nil {
		return fmt.Errorf("the service is registered but did not start: %w\nthe reason is in: journalctl -u %s -n 50", err, name)
	}
	state := waitActive(name, 10*time.Second)
	fmt.Printf("state       %s\n", state)
	fmt.Printf("\nThe server is starting; it answers /readyz once the default model is loaded.\n"+
		"  %s service status\n  journalctl -u %s -f\n", p.Exe, name)
	return nil
}

// ensureOwnedDir creates dir for the account when it does not exist. An
// existing directory is left as it is: it may be shared, and changing its
// owner is the installer's decision, not this command's.
func ensureOwnedDir(dir string, u *user.User) error {
	if _, err := os.Stat(dir); err == nil {
		return nil
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return err
	}
	uid, uidErr := strconv.Atoi(u.Uid)
	gid, gidErr := strconv.Atoi(u.Gid)
	if uidErr == nil && gidErr == nil {
		return os.Chown(dir, uid, gid)
	}
	return nil // not a numeric id: nothing chown could take
}

// waitActive polls until the unit settles, so that install can say whether it
// came up rather than assume.
func waitActive(name string, timeout time.Duration) string {
	deadline := time.Now().Add(timeout)
	for {
		out, _ := output("systemctl", "is-active", name)
		state := strings.TrimSpace(out)
		if state == "active" || state == "failed" || time.Now().After(deadline) {
			return state
		}
		time.Sleep(300 * time.Millisecond)
	}
}

func uninstallSystemd(name, unit string) error {
	if err := requireRoot("service uninstall"); err != nil {
		return err
	}
	if !fileExists(unit) {
		// Removing what is not there is what the caller wanted: an uninstall
		// script must be safe to run twice.
		fmt.Printf("the %s service is not installed\n", name)
		return nil
	}
	if out, err := output("systemctl", "disable", "--now", name); err != nil {
		fmt.Fprintf(os.Stderr, "warning: systemctl disable --now %s: %v\n%s", name, err, out)
	}
	if err := os.Remove(unit); err != nil {
		return err
	}
	if err := run("systemctl", "daemon-reload"); err != nil {
		return err
	}
	_, _ = output("systemctl", "reset-failed", name)
	fmt.Printf("removed %s\n", unit)
	fmt.Println("the configuration, the models and the job database were left in place")
	return nil
}

// statusSystemd prints the questions behind most status checks: is it
// running, since when, which binary and configuration, and where the log is.
// It does not fail on a stopped service, which is an answer, not an error.
func statusSystemd(name, unit string) error {
	if !fileExists(unit) {
		return fmt.Errorf("the %s service is not installed; register it with `sudo nanollm service install`", name)
	}
	out, err := output("systemctl", "show", name,
		"-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "UnitFileState",
		"-p", "ExecMainStatus", "-p", "ActiveEnterTimestamp", "-p", "User")
	if err != nil {
		return fmt.Errorf("systemctl show %s: %w\n%s", name, err, out)
	}
	props := map[string]string{}
	for _, line := range strings.Split(out, "\n") {
		if k, v, ok := strings.Cut(line, "="); ok {
			props[k] = v
		}
	}
	fmt.Printf("service  %s\n", name)
	fmt.Printf("state    %s (%s)\n", props["ActiveState"], props["SubState"])
	if pid := props["MainPID"]; pid != "" && pid != "0" {
		fmt.Printf("pid      %s\n", pid)
	}
	if since := props["ActiveEnterTimestamp"]; since != "" && props["ActiveState"] == "active" {
		fmt.Printf("since    %s\n", since)
	}
	if props["ActiveState"] != "active" && props["ExecMainStatus"] != "" && props["ExecMainStatus"] != "0" {
		fmt.Printf("exit     %s\n", props["ExecMainStatus"])
	}
	fmt.Printf("start    %s\n", props["UnitFileState"])
	fmt.Printf("account  %s\n", props["User"])
	fmt.Printf("unit     %s\n", unit)
	fmt.Printf("log      journalctl -u %s -f\n", name)
	return nil
}
