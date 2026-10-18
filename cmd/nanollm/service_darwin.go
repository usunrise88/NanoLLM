//go:build darwin

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

// launchd: a LaunchAgent in ~/Library/LaunchAgents by default, which runs as
// the user while they are logged in and keeps the data in their Application
// Support (SPEC §10), or with -system a LaunchDaemon in /Library/LaunchDaemons
// for the whole machine.
type launchdJob struct {
	label   string
	plist   string
	domains []string // where it may be loaded, in the order to try
	system  bool
}

func launchdFor(o serviceOptions) (launchdJob, error) {
	label := o.name
	if label == "" {
		label = serviceLabel
	}
	if o.system {
		return launchdJob{
			label:   label,
			plist:   filepath.Join("/Library/LaunchDaemons", label+".plist"),
			domains: []string{"system"},
			system:  true,
		}, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return launchdJob{}, err
	}
	uid := strconv.Itoa(os.Getuid())
	// gui/ is the session of a logged-in user; user/ exists without one (an
	// ssh login, a CI runner) and is where an agent goes when there is no GUI.
	return launchdJob{
		label:   label,
		plist:   filepath.Join(home, "Library", "LaunchAgents", label+".plist"),
		domains: []string{"gui/" + uid, "user/" + uid},
	}, nil
}

func platformService(verb string, o serviceOptions) error {
	if o.manual {
		return errors.New("-manual is not available for launchd: an agent starts at login and a daemon at boot")
	}
	if o.unitDir != "" {
		return errors.New("-unit-dir is for systemd")
	}
	job, err := launchdFor(o)
	if err != nil {
		return err
	}
	if verb == "unit" {
		p, err := launchdParams(o, job, true)
		if err != nil {
			return err
		}
		text, err := renderLaunchd(p)
		if err != nil {
			return err
		}
		fmt.Print(text)
		return nil
	}
	if job.system {
		if err := requireRoot("service " + verb + " -system"); err != nil {
			return err
		}
	} else if os.Geteuid() == 0 && verb != "status" {
		// An agent belongs to a user, and root's agent is not what anyone
		// who typed sudo meant.
		return errors.New("a LaunchAgent is installed for the user who runs it, and this is root;\n" +
			"run it without sudo, or pass -system for a LaunchDaemon")
	}

	switch verb {
	case "install":
		return installLaunchd(job, o)
	case "uninstall":
		return uninstallLaunchd(job)
	case "start":
		if !fileExists(job.plist) {
			return notInstalledLaunchd(job)
		}
		if _, ok := job.loaded(); ok {
			if err := run("launchctl", "kickstart", job.target()); err != nil {
				return err
			}
		} else if err := job.bootstrap(); err != nil {
			return err
		}
		return statusLaunchd(job)
	case "stop":
		if !fileExists(job.plist) {
			return notInstalledLaunchd(job)
		}
		// bootout rather than kill: KeepAlive would not restart a clean exit,
		// but RunAtLoad would start it again at the next login while it stays
		// loaded. `service start` loads it again.
		if domain, ok := job.loaded(); ok {
			if err := run("launchctl", "bootout", domain+"/"+job.label); err != nil {
				return err
			}
		}
		fmt.Printf("%s is stopped\n", job.label)
		return nil
	case "restart":
		if !fileExists(job.plist) {
			return notInstalledLaunchd(job)
		}
		if domain, ok := job.loaded(); ok {
			if err := run("launchctl", "kickstart", "-k", domain+"/"+job.label); err != nil {
				return err
			}
		} else if err := job.bootstrap(); err != nil {
			return err
		}
		return statusLaunchd(job)
	default:
		return statusLaunchd(job)
	}
}

// launchdParams fills in the paths a job runs with.
func launchdParams(o serviceOptions, job launchdJob, preview bool) (unitParams, error) {
	resolve := resolveInstall
	if preview {
		resolve = resolveUnitPreview
	}
	p, err := resolve(o)
	if err != nil {
		return unitParams{}, err
	}
	p.Label = job.label
	p.System = job.system
	p.LogFile = o.logFile
	if p.LogFile == "" {
		if job.system {
			p.LogFile = "/Library/Logs/NanoLLM/nanollm.log"
		} else {
			home, err := os.UserHomeDir()
			if err != nil {
				return unitParams{}, err
			}
			p.LogFile = filepath.Join(home, "Library", "Logs", "NanoLLM", "nanollm.log")
		}
	}
	if p.LogFile, err = filepath.Abs(p.LogFile); err != nil {
		return unitParams{}, err
	}
	if job.system && p.User == "" {
		p.User = "root"
	}
	if !job.system {
		p.User = ""
	}
	return p, nil
}

func installLaunchd(job launchdJob, o serviceOptions) error {
	p, err := launchdParams(o, job, false)
	if err != nil {
		return err
	}
	if job.system && p.User != "root" {
		if _, err := user.Lookup(p.User); err != nil {
			return fmt.Errorf("there is no account %q to run the daemon as: %w", p.User, err)
		}
	}
	// Created here: launchd opens StandardErrorPath before the process runs,
	// and a directory that is not there is a job that never starts.
	for _, dir := range []string{p.DataDir, filepath.Dir(p.LogFile)} {
		if err := os.MkdirAll(dir, 0o750); err != nil {
			return err
		}
	}
	text, err := renderLaunchd(p)
	if err != nil {
		return err
	}
	// Unloaded before the plist is replaced: launchd keeps the definition it
	// loaded, so an upgrade that skipped this would run the old command line.
	if domain, ok := job.loaded(); ok {
		if out, err := output("launchctl", "bootout", domain+"/"+job.label); err != nil {
			fmt.Fprintf(os.Stderr, "warning: launchctl bootout: %v\n%s", err, out)
		}
	}
	if err := writeFileAtomic(job.plist, []byte(text), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", job.plist, err)
	}

	kind := "LaunchAgent (runs while you are logged in)"
	if job.system {
		kind = "LaunchDaemon (runs at boot as " + p.User + ")"
	}
	fmt.Printf("service     %s (%s)\n", job.label, serviceDisplayName)
	fmt.Printf("kind        %s\n", kind)
	fmt.Printf("plist       %s\n", job.plist)
	fmt.Printf("command     %s serve -config %s\n", p.Exe, p.Config)
	fmt.Printf("data        %s\n", p.DataDir)
	fmt.Printf("log         %s\n", p.LogFile)
	if o.noStart {
		fmt.Printf("\nStart it with:\n  %s service start%s\n", p.Exe, map[bool]string{true: " -system", false: ""}[job.system])
		return nil
	}
	if err := job.bootstrap(); err != nil {
		return fmt.Errorf("the plist is written but launchd did not load it: %w\nthe reason may be in %s", err, p.LogFile)
	}
	time.Sleep(time.Second)
	fmt.Println()
	return statusLaunchd(job)
}

func uninstallLaunchd(job launchdJob) error {
	if domain, ok := job.loaded(); ok {
		if out, err := output("launchctl", "bootout", domain+"/"+job.label); err != nil {
			fmt.Fprintf(os.Stderr, "warning: launchctl bootout: %v\n%s", err, out)
		}
	}
	if !fileExists(job.plist) {
		fmt.Printf("the %s service is not installed\n", job.label)
		return nil
	}
	if err := os.Remove(job.plist); err != nil {
		return err
	}
	fmt.Printf("removed %s\n", job.plist)
	fmt.Println("the configuration, the models and the job database were left in place")
	return nil
}

// loaded reports the domain the job is loaded in.
func (j launchdJob) loaded() (string, bool) {
	for _, d := range j.domains {
		if _, err := output("launchctl", "print", d+"/"+j.label); err == nil {
			return d, true
		}
	}
	return "", false
}

func (j launchdJob) target() string {
	if d, ok := j.loaded(); ok {
		return d + "/" + j.label
	}
	return j.domains[0] + "/" + j.label
}

// bootstrap loads the plist into the first domain that takes it.
func (j launchdJob) bootstrap() error {
	var errs []string
	for _, d := range j.domains {
		out, err := output("launchctl", "bootstrap", d, j.plist)
		if err == nil {
			return nil
		}
		errs = append(errs, fmt.Sprintf("%s: %v %s", d, err, strings.TrimSpace(out)))
	}
	return fmt.Errorf("launchctl bootstrap %s: %s", j.plist, strings.Join(errs, "; "))
}

func notInstalledLaunchd(job launchdJob) error {
	flag := ""
	if job.system {
		flag = " -system"
	}
	return fmt.Errorf("the %s service is not installed (%s); register it with `nanollm service install%s`", job.label, job.plist, flag)
}

// statusLaunchd reads launchctl print, which has no stable format but has had
// the same few lines for years: state, pid and the last exit code.
func statusLaunchd(job launchdJob) error {
	if !fileExists(job.plist) {
		return notInstalledLaunchd(job)
	}
	fmt.Printf("service  %s\n", job.label)
	domain, ok := job.loaded()
	if !ok {
		fmt.Printf("state    not loaded (stopped)\n")
		fmt.Printf("plist    %s\n", job.plist)
		return nil
	}
	out, _ := output("launchctl", "print", domain+"/"+job.label)
	props := map[string]string{}
	for _, line := range strings.Split(out, "\n") {
		k, v, ok := strings.Cut(strings.TrimSpace(line), " = ")
		if ok {
			if _, seen := props[k]; !seen {
				props[k] = v
			}
		}
	}
	state := props["state"]
	if state == "" {
		state = "loaded"
	}
	fmt.Printf("state    %s\n", state)
	if pid := props["pid"]; pid != "" {
		fmt.Printf("pid      %s\n", pid)
	}
	if code := props["last exit code"]; code != "" && code != "0" && code != "(never exited)" {
		fmt.Printf("exit     %s\n", code)
	}
	fmt.Printf("domain   %s\n", domain)
	fmt.Printf("plist    %s\n", job.plist)
	return nil
}
