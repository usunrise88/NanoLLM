//go:build linux || darwin

package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/config"
)

// managerTimeout bounds one systemctl or launchctl call. Starting a unit
// returns once it is forked; stopping waits for the 45-second stop timeout.
const managerTimeout = 2 * time.Minute

// runAsService is the Windows service entry point, and there is nothing to
// enter here: systemd and launchd run an ordinary foreground process and
// signal it, which is what serve already is.
func runAsService() (bool, error) { return false, nil }

// serviceOptions are the flags of `nanollm service`, shared by the verbs.
type serviceOptions struct {
	name    string
	config  string
	dataDir string
	user    string
	logFile string
	unitDir string
	system  bool
	manual  bool
	noStart bool
}

func parseServiceFlags(verb string, args []string) (serviceOptions, error) {
	var o serviceOptions
	fs := flag.NewFlagSet("service "+verb, flag.ExitOnError)
	fs.StringVar(&o.name, "name", "", "unit name (systemd, default "+serviceName+") or label (launchd, default "+serviceLabel+")")
	fs.StringVar(&o.config, "config", "", "install: nanollm.yaml (default: the one in the data directory, or beside the binary)")
	fs.StringVar(&o.dataDir, "data-dir", "", "install: the directory the service may write (default: server.data_dir)")
	fs.StringVar(&o.user, "user", "", "install: account to run as (systemd default "+serviceName+"; launchd -system default root)")
	fs.StringVar(&o.logFile, "log-file", "", "install, launchd: the rolling log (default ~/Library/Logs/NanoLLM/nanollm.log)")
	fs.StringVar(&o.unitDir, "unit-dir", "", "systemd: where the unit goes (default /etc/systemd/system)")
	fs.BoolVar(&o.system, "system", false, "launchd: a LaunchDaemon for the whole machine instead of a LaunchAgent")
	fs.BoolVar(&o.manual, "manual", false, "install: do not start the service at boot")
	fs.BoolVar(&o.noStart, "no-start", false, "install: register the service but leave it stopped")
	if _, err := parseFlags(fs, args); err != nil {
		return o, err
	}
	return o, nil
}

// serviceCommand registers, removes and controls the service.
//
//	nanollm service install [-config FILE] [-user NAME] [-data-dir DIR] [-manual] [-no-start]
//	nanollm service uninstall
//	nanollm service start | stop | restart | status
//	nanollm service unit [install flags]     print the unit install would write
//
// The same verbs are accepted as `nanollm --install` and so on.
func serviceCommand(args []string) error {
	verb, args := splitVerb(args)
	switch verb {
	case "install", "uninstall", "start", "stop", "restart", "status", "unit":
	default:
		return fmt.Errorf("service %s: unknown subcommand; use %s", verb, listVerbs())
	}
	o, err := parseServiceFlags(verb, args)
	if err != nil {
		return err
	}
	return platformService(verb, o)
}

// run executes a service manager command, passing its output through.
func run(name string, args ...string) error {
	ctx, cancel := context.WithTimeout(context.Background(), managerTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s %s: %w", name, strings.Join(args, " "), err)
	}
	return nil
}

// output executes a command and returns what it printed, stderr included, so
// that a failure can say why.
func output(name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), managerTimeout)
	defer cancel()
	var b bytes.Buffer
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdout = &b
	cmd.Stderr = &b
	err := cmd.Run()
	return b.String(), err
}

func requireRoot(what string) error {
	if os.Geteuid() != 0 {
		return fmt.Errorf("%s needs root; run it with sudo", what)
	}
	return nil
}

// resolveInstall works out the binary, the configuration and the data
// directory an install registers.
func resolveInstall(o serviceOptions) (unitParams, error) {
	exe, err := exePath()
	if err != nil {
		return unitParams{}, err
	}
	cfgPath, cfg, err := resolveServiceConfig(o.config, filepath.Dir(exe))
	if err != nil {
		return unitParams{}, err
	}
	dataDir, err := absDataDir(o.dataDir, cfg)
	if err != nil {
		return unitParams{}, err
	}
	return unitParams{Exe: exe, Config: cfgPath, DataDir: dataDir, User: o.user}, nil
}

// resolveUnitPreview is resolveInstall without the checks, for `service
// unit`: printing a unit should not need the configuration to exist yet.
func resolveUnitPreview(o serviceOptions) (unitParams, error) {
	if p, err := resolveInstall(o); err == nil {
		return p, nil
	}
	exe, err := exePath()
	if err != nil {
		return unitParams{}, err
	}
	cfgPath := o.config
	if cfgPath == "" {
		cfgPath = defaultServiceConfig(filepath.Dir(exe))
	}
	if cfgPath, err = filepath.Abs(cfgPath); err != nil {
		return unitParams{}, err
	}
	dataDir := o.dataDir
	if dataDir == "" {
		dataDir = filepath.Dir(cfgPath)
	}
	if dataDir, err = filepath.Abs(dataDir); err != nil {
		return unitParams{}, err
	}
	return unitParams{Exe: exe, Config: cfgPath, DataDir: dataDir, User: o.user}, nil
}

// absDataDir is the data directory a configuration names, made absolute: a
// unit's ReadWritePaths and WorkingDirectory take only absolute paths.
func absDataDir(explicit string, cfg config.Config) (string, error) {
	dir := explicit
	if dir == "" {
		dir = cfg.Server.DataDir
	}
	if dir == "" {
		dir = config.DefaultDataDir()
	}
	return filepath.Abs(dir)
}

// writeFileAtomic replaces path with data in one rename, so that a service
// manager reading the unit never sees half of it.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".nanollm-unit-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Chmod(perm); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}
