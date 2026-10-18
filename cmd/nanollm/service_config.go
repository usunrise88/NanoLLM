package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/usunrise88/nanollm/internal/config"
)

// resolveServiceConfig picks the configuration the service will be registered
// with and refuses one the server would not start with.
//
// Registering an unusable configuration is the most expensive mistake
// available here: the service installs, the manager starts it, the process
// exits, and the evidence is a log line nobody reads. The archive ships a
// nanollm.yaml with placeholder keys beside the binary, which is precisely
// the file a service should not run from, so the data directory wins.
func resolveServiceConfig(explicit, exeDir string) (string, config.Config, error) {
	path := explicit
	if path == "" {
		path = defaultServiceConfig(exeDir)
	}
	path, err := filepath.Abs(path)
	if err != nil {
		return "", config.Config{}, err
	}
	hint := fmt.Sprintf("\nwrite one — it issues the API keys:\n"+
		"  nanollm init -config %s -data-dir %s", path, filepath.Dir(path))
	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", config.Config{}, fmt.Errorf("there is no configuration at %s%s", path, hint)
		}
		return "", config.Config{}, err
	}
	cfg, err := config.Load(path)
	if err != nil {
		return "", config.Config{}, fmt.Errorf("the configuration at %s is not one the server would start with: %w%s",
			path, err, hint+" -force")
	}
	return path, cfg, nil
}

// defaultServiceConfig is where a configuration written for a service lives:
// nanollm.yaml in the platform's data directory (SPEC §10), which is what the
// installers use, or the one beside the binary when there is none.
func defaultServiceConfig(exeDir string) string {
	shared := filepath.Join(config.DefaultDataDir(), "nanollm.yaml")
	if _, err := os.Stat(shared); err == nil {
		return shared
	}
	return filepath.Join(exeDir, "nanollm.yaml")
}

// exePath is this binary, resolved: a service manager stores the command line
// verbatim, so a relative path or a symlink would be recorded as one.
func exePath() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		exe = resolved
	}
	return filepath.Abs(exe)
}
