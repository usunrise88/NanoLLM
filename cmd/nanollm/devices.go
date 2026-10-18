package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/usunrise88/nanollm/internal/api/adapter"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/llamacpp"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/ui"
)

func printVersion() {
	fmt.Printf("nanollm    %s\n", version)
	fmt.Printf("llama.cpp  %s %s (minimum %s)\n", llamacpp.PinnedTag, shortCommit(llamacpp.PinnedCommit), llamacpp.MinTag)
	fmt.Printf("backends   %v\n", backend.Registered())
	fmt.Printf("dialects   %v\n", adapter.Available())
	fmt.Printf("ui         %v\n", ui.Enabled)
}

// devicesCommand reports, per enabled backend, whether its libraries load,
// the build they are, the devices they see, and the backend libraries in
// lib/ that did not load and why (a missing libvulkan.so.1, a CUDA driver
// that is too old) — the questions fetch-libs and installers get wrong.
func devicesCommand(args []string) error {
	fs := flag.NewFlagSet("devices", flag.ExitOnError)
	cfgPath := configFlag(fs)
	libDirFlag := fs.String("lib-dir", "", "native library directory (overrides backends.lib_dir)")
	if _, err := parseFlags(fs, args); err != nil {
		return err
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}
	libDir, err := resolveLibDir(cfg, *libDirFlag)
	if err != nil {
		return err
	}
	fmt.Printf("lib_dir %s\n\n", libDir)

	for _, name := range cfg.Backends.Enabled {
		b, err := backend.Open(name, libDir)
		if err != nil {
			fmt.Printf("%-10s unavailable: %v\n", name, err)
			continue
		}
		devs, err := b.Devices(context.Background())
		switch {
		case errors.Is(err, backend.ErrNotImplemented):
			fmt.Printf("%-10s %s: libraries found; %v\n", name, b.Version(), err)
		case err != nil:
			fmt.Printf("%-10s %s: %v\n", name, b.Version(), err)
		case len(devs) == 0:
			fmt.Printf("%-10s %s: no devices\n", name, b.Version())
		default:
			fmt.Printf("%-10s %s\n", name, b.Version())
			if v, ok := b.(interface{ CPUVariant() string }); ok && v.CPUVariant() != "" {
				fmt.Printf("  cpu variant %s\n", v.CPUVariant())
			}
			for _, d := range devs {
				kind := string(d.Type)
				if d.Unified {
					kind += " (unified memory)"
				}
				fmt.Printf("  %-22s %-22s %-40s %s free of %s\n", d.Selector(), kind, d.Name,
					gib(d.MemoryFree), gib(d.MemoryTotal))
			}
		}
		if w, ok := b.(backend.Warner); ok {
			for _, msg := range w.Warnings() {
				fmt.Printf("  warning: %s\n", msg)
			}
		}
		_ = b.Close()
	}
	return nil
}

// resolveLibDir applies the -lib-dir flag over backends.lib_dir.
func resolveLibDir(cfg config.Config, flagValue string) (string, error) {
	dir := cfg.Backends.LibDir
	if flagValue != "" {
		abs, err := filepath.Abs(flagValue)
		if err != nil {
			return "", err
		}
		dir = abs
	}
	return backend.ResolveLibDir(dir)
}

func modelsCommand(args []string) error {
	sub := "list"
	if len(args) > 0 {
		sub = args[0]
	}
	switch sub {
	case "catalog":
		return showCatalog()
	case "inspect":
		return inspectCommand(args[1:], os.Stdout)
	case "list", "pull", "rm":
		return notYet("models "+sub, "v1.0")
	default:
		return fmt.Errorf("models %s: unknown subcommand; use catalog, list, pull, rm or inspect", sub)
	}
}

func showCatalog() error {
	cat, err := registry.Builtin()
	if err != nil {
		return err
	}
	fmt.Printf("%-18s %-8s %-6s %-11s %-9s %-18s %s\n",
		"ID", "ARCH", "PARAMS", "QUANT", "ROLE", "SIZE", "PINNED")
	for _, m := range cat {
		size := ""
		if f, ok := m.File(registry.RoleModel); ok {
			size = f.ApproxSize
		}
		if d, ok := m.File(registry.RoleDrafter); ok {
			size += " + " + d.ApproxSize
		}
		fmt.Printf("%-18s %-8s %-6s %-11s %-9s %-18s %v\n",
			m.ID, m.Arch, m.Params, m.Quant, m.Role, size, m.Pinned())
	}
	return nil
}

// shortCommit is the abbreviation llama.cpp itself reports (ggml_commit).
func shortCommit(c string) string {
	if len(c) > 9 {
		return c[:9]
	}
	return c
}

func gib(b int64) string {
	return fmt.Sprintf("%.1f GiB", float64(b)/(1<<30))
}
