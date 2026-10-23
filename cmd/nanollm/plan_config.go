package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/llamacpp"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/placement"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/tune"
)

// planFromConfig plans a catalog model the way the server places it: the real
// devices, the model's block of the configuration, the autotuner's mode and
// profile. Every setting that the configuration, a profile or a mode had a say
// in is listed with its source (docs/OPTIMIZE.md, §3.7).
func planFromConfig(w io.Writer, cfgPath, modelID, deviceSel string, asJSON bool) error {
	cfg, err := config.Load(cfgPath)
	if err != nil {
		return err
	}
	cat, err := registry.Builtin()
	if err != nil {
		return err
	}
	if modelID == "" {
		modelID = cfg.Models.Default
	}
	m, ok := registry.Find(cat, modelID)
	if !ok {
		return fmt.Errorf("plan: no %q in the catalog", modelID)
	}
	libDir, err := resolveLibDir(cfg, "")
	if err != nil {
		return err
	}
	var devs []backend.Device
	for _, name := range cfg.Backends.Enabled {
		b, err := backend.Open(name, libDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "plan: backend %s unavailable: %v\n", name, err)
			continue
		}
		ds, err := b.Devices(context.Background())
		_ = b.Close()
		if err != nil {
			continue
		}
		devs = append(devs, ds...)
	}
	if len(devs) == 0 {
		return fmt.Errorf("plan: no device could be listed; see `nanollm devices`")
	}
	act, err := tune.LoadActive(context.Background(), tune.ActiveInput{
		Config: cfg, Manifest: m, Devices: devs, LibDir: libDir,
		LibsRelease: llamacpp.LibsRelease, LlamaCommit: llamacpp.PinnedCommit,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "plan: tune:", err)
	}
	if deviceSel == "" {
		deviceSel = act.DefaultDevice(cfg.Devices.Default)
	}
	if deviceSel == "" {
		deviceSel = cfg.Devices.Default
	}
	dev, err := pickPlanDevice(devs, deviceSel)
	if err != nil {
		return err
	}
	modelPath, drafterPath := "model.gguf", "drafter.gguf"
	if mp, dp, err := newStore(cfg).Ready(m); err == nil {
		modelPath, drafterPath = mp, dp
	}
	res, err := placement.Resolve(placement.Input{
		Manifest: m, Device: dev, Override: cfg.Models.Overrides[m.ID], ModelPath: modelPath, DrafterPath: drafterPath,
		Threads: cfg.Devices.CPUThreads, ThreadsBatch: cfg.Devices.CPUThreadsBatch, Tune: act.PlaceTune(dev.Selector()),
	})
	if err != nil {
		return err
	}
	if asJSON {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(res.Plan)
	}
	printPlan(w, res.Plan)
	if msg := act.Message(); msg != "" {
		fmt.Fprintln(w, "\ntune:", msg)
	}
	printSources(w, res.Plan.Sources, act)
	return nil
}

// pickPlanDevice resolves a selector the way the server does for a request with
// none: auto (the first GPU, else the CPU), a class, or backend:id.
func pickPlanDevice(devs []backend.Device, sel string) (backend.Device, error) {
	first := func(t backend.DeviceType) (backend.Device, bool) {
		for _, d := range devs {
			if d.Type == t {
				return d, true
			}
		}
		return backend.Device{}, false
	}
	switch sel {
	case "", config.DeviceAuto:
		if d, ok := first(backend.GPU); ok {
			return d, nil
		}
		if d, ok := first(backend.CPU); ok {
			return d, nil
		}
	case config.DeviceCPU, config.DeviceGPU, config.DeviceNPU:
		if d, ok := first(backend.DeviceType(sel)); ok {
			return d, nil
		}
	default:
		for _, d := range devs {
			if d.Selector() == sel {
				return d, nil
			}
		}
	}
	return backend.Device{}, fmt.Errorf("plan: no device matches %q", sel)
}

// printSources lists what the layers decided, one setting per line.
func printSources(w io.Writer, src map[string]string, act *tune.Active) {
	if len(src) == 0 {
		return
	}
	names := make([]string, 0, len(src))
	for k := range src {
		names = append(names, k)
	}
	sort.Strings(names)
	fmt.Fprintf(w, "\nsources (config beats profile, which beats mode, which beats rule)\n")
	for _, k := range names {
		fmt.Fprintf(w, "  %-16s %s\n", k, src[k])
	}
	if act != nil && act.Mode.Valid() {
		fmt.Fprintf(w, "  %-16s %s%s\n", "tune.mode", act.Mode, strings.TrimSuffix(" ("+profileNote(act)+")", " ()"))
	}
}

func profileNote(a *tune.Active) string {
	switch {
	case a.Profile != nil:
		return "profile " + a.Profile.ID()
	case a.Stale:
		return "profile stale"
	case a.Missing:
		return "no profile"
	}
	return ""
}
