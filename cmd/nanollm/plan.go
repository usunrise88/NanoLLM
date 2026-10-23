package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

// planCommand prints the placement of a model under a preset without loading
// it (SPEC §4, §9). Devices are not probed yet, so the device is the
// preset's synthetic one unless --device and --device-memory describe the
// real one.
func planCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("plan", flag.ContinueOnError)
	cfgPath := configFlag(fs)
	presetName := fs.String("preset", "", "memory budget, one of plan.Presets; without it the model is planned the way the server places it, from the configuration, and each setting says where it comes from")
	device := fs.String("device", "", "accelerator selector, e.g. llamacpp:vulkan0 (default: the preset's synthetic device)")
	deviceMem := fs.String("device-memory", "", "the device's memory, e.g. 6g or 5800m (default: the preset's budget)")
	mode := fs.String("mode", "auto", "force a mode: M1…M9 or its name")
	kv := fs.String("kv", "f16", "KV cache type: f16 or q8_0")
	nCtx := fs.Int("n-ctx", 0, "fix the context per slot instead of growing it to the preset's target")
	slots := fs.Int("slots", 0, "fix the slot count")
	noMTP := fs.Bool("no-mtp", false, "plan without the drafter")
	noFA := fs.Bool("no-fa", false, "plan for a device without flash attention")
	drafter := fs.String("drafter", "", "drafter GGUF, when MODEL is a file")
	asJSON := fs.Bool("json", false, "print the plan as JSON")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if len(pos) > 1 || (len(pos) == 0 && *presetName != "") {
		return errors.New("plan: give one model, a catalog id or a GGUF path")
	}
	if *presetName == "" {
		model := ""
		if len(pos) == 1 {
			model = pos[0]
		}
		return planFromConfig(stdout, *cfgPath, model, *device, *asJSON)
	}
	preset, err := plan.LookupPreset(*presetName)
	if err != nil {
		return err
	}
	m, err := plan.ParseMode(*mode)
	if err != nil {
		return err
	}
	model, err := planModel(pos[0], *drafter)
	if err != nil {
		return err
	}

	req := plan.Request{
		Model: model, Preset: preset, Mode: m, KVType: *kv,
		NCtx: *nCtx, Slots: *slots, NoMTP: *noMTP, NoFlashAttention: *noFA,
	}
	if *device != "" || *deviceMem != "" {
		dev := plan.SyntheticDevice(preset)
		if *device != "" {
			d, err := plan.ParseDevice(*device)
			if err != nil {
				return err
			}
			dev = &d
		}
		if dev == nil {
			return fmt.Errorf("plan: %s is a RAM preset; it plans for the host without a device", preset.Name)
		}
		if *deviceMem != "" {
			if dev.Memory, err = parseSize(*deviceMem); err != nil {
				return err
			}
		}
		req.Device = dev
	}

	p, err := plan.Make(req)
	if err != nil {
		return err
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(p); err != nil {
			return err
		}
	} else {
		printPlan(stdout, p)
	}
	if !p.OK() {
		return fmt.Errorf("plan: %s", p.Refused)
	}
	return nil
}

// planModel resolves a catalog id to its built-in inventory, or reads a
// GGUF's header directly. A file is not hashed: the plan is a question about
// its header, not a statement about pinned bytes.
func planModel(arg, drafterPath string) (plan.Model, error) {
	if _, err := os.Stat(arg); err == nil {
		files := []gguf.Inventory{}
		for i, p := range []string{arg, drafterPath} {
			if p == "" {
				continue
			}
			f, err := gguf.Open(p)
			if err != nil {
				return plan.Model{}, err
			}
			inv, unknown := gguf.BuildInventory(f)
			if unknown != nil {
				return plan.Model{}, fmt.Errorf("%s: %w", p, unknown)
			}
			inv.Role = registry.RoleModel
			if i > 0 {
				inv.Role = registry.RoleDrafter
			}
			files = append(files, inv)
		}
		return plan.ModelFrom(arg, files)
	}
	cat, err := registry.Builtin()
	if err != nil {
		return plan.Model{}, err
	}
	m, ok := registry.Find(cat, arg)
	if !ok {
		return plan.Model{}, fmt.Errorf("plan: %q is neither a file nor a catalog id", arg)
	}
	return plan.ModelFrom(m.ID, m.Inventory)
}

// parseSize reads "6g", "1.5g", "512m" (binary units, as presets are) or a
// byte count.
func parseSize(s string) (int64, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	mult := float64(1)
	switch {
	case strings.HasSuffix(s, "g"):
		mult, s = 1<<30, strings.TrimSuffix(s, "g")
	case strings.HasSuffix(s, "m"):
		mult, s = 1<<20, strings.TrimSuffix(s, "m")
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("size %q: want a positive number with an optional g or m suffix", s)
	}
	return int64(v * mult), nil
}

func printPlan(w io.Writer, p *plan.Plan) {
	fmt.Fprintf(w, "%s under %s", p.Model, p.Preset)
	if p.Device != nil {
		fmt.Fprintf(w, " on %s (%s)", p.Device.Selector, p.Device.Backend)
	}
	fmt.Fprintln(w)
	if !p.OK() {
		fmt.Fprintf(w, "\nrefused: %s\n", p.Refused)
	} else {
		fmt.Fprintf(w, "\nmode         %s %s", p.ModeID, p.Mode)
		if p.Mode == plan.VRAMPartial {
			fmt.Fprintf(w, ", the last %d layers on the device", p.LayersOnGPU)
		}
		fmt.Fprintf(w, "\ncontext      %d tokens × %d slots (target %d × %d)\n", p.NCtx, p.Slots, p.Target.NCtx, p.Target.Slots)
		fmt.Fprintf(w, "load         %s, lazy %s, repack %v, gpu_layers %d\n", p.LoadMode, p.LazyMode, p.Repack, p.GPULayers)
		fmt.Fprintf(w, "batch        n_batch %d, n_ubatch %d, n_outputs_max %d, flash attention %v\n",
			p.NBatch, p.NUBatch, p.NOutputsMax, p.FlashAttention)
		fmt.Fprintf(w, "KV           %s/%s, %d full + %d SWA cells × %d streams, %s, swa_full %v\n",
			p.KV.TypeK, p.KV.TypeV, p.KV.CellsFull, p.KV.CellsSWA, p.KV.Streams, mib(p.KV.Bytes), p.KV.SWAFull)
		fmt.Fprintf(w, "             %s\n", p.KV.SWAFullReason)
		mtp := "off"
		if p.MTP.Enabled {
			mtp = fmt.Sprintf("on, up to %d drafts", p.MTP.DraftMax)
		}
		fmt.Fprintf(w, "MTP          %s: %s\n", mtp, p.MTP.Reason)
		for _, o := range p.Overrides {
			fmt.Fprintf(w, "override     %s → %s\n", o.Pattern, o.Device)
		}
		if len(p.LockRanges) > 0 {
			var n int64
			for _, r := range p.LockRanges {
				n += r.Length
			}
			fmt.Fprintf(w, "range-lock   %d ranges, %s\n", len(p.LockRanges), mib(n))
		}

		fmt.Fprintf(w, "\n  %-5s %12s %12s %12s\n", "group", "vram", "ram_locked", "file_backed")
		for _, g := range gguf.Groups {
			b, ok := p.Groups[g]
			if !ok || b.Total() == 0 {
				continue
			}
			fmt.Fprintf(w, "  %-5s %12s %12s %12s\n", g, mibOrDash(b.VRAM), mibOrDash(b.RAMLocked), mibOrDash(b.FileBacked))
		}
		usage := func(name string, u *plan.Usage) {
			if u == nil {
				return
			}
			if !u.Budgeted {
				fmt.Fprintf(w, "  %-11s %s, not budgeted\n", name, mib(u.Used))
				return
			}
			fmt.Fprintf(w, "  %-11s %s of %s\n", name, mib(u.Used), mib(u.Budget))
		}
		fmt.Fprintln(w)
		usage("vram", p.Memory.VRAM)
		usage("ram", p.Memory.RAM)
		usage("unified", p.Memory.Unified)
		usage("limit", p.Memory.Combined)
		fmt.Fprintf(w, "  %-11s %s (reported, not budgeted)\n", "file-backed", mib(p.Memory.FileBacked))
	}

	fmt.Fprintln(w, "\ntried")
	for _, a := range p.Tried {
		switch {
		case a.Fits:
			fmt.Fprintf(w, "  %-4s fits\n", a.ID)
		case a.Skipped != "":
			fmt.Fprintf(w, "  %-4s skipped: %s\n", a.ID, a.Skipped)
		case a.Shortfall != nil:
			var parts []string
			for _, s := range []struct {
				name string
				n    int64
			}{{"vram", a.Shortfall.VRAM}, {"ram", a.Shortfall.RAM}, {"unified", a.Shortfall.Unified}, {"memory limit", a.Shortfall.Combined}} {
				if s.n > 0 {
					parts = append(parts, s.name+" "+mib(s.n))
				}
			}
			fmt.Fprintf(w, "  %-4s short by %s\n", a.ID, strings.Join(parts, ", "))
		}
	}
	for _, msg := range p.Warnings {
		fmt.Fprintln(w, "\nnote:", msg)
	}
}

func mibOrDash(b int64) string {
	if b == 0 {
		return "-"
	}
	return mib(b)
}
