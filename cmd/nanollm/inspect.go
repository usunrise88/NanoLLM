package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/registry"
)

// inspectCommand measures a GGUF header: metadata, bytes per placement group
// and the hyperparameters that size the KV cache (SPEC §4, docs/STAGE1.md
// 1.3). With --tensors --json it prints exactly the inventory file the catalog
// embeds, so tensors/<id>.json is regenerated with
//
//	nanollm models inspect MODEL.gguf --drafter DRAFTER.gguf --tensors --json
//
// A catalog id instead of a path inspects the downloaded files if they are in
// the data directory, and the built-in inventory otherwise.
func inspectCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("models inspect", flag.ContinueOnError)
	cfgPath := configFlag(fs)
	tensors := fs.Bool("tensors", false, "list every tensor")
	asJSON := fs.Bool("json", false, "print the inventory as JSON")
	drafter := fs.String("drafter", "", "the MTP drafter GGUF to inventory with the model")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("models inspect: give one GGUF path or catalog id")
	}
	cat, err := registry.Builtin()
	if err != nil {
		return err
	}

	inv, warnings, source, err := inventoryFor(pos[0], *drafter, *cfgPath, cat)
	if err != nil {
		return err
	}
	if *asJSON {
		if !*tensors {
			for i := range inv.Files {
				inv.Files[i].Tensors = nil
			}
		}
		for _, w := range warnings {
			fmt.Fprintln(os.Stderr, "warning:", w)
		}
		return registry.EncodeInventory(stdout, inv)
	}
	printInventory(stdout, inv, source, warnings, *tensors)
	return nil
}

// inventoryFor resolves the argument to an inventory. Warnings are unknown
// tensors and differences from the built-in inventory: worth showing, not
// worth refusing to inspect a file over.
func inventoryFor(arg, drafterPath, cfgPath string, cat []registry.Manifest) (registry.InventoryFile, []string, string, error) {
	inv := registry.InventoryFile{Schema: registry.InventorySchema}
	var warnings []string

	if _, err := os.Stat(arg); err == nil {
		paths := []string{arg}
		if drafterPath != "" {
			paths = append(paths, drafterPath)
		}
		var entry registry.Manifest
		for i, p := range paths {
			fi, unknown, err := registry.InspectFile(p, "")
			if err != nil {
				return inv, nil, "", err
			}
			if unknown != nil {
				warnings = append(warnings, strings.Split(unknown.Error(), "\n")...)
			}
			fi.Role = registry.RoleModel
			if i > 0 {
				fi.Role = registry.RoleDrafter
			}
			if m, role, ok := registry.MatchFile(cat, fi.SHA256); ok {
				entry, fi.Role = m, role
			}
			inv.Files = append(inv.Files, fi)
		}
		inv.ID = entry.ID
		if inv.ID == "" {
			inv.ID = strings.TrimSuffix(filepath.Base(arg), filepath.Ext(arg))
			return inv, warnings, "measured; not a catalog file", nil
		}
		warnings = append(warnings, compareBuiltin(entry, inv)...)
		return inv, warnings, "measured; catalog entry " + entry.ID, nil
	}

	m, ok := registry.Find(cat, arg)
	if !ok {
		return inv, nil, "", fmt.Errorf("models inspect: %q is neither a file nor a catalog id", arg)
	}
	inv.ID = m.ID
	if dir, err := modelDir(cfgPath, m); err == nil {
		for _, f := range m.Files {
			p := filepath.Join(dir, f.Revision, filepath.FromSlash(f.Path))
			if _, err := os.Stat(p); err != nil {
				inv.Files = nil
				break
			}
			fi, unknown, err := registry.InspectFile(p, f.Role)
			if err != nil {
				return inv, nil, "", err
			}
			if unknown != nil {
				warnings = append(warnings, strings.Split(unknown.Error(), "\n")...)
			}
			inv.Files = append(inv.Files, fi)
		}
		if len(inv.Files) > 0 {
			warnings = append(warnings, compareBuiltin(m, inv)...)
			return inv, warnings, "measured from " + dir, nil
		}
	}
	if len(m.Inventory) == 0 {
		return inv, nil, "", fmt.Errorf("models inspect: %s is not downloaded and has no built-in inventory", m.ID)
	}
	inv.Files = append(inv.Files, m.Inventory...)
	return inv, nil, "built-in inventory; files not downloaded", nil
}

// modelDir is <data_dir>/models/<id>, the registry layout of docs/STAGE1.md
// 1.2.
func modelDir(cfgPath string, m registry.Manifest) (string, error) {
	cfg, err := config.Load(cfgPath)
	if err != nil {
		return "", err
	}
	return filepath.Join(cfg.Server.DataDir, "models", m.ID), nil
}

// compareBuiltin reports where a measurement differs from what the binary
// embeds. For a pinned file it never should; if it does, the group table
// changed and tensors/<id>.json needs regenerating.
func compareBuiltin(m registry.Manifest, inv registry.InventoryFile) []string {
	var out []string
	for _, fi := range inv.Files {
		want, ok := m.FileInventory(fi.Role)
		if !ok {
			out = append(out, fmt.Sprintf("%s: no built-in inventory for the %s file", m.ID, fi.Role))
			continue
		}
		var a, b bytes.Buffer
		_ = registry.EncodeInventory(&a, registry.InventoryFile{ID: m.ID, Files: []gguf.Inventory{want}})
		_ = registry.EncodeInventory(&b, registry.InventoryFile{ID: m.ID, Files: []gguf.Inventory{fi}})
		if !bytes.Equal(a.Bytes(), b.Bytes()) {
			out = append(out, fmt.Sprintf("%s: the %s file differs from the built-in inventory; regenerate tensors/%s.json",
				m.ID, fi.Role, m.ID))
		}
	}
	return out
}

func printInventory(w io.Writer, inv registry.InventoryFile, source string, warnings []string, tensors bool) {
	fmt.Fprintf(w, "%s (%s)\n", inv.ID, source)
	for _, fi := range inv.Files {
		h := fi.HParams
		fmt.Fprintf(w, "\n%s file: %s, GGUF v%d, %s\n", fi.Role, h.Arch, fi.Version, mib(fi.Size))
		fmt.Fprintf(w, "  sha256       %s\n", fi.SHA256)
		fmt.Fprintf(w, "  data area    %d + %d bytes, alignment %d (padded tensor sizes match)\n",
			fi.DataOffset, fi.DataSize, fi.Alignment)

		total := fi.Total()
		row := func(name string, n int64) {
			fmt.Fprintf(w, "  %-12s %12d B %10.2f MiB %6.1f%%\n", name, n, float64(n)/(1<<20),
				100*float64(n)/float64(total))
		}
		for _, g := range gguf.Groups {
			if n, ok := fi.Groups[g]; ok {
				row(string(g), n)
			}
			if sub := fi.SumSub(gguf.G2, gguf.SubPLEProj); g == gguf.G2 && sub > 0 {
				row("  "+gguf.SubPLEProj, sub)
			}
		}
		row("total", total)

		swa, swaKV := 0, 0
		for il := range h.BlockCount {
			if h.IsSWA(il) {
				swa++
				if il < h.KVLayers() {
					swaKV++
				}
			}
		}
		fmt.Fprintf(w, "  layers       %d (%d sliding-window, %d full)\n", h.BlockCount, swa, h.BlockCount-swa)
		if h.KVLayers() == 0 {
			fmt.Fprintf(w, "  KV layers    none: every layer reads the target's cache\n")
		} else {
			fmt.Fprintf(w, "  KV layers    %d (%d sliding-window, %d full); the last %d reuse them\n",
				h.KVLayers(), swaKV, h.KVLayers()-swaKV, h.SharedKVLayers)
		}
		fmt.Fprintf(w, "  n_embd       %d, per-layer input %d, vocab %d, tied output %v\n",
			h.EmbeddingLength, h.EmbeddingLengthPerLayerInput, h.VocabSize, h.TiedOutput)
		if h.EmbeddingLengthOut > 0 {
			fmt.Fprintf(w, "  n_embd_out   %d (the target's hidden size)\n", h.EmbeddingLengthOut)
		}
		fmt.Fprintf(w, "  heads        %v q, %v kv\n", compact(h.HeadCount), compact(h.HeadCountKV))
		fmt.Fprintf(w, "  head dim     k %d v %d full, k %d v %d sliding-window, window %d\n",
			h.KeyLength, h.ValueLength, h.KeyLengthSWA, h.ValueLengthSWA, h.SlidingWindow)
		fmt.Fprintf(w, "  context      %d trained\n", h.ContextLength)

		if tensors {
			tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
			fmt.Fprintln(tw, "\n  NAME\tGROUP\tLAYER\tTYPE\tSHAPE\tBYTES\tOFFSET")
			for _, t := range fi.Tensors {
				g := string(t.Group)
				if t.Subgroup != "" {
					g += "/" + t.Subgroup
				}
				if g == "" {
					g = "?"
				}
				fmt.Fprintf(tw, "  %s\t%s\t%d\t%s\t%v\t%d\t%d\n", t.Name, g, t.Layer, t.Type, t.Shape, t.Bytes, t.Offset)
			}
			_ = tw.Flush()
		}
	}
	for _, msg := range warnings {
		fmt.Fprintln(w, "\nwarning:", msg)
	}
}

// compact prints a per-layer list as one value when every layer agrees.
func compact(xs []int) any {
	for _, x := range xs {
		if x != xs[0] {
			return xs
		}
	}
	if len(xs) == 0 {
		return 0
	}
	return xs[0]
}

func mib(b int64) string {
	return fmt.Sprintf("%.2f MiB", float64(b)/(1<<20))
}
