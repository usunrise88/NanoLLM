package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

// relayoutCommand writes the nano-layout copy of a catalog model's file
// (OPT-7, docs/AUTOTUNE.md): the same tensors, byte for byte, in the order the
// graph reads them, the pinned set and each layer's weights in one range. The
// copy lives beside the original, <file>.nano.gguf with <file>.nano.json;
// `models.overrides.<id>.layout: nano` and `bench --set layout=nano` use it.
func relayoutCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("models relayout", flag.ContinueOnError)
	cfgPath := configFlag(fs)
	align := fs.String("align", "4k", "general.alignment of the copy: a power of two from 32 bytes, e.g. 4k, 64k, 2m (every tensor is padded to it)")
	force := fs.Bool("force", false, "replace a copy that is already there")
	verify := fs.Bool("verify", false, "re-read both files and compare every tensor; alone, on a copy that exists, it only verifies")
	dry := fs.Bool("dry-run", false, "print the layout and the ranges the planner makes from it, write nothing")
	pos, err := parseFlags(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("models relayout: give one catalog id")
	}
	a, err := parseAlignment(*align)
	if err != nil {
		return fmt.Errorf("models relayout: --align: %w", err)
	}
	store, cat, _, err := openStore(*cfgPath)
	if err != nil {
		return err
	}
	m, ok := registry.Find(cat, pos[0])
	if !ok {
		return fmt.Errorf("models relayout: no %q in the catalog", pos[0])
	}
	src, _, err := store.Ready(m)
	if err != nil {
		return err
	}
	dst, err := store.NanoPath(m)
	if err != nil {
		return err
	}
	f, _ := m.File(registry.RoleModel)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	return relayoutFile(ctx, stdout, relayoutJob{
		ID: m.ID, Src: src, Dst: dst, SrcSHA256: f.SHA256, Align: a, Force: *force, Verify: *verify, Dry: *dry,
	})
}

type relayoutJob struct {
	ID  string
	Src string
	Dst string
	// SrcSHA256 is the catalog's pin of Src, which the store has verified.
	SrcSHA256 string
	Align     int64
	Force     bool
	Verify    bool
	Dry       bool
}

func relayoutFile(ctx context.Context, w io.Writer, j relayoutJob) error {
	f, err := gguf.Open(j.Src)
	if err != nil {
		return err
	}
	inv, err := gguf.BuildInventory(f)
	if err != nil {
		return fmt.Errorf("models relayout: %w", err)
	}
	l, err := gguf.PlanFile(f, j.Align)
	if err != nil {
		return err
	}
	printLayout(w, j, f, l)
	printRanges(w, inv, l.Inventory(inv))
	if j.Dry {
		return nil
	}

	_, statErr := os.Stat(j.Dst)
	exists := statErr == nil
	if exists && !j.Force {
		if !j.Verify {
			return fmt.Errorf("models relayout: %s exists; --force replaces it, --verify checks it", j.Dst)
		}
		// A copy that is there is only checked.
		rep, err := gguf.VerifyRelayout(ctx, j.Src, j.Dst, relayoutProgress("verify"))
		fmt.Fprintln(os.Stderr)
		if err != nil {
			return fmt.Errorf("models relayout: %s does not match %s: %w", j.Dst, j.Src, err)
		}
		if err := gguf.Restamp(j.Dst); err != nil {
			return err
		}
		printVerified(w, rep)
		return nil
	}

	sc, err := gguf.Relayout(ctx, j.Src, j.Dst, gguf.RelayoutOptions{
		Alignment: j.Align, SourceSHA256: j.SrcSHA256, Tool: "nanollm " + version, Force: j.Force,
		Progress: relayoutProgress("relayout"),
	})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return err
	}
	fmt.Fprintf(w, "\nwrote %s\n  %d bytes, sha256 %s\n  sidecar %s\n", j.Dst, sc.Result.Size, sc.Result.SHA256, gguf.SidecarPath(j.Dst))
	if j.Verify {
		rep, err := gguf.VerifyRelayout(ctx, j.Src, j.Dst, relayoutProgress("verify"))
		fmt.Fprintln(os.Stderr)
		if err != nil {
			return fmt.Errorf("models relayout: the copy does not match the original: %w", err)
		}
		printVerified(w, rep)
	}
	return nil
}

func printVerified(w io.Writer, rep gguf.VerifyReport) {
	fmt.Fprintf(w, "verified: %d tensors, %s of tensor data identical to the original, file %s, sha256 %s\n",
		rep.Tensors, fmtBytes(rep.Bytes), fmtBytes(rep.FileSize), rep.SHA256)
}

// relayoutProgress prints a percentage to stderr at the pace the library
// reports it.
func relayoutProgress(what string) func(done, total int64) {
	return func(done, total int64) {
		fmt.Fprintf(os.Stderr, "\r%s: %s / %s (%.0f%%)   ", what, fmtBytes(done), fmtBytes(total), 100*float64(done)/float64(total))
	}
}

func printLayout(w io.Writer, j relayoutJob, f *gguf.File, l *gguf.Layout) {
	fmt.Fprintf(w, "%s: layout nano version %d, alignment %d\n", j.ID, l.Version, l.Alignment)
	fmt.Fprintf(w, "  original %s, %d bytes, alignment %d\n", j.Src, f.FileSize, f.Alignment)
	fmt.Fprintf(w, "  nano     %s, %d bytes (%+.1f MiB), %d tensors\n", j.Dst, l.FileSize,
		float64(l.FileSize-f.FileSize)/(1<<20), len(l.Tensors))
	fmt.Fprintf(w, "\n  %-3s %7s %14s %14s  %s\n", "", "tensors", "offset", "length", "")
	for _, e := range l.Sections {
		fmt.Fprintf(w, "  %-3s %7d %14d %14d  %s\n", e.Section, e.Tensors, e.Offset, e.Length, e.Section.Description())
	}
	if l.Alignment > 1<<16 {
		fmt.Fprintf(w, "\nnote: every tensor is padded to %d bytes, the file grows and a section spans the padding;\n"+
			"the planner does not lock across gaps over 64 KiB, so the pinned set is not one range\n", l.Alignment)
	}
}

// rangeCase is a mode and a preset that makes it, the way AUTOTUNE.md counts
// the ranges of E4B.
type rangeCase struct {
	label    string
	mode     plan.Mode
	preset   string
	noRepack bool
}

var rangeCases = []rangeCase{
	{"M7 ram-4g", plan.RAMPLEDisk, "ram-4g", false},
	{"M7 ram-4g, repack off", plan.RAMPLEDisk, "ram-4g", true},
	{"M8 ram-2g", plan.RAMHybrid, "ram-2g", false},
	{"M9 ram-1g", plan.MMap, "ram-1g", false},
}

// printRanges plans the original inventory and the relaid one under the same
// presets and prints how many lock and stream ranges each makes.
func printRanges(w io.Writer, before, after gguf.Inventory) {
	fmt.Fprintf(w, "\n  ranges the planner makes   %-30s %-30s\n", "original", "nano")
	count := func(inv gguf.Inventory, c rangeCase) string {
		preset, err := plan.LookupPreset(c.preset)
		if err != nil {
			return "-"
		}
		model, err := plan.ModelFrom("relayout", []gguf.Inventory{withRole(inv)})
		if err != nil {
			return "-"
		}
		p, err := plan.Make(plan.Request{Model: model, Preset: preset, Mode: c.mode, NoRepack: c.noRepack, NoMTP: true})
		if err != nil || !p.OK() {
			return "does not fit"
		}
		perLayer := map[int]int{}
		for _, r := range p.StreamRanges {
			perLayer[r.Layer]++
		}
		worst := 0
		for _, n := range perLayer {
			worst = max(worst, n)
		}
		s := fmt.Sprintf("lock %d", len(p.LockRanges))
		if len(p.StreamRanges) > 0 {
			s += fmt.Sprintf(", stream %d (%d/layer)", len(p.StreamRanges), worst)
		}
		return s
	}
	for _, c := range rangeCases {
		fmt.Fprintf(w, "  %-26s %-30s %-30s\n", c.label, count(before, c), count(after, c))
	}
}

func withRole(inv gguf.Inventory) gguf.Inventory {
	inv.Role = registry.RoleModel
	return inv
}

// parseAlignment reads "4096", "4k", "64k" or "2m".
func parseAlignment(s string) (int64, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	mult := int64(1)
	switch {
	case strings.HasSuffix(s, "k"):
		mult, s = 1<<10, strings.TrimSuffix(s, "k")
	case strings.HasSuffix(s, "m"):
		mult, s = 1<<20, strings.TrimSuffix(s, "m")
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%q is not a number of bytes, optionally with k or m", s)
	}
	n *= mult
	if err := gguf.CheckAlignment(n); err != nil {
		return 0, err
	}
	return n, nil
}
