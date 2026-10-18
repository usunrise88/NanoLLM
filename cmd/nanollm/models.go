package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/registry"
)

// modelsFlags parses the flags the models verbs share.
func modelsFlags(name string, args []string) (cfgPath, from string, pos []string, err error) {
	fs := flag.NewFlagSet("models "+name, flag.ExitOnError)
	c := configFlag(fs)
	f := fs.String("from", "", "pull: take the files from this directory instead of downloading (offline installs)")
	pos, err = parseFlags(fs, args)
	return *c, *f, pos, err
}

func openStore(cfgPath string) (*registry.Store, []registry.Manifest, config.Config, error) {
	cfg, err := config.Load(cfgPath)
	if err != nil {
		return nil, nil, cfg, err
	}
	cat, err := registry.Builtin()
	if err != nil {
		return nil, nil, cfg, err
	}
	return newStore(cfg), cat, cfg, nil
}

// modelsList shows every catalog entry and what is on disk for it.
func modelsList(args []string) error {
	cfgPath, _, _, err := modelsFlags("list", args)
	if err != nil {
		return err
	}
	store, cat, cfg, err := openStore(cfgPath)
	if err != nil {
		return err
	}
	fmt.Printf("%-20s %-9s %-10s %s\n", "ID", "STATE", "SIZE", "PATH")
	for _, m := range cat {
		st, err := store.Status(m)
		if err != nil {
			fmt.Printf("%-20s %-9s %-10s %v\n", m.ID, "-", "-", err)
			continue
		}
		var size int64
		for _, f := range m.Files {
			size += f.Size
		}
		mark := ""
		if m.ID == cfg.Models.Default {
			mark = " (default)"
		}
		fmt.Printf("%-20s %-9s %-10s %s%s\n", m.ID, st.State, fmtBytes(size), st.Dir, mark)
	}
	return nil
}

// modelsPull downloads (or adopts and verifies) the named entries, or
// models.default when none is named.
func modelsPull(args []string) error {
	cfgPath, from, ids, err := modelsFlags("pull", args)
	if err != nil {
		return err
	}
	store, cat, cfg, err := openStore(cfgPath)
	if err != nil {
		return err
	}
	if len(ids) == 0 {
		ids = []string{cfg.Models.Default}
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	for _, id := range ids {
		m, ok := registry.Find(cat, id)
		if !ok {
			return fmt.Errorf("models pull: no %q in the catalog", id)
		}
		if from != "" {
			n, err := store.Adopt(m, from)
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stderr, "%s: took %d file(s) from %s\n", id, n, from)
		}
		var last time.Time
		err := store.Pull(ctx, m, func(p registry.Progress) {
			if time.Since(last) < time.Second && p.Downloaded < p.Total {
				return
			}
			last = time.Now()
			fmt.Fprintf(os.Stderr, "\r%s %s: %s / %s (%.0f%%)   ", id, p.File, fmtBytes(p.Downloaded), fmtBytes(p.Total), p.Percent)
		})
		fmt.Fprintln(os.Stderr)
		if err != nil {
			return err
		}
		st, _ := store.Status(m)
		fmt.Printf("%s: %s in %s\n", id, st.State, st.Dir)
	}
	return nil
}

func modelsRemove(args []string) error {
	cfgPath, _, ids, err := modelsFlags("rm", args)
	if err != nil {
		return err
	}
	if len(ids) == 0 {
		return fmt.Errorf("usage: nanollm models rm ID")
	}
	store, _, _, err := openStore(cfgPath)
	if err != nil {
		return err
	}
	for _, id := range ids {
		if err := store.Remove(id); err != nil {
			return err
		}
		fmt.Printf("%s: removed\n", id)
	}
	return nil
}

func fmtBytes(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.2f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
	}
	return fmt.Sprintf("%d B", n)
}
