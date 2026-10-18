package store

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const backupLayout = "20060102T150405Z"

// Backup writes a consistent snapshot of the database into dir as
// bench-<UTC time>.db and deletes snapshots older than keep. A withdrawn
// submission is gone from the live database at once and from the backups
// once the last snapshot taken before the withdrawal ages out: retention is
// the promise, so keep must be bounded.
//
// The tombstone file is copied next to the snapshots as "withdrawals": it
// only grows, and a restore applies it (see Open), so a snapshot that still
// holds a withdrawn submission never brings it back.
func (s *Store) Backup(ctx context.Context, dir string, keep time.Duration, now time.Time) (string, []string, error) {
	if keep <= 0 {
		return "", nil, fmt.Errorf("backup retention must be positive")
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return "", nil, err
	}
	path := filepath.Join(dir, "bench-"+now.UTC().Format(backupLayout)+".db")
	if _, err := os.Stat(path); err == nil {
		return "", nil, fmt.Errorf("%s exists", path)
	}
	if _, err := s.DB.ExecContext(ctx, `VACUUM INTO ?`, path); err != nil {
		return "", nil, fmt.Errorf("backup: %w", err)
	}
	if b, err := os.ReadFile(s.Tombstones); err == nil {
		if err := os.WriteFile(filepath.Join(dir, "withdrawals"), b, 0o640); err != nil {
			return path, nil, err
		}
	}
	removed, err := PruneBackups(dir, keep, now)
	return path, removed, err
}

// PruneBackups deletes snapshots older than keep.
func PruneBackups(dir string, keep time.Duration, now time.Time) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var removed []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasPrefix(name, "bench-") || !strings.HasSuffix(name, ".db") {
			continue
		}
		t, err := time.Parse(backupLayout, strings.TrimSuffix(strings.TrimPrefix(name, "bench-"), ".db"))
		if err != nil {
			continue
		}
		if now.Sub(t) > keep {
			if err := os.Remove(filepath.Join(dir, name)); err != nil {
				return removed, err
			}
			removed = append(removed, name)
		}
	}
	sort.Strings(removed)
	return removed, nil
}
