package store

import (
	"context"
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func open(t *testing.T, path string) *Store {
	t.Helper()
	s, err := Open(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func insert(t *testing.T, s *Store, id string) {
	t.Helper()
	err := InsertSubmission(context.Background(), s.DB, Submission{ID: id, ReceivedOn: Today(), InstallID: "k", RawSHA256: "sha-" + id, Raw: []byte(`{"x":1}`)},
		Moderation{State: StatePublished})
	if err != nil {
		t.Fatal(err)
	}
}

func TestMigrateIsIdempotent(t *testing.T) {
	p := filepath.Join(t.TempDir(), "b.db")
	s := open(t, p)
	v, err := s.SchemaVersion(context.Background())
	if err != nil || v < 1 {
		t.Fatalf("version %d, %v", v, err)
	}
	s.Close()
	s2 := open(t, p)
	if v2, _ := s2.SchemaVersion(context.Background()); v2 != v {
		t.Fatalf("reopen changed the version: %d → %d", v, v2)
	}
}

func TestRawIsImmutable(t *testing.T) {
	s := open(t, filepath.Join(t.TempDir(), "b.db"))
	insert(t, s, "a")
	_, err := s.DB.Exec(`UPDATE submissions SET raw = ? WHERE id = 'a'`, []byte("{}"))
	if err == nil || !strings.Contains(err.Error(), "immutable") {
		t.Fatalf("an update of a raw submission must fail, got %v", err)
	}
	_, m, err := GetSubmission(context.Background(), s.DB, "a")
	if err != nil || m.State != StatePublished {
		t.Fatal(m, err)
	}
}

func TestWithdrawSurvivesRestore(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	p := filepath.Join(dir, "b.db")
	s := open(t, p)
	insert(t, s, "a")
	insert(t, s, "b")
	snap, _, err := s.Backup(ctx, filepath.Join(dir, "backups"), time.Hour, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Tx(ctx, func(tx *sql.Tx) error { return s.Withdraw(ctx, tx, "a") }); err != nil {
		t.Fatal(err)
	}
	if _, _, err := GetSubmission(ctx, s.DB, "a"); !errors.Is(err, ErrNotFound) {
		t.Fatal("withdrawn submission still there")
	}
	if w, _ := Withdrawn(ctx, s.DB, "a"); !w {
		t.Fatal("no tombstone")
	}
	s.Close()

	// Restore the snapshot taken before the withdrawal over the database.
	b, _ := os.ReadFile(snap)
	for _, suffix := range []string{"", "-wal", "-shm"} {
		os.Remove(p + suffix)
	}
	if err := os.WriteFile(p, b, 0o640); err != nil {
		t.Fatal(err)
	}
	s2 := open(t, p)
	if _, _, err := GetSubmission(ctx, s2.DB, "a"); !errors.Is(err, ErrNotFound) {
		t.Fatal("a restore brought a withdrawn submission back")
	}
	if _, _, err := GetSubmission(ctx, s2.DB, "b"); err != nil {
		t.Fatal(err)
	}
}

func TestBackupRetention(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s := open(t, filepath.Join(dir, "b.db"))
	bdir := filepath.Join(dir, "backups")
	now := time.Date(2026, 10, 18, 3, 0, 0, 0, time.UTC)
	for _, ago := range []time.Duration{40 * 24 * time.Hour, 20 * 24 * time.Hour} {
		if _, _, err := s.Backup(ctx, bdir, 1000*24*time.Hour, now.Add(-ago)); err != nil {
			t.Fatal(err)
		}
	}
	_, removed, err := s.Backup(ctx, bdir, 30*24*time.Hour, now)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 1 || !strings.Contains(removed[0], "20260908") {
		t.Fatalf("removed %v", removed)
	}
	left, _ := filepath.Glob(filepath.Join(bdir, "bench-*.db"))
	if len(left) != 2 {
		t.Fatalf("left %v", left)
	}
}

func TestAliasesAndVerified(t *testing.T) {
	ctx := context.Background()
	s := open(t, filepath.Join(t.TempDir(), "b.db"))
	if err := SetAlias(ctx, s.DB, Alias{Kind: "vendor", Pattern: " Foo Corp ", Name: "Foo"}); err != nil {
		t.Fatal(err)
	}
	if err := SetAlias(ctx, s.DB, Alias{Kind: "nonsense", Pattern: "x", Name: "y"}); err == nil {
		t.Fatal("unknown alias kind accepted")
	}
	as, _ := ListAliases(ctx, s.DB)
	if len(as) != 1 || as[0].Pattern != "foo corp" {
		t.Fatal(as)
	}
	_ = SetVerified(ctx, s.DB, Verified{InstallID: "AB", Label: "owner EPYC"})
	vs, _ := ListVerified(ctx, s.DB)
	if len(vs) != 1 || vs[0].InstallID != "ab" {
		t.Fatal(vs)
	}
}
