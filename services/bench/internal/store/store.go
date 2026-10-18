// Package store is the service's SQLite database (modernc, pure Go, WAL):
// immutable raw submissions, the moderation and admin state about them, and
// the derived tables `benchd reindex` rebuilds.
package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	_ "modernc.org/sqlite" // the "sqlite" driver

	"github.com/usunrise88/nanollm/services/bench/migrations"
)

// Querier is what *sql.DB and *sql.Tx share.
type Querier interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

// Store is the database.
type Store struct {
	DB   *sql.DB
	Path string
	// Tombstones is an append-only file of withdrawn ids next to the
	// database. A restored backup older than a withdrawal is cleaned from it
	// on open, so a withdrawn submission never comes back.
	Tombstones string
}

// ErrNotFound is returned for a missing row.
var ErrNotFound = errors.New("not found")

// Open opens (creating if needed) the database at path and migrates it.
// secure_delete zeroes deleted content, so a withdrawn submission's bytes do
// not linger in free pages, and from there in backups.
func Open(ctx context.Context, path string) (*Store, error) {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o750); err != nil {
			return nil, err
		}
	}
	q := url.Values{}
	for _, p := range []string{"foreign_keys(1)", "journal_mode(WAL)", "busy_timeout(10000)", "secure_delete(1)", "synchronous(NORMAL)"} {
		q.Add("_pragma", p)
	}
	q.Set("_txlock", "immediate")
	db, err := sql.Open("sqlite", "file:"+path+"?"+q.Encode())
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(4)
	s := &Store{DB: db, Path: path, Tombstones: path + ".withdrawals"}
	if err := s.migrate(ctx); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := s.applyTombstones(ctx); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

// Close closes the database.
func (s *Store) Close() error { return s.DB.Close() }

// Tx runs f in a write transaction.
func (s *Store) Tx(ctx context.Context, f func(tx *sql.Tx) error) error {
	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	if err := f(tx); err != nil {
		_ = tx.Rollback()
		return err
	}
	return tx.Commit()
}

// SchemaVersion is the highest migration applied.
func (s *Store) SchemaVersion(ctx context.Context) (int, error) {
	var v sql.NullInt64
	err := s.DB.QueryRowContext(ctx, `SELECT MAX(version) FROM schema_version`).Scan(&v)
	return int(v.Int64), err
}

type migration struct {
	version int
	name    string
	sql     string
}

func loadMigrations(fsys fs.FS) ([]migration, error) {
	names, err := fs.Glob(fsys, "*.sql")
	if err != nil {
		return nil, err
	}
	var ms []migration
	seen := map[int]string{}
	for _, n := range names {
		num, _, ok := strings.Cut(n, "_")
		v, err := strconv.Atoi(num)
		if !ok || err != nil || v <= 0 {
			return nil, fmt.Errorf("migration %s: want NNN_name.sql", n)
		}
		if prev, dup := seen[v]; dup {
			return nil, fmt.Errorf("migrations %s and %s share version %d", prev, n, v)
		}
		seen[v] = n
		b, err := fs.ReadFile(fsys, n)
		if err != nil {
			return nil, err
		}
		ms = append(ms, migration{v, n, string(b)})
	}
	sort.Slice(ms, func(i, j int) bool { return ms[i].version < ms[j].version })
	return ms, nil
}

func (s *Store) migrate(ctx context.Context) error {
	if _, err := s.DB.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_on TEXT NOT NULL DEFAULT (date('now')))`); err != nil {
		return err
	}
	ms, err := loadMigrations(migrations.FS)
	if err != nil {
		return err
	}
	cur, err := s.SchemaVersion(ctx)
	if err != nil {
		return err
	}
	for _, m := range ms {
		if m.version <= cur {
			continue
		}
		err := s.Tx(ctx, func(tx *sql.Tx) error {
			if _, err := tx.ExecContext(ctx, m.sql); err != nil {
				return err
			}
			_, err := tx.ExecContext(ctx, `INSERT INTO schema_version (version) VALUES (?)`, m.version)
			return err
		})
		if err != nil {
			return fmt.Errorf("migration %s: %w", m.name, err)
		}
	}
	return nil
}
