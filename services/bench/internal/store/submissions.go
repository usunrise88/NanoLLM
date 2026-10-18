package store

import (
	"bufio"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"
)

// Submission is a raw submission row.
type Submission struct {
	ID         string
	ReceivedOn string
	InstallID  string
	RawSHA256  string
	Raw        []byte
}

// Moderation states.
const (
	StatePublished = "published"
	StateQueued    = "queued"
	StateHidden    = "hidden"
)

// Moderation is the state of one submission.
type Moderation struct {
	State     string
	Reason    string
	DecidedBy string // auto | admin
	UpdatedOn string
}

// Today is the UTC day, the only time the service records.
func Today() string { return time.Now().UTC().Format("2006-01-02") }

// InsertSubmission stores a raw submission and its initial moderation state.
func InsertSubmission(ctx context.Context, q Querier, sub Submission, mod Moderation) error {
	if _, err := q.ExecContext(ctx, `INSERT INTO submissions (id, received_on, install_id, raw_sha256, raw) VALUES (?, ?, ?, ?, ?)`,
		sub.ID, sub.ReceivedOn, sub.InstallID, sub.RawSHA256, sub.Raw); err != nil {
		return err
	}
	return SetModeration(ctx, q, sub.ID, mod)
}

// SetModeration records a moderation decision.
func SetModeration(ctx context.Context, q Querier, id string, mod Moderation) error {
	if mod.UpdatedOn == "" {
		mod.UpdatedOn = Today()
	}
	if mod.DecidedBy == "" {
		mod.DecidedBy = "auto"
	}
	res, err := q.ExecContext(ctx, `INSERT INTO moderation (submission_id, state, reason, decided_by, updated_on) VALUES (?, ?, ?, ?, ?)
		ON CONFLICT (submission_id) DO UPDATE SET state = excluded.state, reason = excluded.reason, decided_by = excluded.decided_by, updated_on = excluded.updated_on`,
		id, mod.State, mod.Reason, mod.DecidedBy, mod.UpdatedOn)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	return nil
}

// SubmissionByRawSHA finds an identical earlier submission.
func SubmissionByRawSHA(ctx context.Context, q Querier, sha string) (string, error) {
	var id string
	err := q.QueryRowContext(ctx, `SELECT id FROM submissions WHERE raw_sha256 = ?`, sha).Scan(&id)
	if errors.Is(err, sql.ErrNoRows) {
		return "", ErrNotFound
	}
	return id, err
}

// GetSubmission reads one raw submission with its moderation state.
func GetSubmission(ctx context.Context, q Querier, id string) (Submission, Moderation, error) {
	var s Submission
	var m Moderation
	err := q.QueryRowContext(ctx, `SELECT s.id, s.received_on, s.install_id, s.raw_sha256, s.raw, m.state, m.reason, m.decided_by, m.updated_on
		FROM submissions s JOIN moderation m ON m.submission_id = s.id WHERE s.id = ?`, id).
		Scan(&s.ID, &s.ReceivedOn, &s.InstallID, &s.RawSHA256, &s.Raw, &m.State, &m.Reason, &m.DecidedBy, &m.UpdatedOn)
	if errors.Is(err, sql.ErrNoRows) {
		return s, m, ErrNotFound
	}
	return s, m, err
}

// EachSubmission calls f for every raw submission, oldest first.
func EachSubmission(ctx context.Context, q Querier, f func(Submission, Moderation) error) error {
	rows, err := q.QueryContext(ctx, `SELECT s.id, s.received_on, s.install_id, s.raw_sha256, s.raw, m.state, m.reason, m.decided_by, m.updated_on
		FROM submissions s JOIN moderation m ON m.submission_id = s.id ORDER BY s.received_on, s.rowid`)
	if err != nil {
		return err
	}
	var all []struct {
		s Submission
		m Moderation
	}
	for rows.Next() {
		var s Submission
		var m Moderation
		if err := rows.Scan(&s.ID, &s.ReceivedOn, &s.InstallID, &s.RawSHA256, &s.Raw, &m.State, &m.Reason, &m.DecidedBy, &m.UpdatedOn); err != nil {
			_ = rows.Close()
			return err
		}
		all = append(all, struct {
			s Submission
			m Moderation
		}{s, m})
	}
	_ = rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	for _, x := range all {
		if err := f(x.s, x.m); err != nil {
			return err
		}
	}
	return nil
}

// CountSince counts an install's submissions on and after a day.
func CountSince(ctx context.Context, q Querier, installID, day string) (int, error) {
	var n int
	err := q.QueryRowContext(ctx, `SELECT COUNT(*) FROM submissions WHERE install_id = ? AND received_on >= ?`, installID, day).Scan(&n)
	return n, err
}

// Withdrawn reports whether an id was withdrawn.
func Withdrawn(ctx context.Context, q Querier, id string) (bool, error) {
	var n int
	err := q.QueryRowContext(ctx, `SELECT COUNT(*) FROM withdrawals WHERE id = ?`, id).Scan(&n)
	return n > 0, err
}

// Withdraw deletes a submission and everything derived from it, and leaves
// a tombstone: the id and the day, in the database and in the tombstone
// file.
func (s *Store) Withdraw(ctx context.Context, tx *sql.Tx, id string) error {
	res, err := tx.ExecContext(ctx, `DELETE FROM submissions WHERE id = ?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return ErrNotFound
	}
	if _, err := tx.ExecContext(ctx, `INSERT OR IGNORE INTO withdrawals (id, withdrawn_on) VALUES (?, ?)`, id, Today()); err != nil {
		return err
	}
	return s.appendTombstone(id)
}

func (s *Store) appendTombstone(id string) error {
	if s.Tombstones == "" {
		return nil
	}
	f, err := os.OpenFile(s.Tombstones, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o640)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(f, "%s %s\n", id, Today())
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// applyTombstones deletes any submission the tombstone file names: after a
// restore from a backup taken before a withdrawal.
func (s *Store) applyTombstones(ctx context.Context) error {
	f, err := os.Open(s.Tombstones)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return s.Tx(ctx, func(tx *sql.Tx) error {
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			id, day, _ := strings.Cut(strings.TrimSpace(sc.Text()), " ")
			if id == "" {
				continue
			}
			if _, err := tx.ExecContext(ctx, `DELETE FROM submissions WHERE id = ?`, id); err != nil {
				return err
			}
			if day == "" {
				day = Today()
			}
			if _, err := tx.ExecContext(ctx, `INSERT OR IGNORE INTO withdrawals (id, withdrawn_on) VALUES (?, ?)`, id, day); err != nil {
				return err
			}
		}
		return sc.Err()
	})
}

// QueueItem is one submission awaiting a decision, for the admin.
type QueueItem struct {
	ID         string `json:"id"`
	ReceivedOn string `json:"received_on"`
	State      string `json:"state"`
	Reason     string `json:"reason"`
	DecidedBy  string `json:"decided_by"`
	Model      string `json:"model"`
	System     string `json:"system"`
	CPU        string `json:"cpu"`
	GPU        string `json:"gpu"`
}

// ListModeration lists submissions in a state, newest first.
func ListModeration(ctx context.Context, q Querier, state string, limit int) ([]QueueItem, error) {
	rows, err := q.QueryContext(ctx, `SELECT s.id, s.received_on, m.state, m.reason, m.decided_by,
		COALESCE(n.model_id, ''), COALESCE(n.system, ''), COALESCE(n.cpu, ''), COALESCE(n.gpu, '')
		FROM submissions s JOIN moderation m ON m.submission_id = s.id LEFT JOIN norm_submissions n ON n.id = s.id
		WHERE m.state = ? ORDER BY s.received_on DESC, s.rowid DESC LIMIT ?`, state, limit)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	out := []QueueItem{}
	for rows.Next() {
		var it QueueItem
		if err := rows.Scan(&it.ID, &it.ReceivedOn, &it.State, &it.Reason, &it.DecidedBy, &it.Model, &it.System, &it.CPU, &it.GPU); err != nil {
			return nil, err
		}
		out = append(out, it)
	}
	return out, rows.Err()
}

// Alias is one row of the alias table.
type Alias struct {
	Kind    string `json:"kind"`
	Pattern string `json:"pattern"`
	Name    string `json:"name"`
}

// ListAliases reads the alias table.
func ListAliases(ctx context.Context, q Querier) ([]Alias, error) {
	rows, err := q.QueryContext(ctx, `SELECT kind, pattern, name FROM aliases ORDER BY kind, pattern`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	out := []Alias{}
	for rows.Next() {
		var a Alias
		if err := rows.Scan(&a.Kind, &a.Pattern, &a.Name); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

// SetAlias adds or replaces an alias. Patterns are matched case-blind.
func SetAlias(ctx context.Context, q Querier, a Alias) error {
	_, err := q.ExecContext(ctx, `INSERT INTO aliases (kind, pattern, name) VALUES (?, ?, ?)
		ON CONFLICT (kind, pattern) DO UPDATE SET name = excluded.name`, a.Kind, strings.ToLower(strings.TrimSpace(a.Pattern)), strings.TrimSpace(a.Name))
	return err
}

// DeleteAlias removes an alias.
func DeleteAlias(ctx context.Context, q Querier, kind, pattern string) error {
	_, err := q.ExecContext(ctx, `DELETE FROM aliases WHERE kind = ? AND pattern = ?`, kind, strings.ToLower(strings.TrimSpace(pattern)))
	return err
}

// Verified is a vendor-verified install.
type Verified struct {
	InstallID string `json:"install_id"`
	Label     string `json:"label"`
}

// ListVerified reads the vendor-verified installs.
func ListVerified(ctx context.Context, q Querier) ([]Verified, error) {
	rows, err := q.QueryContext(ctx, `SELECT install_id, label FROM verified_installs ORDER BY label, install_id`)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	out := []Verified{}
	for rows.Next() {
		var v Verified
		if err := rows.Scan(&v.InstallID, &v.Label); err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, rows.Err()
}

// SetVerified marks an install vendor-verified.
func SetVerified(ctx context.Context, q Querier, v Verified) error {
	_, err := q.ExecContext(ctx, `INSERT INTO verified_installs (install_id, label) VALUES (?, ?)
		ON CONFLICT (install_id) DO UPDATE SET label = excluded.label`, strings.ToLower(strings.TrimSpace(v.InstallID)), v.Label)
	return err
}

// DeleteVerified removes the badge.
func DeleteVerified(ctx context.Context, q Querier, installID string) error {
	_, err := q.ExecContext(ctx, `DELETE FROM verified_installs WHERE install_id = ?`, strings.ToLower(strings.TrimSpace(installID)))
	return err
}
