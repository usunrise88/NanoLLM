package queue

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	_ "modernc.org/sqlite"

	"github.com/usunrise88/nanollm/internal/core"
)

// Job statuses.
const (
	StatusQueued    = "queued"
	StatusRunning   = "running"
	StatusSucceeded = "succeeded"
	StatusFailed    = "failed"
	StatusCanceled  = "canceled"
)

// Job is a batch as the API reports it.
type Job struct {
	ID         string        `json:"id"`
	Status     string        `json:"status"`
	Model      string        `json:"model,omitempty"`
	CreatedAt  time.Time     `json:"created_at"`
	StartedAt  *time.Time    `json:"started_at,omitempty"`
	FinishedAt *time.Time    `json:"finished_at,omitempty"`
	Progress   Progress      `json:"progress"`
	Error      *ErrorBody    `json:"error,omitempty"`
	Results    []ItemResult  `json:"results,omitempty"`
	WebhookURL string        `json:"webhook_url,omitempty"`
	Key        string        `json:"-"`
	items      []itemRequest // only on submit
}

type Progress struct {
	Done   int `json:"done"`
	Total  int `json:"total"`
	Failed int `json:"failed,omitempty"`
}

type ErrorBody struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ItemResult is one line's outcome.
type ItemResult struct {
	CustomID string           `json:"custom_id,omitempty"`
	Index    int              `json:"index"`
	Result   *core.ChatResult `json:"result,omitempty"`
	Error    *ErrorBody       `json:"error,omitempty"`
}

type itemRequest struct {
	customID string
	body     []byte
}

func (j *Job) terminal() bool {
	switch j.Status {
	case StatusSucceeded, StatusFailed, StatusCanceled:
		return true
	}
	return false
}

// store is the job database: metadata, the request of every line and its
// result. Pure Go SQLite, so the binary stays cgo-free.
type store struct{ db *sql.DB }

const schema = `
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  model TEXT NOT NULL DEFAULT '',
  key_name TEXT NOT NULL DEFAULT '',
  webhook_url TEXT NOT NULL DEFAULT '',
  created_at INTEGER NOT NULL,
  started_at INTEGER,
  finished_at INTEGER,
  total INTEGER NOT NULL,
  done INTEGER NOT NULL DEFAULT 0,
  failed INTEGER NOT NULL DEFAULT 0,
  error_code TEXT,
  error_message TEXT
);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
CREATE TABLE IF NOT EXISTS items (
  job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  custom_id TEXT NOT NULL DEFAULT '',
  request BLOB NOT NULL,
  result BLOB,
  error_code TEXT,
  error_message TEXT,
  PRIMARY KEY (job_id, idx)
);
`

func openStore(path string) (*store, error) {
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)&_pragma=foreign_keys(1)")
	if err != nil {
		return nil, err
	}
	// One writer: SQLite serialises them anyway, and a single connection
	// makes "database is locked" impossible rather than unlikely.
	db.SetMaxOpenConns(1)
	if _, err := db.ExecContext(context.Background(), schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("queue database %s: %w", path, err)
	}
	return &store{db: db}, nil
}

func (s *store) close() error { return s.db.Close() }

func ms(t time.Time) int64 { return t.UnixMilli() }

func fromMS(v sql.NullInt64) *time.Time {
	if !v.Valid {
		return nil
	}
	t := time.UnixMilli(v.Int64).UTC()
	return &t
}

func (s *store) insert(ctx context.Context, j *Job) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, `INSERT INTO jobs(id, status, model, key_name, webhook_url, created_at, total) VALUES (?,?,?,?,?,?,?)`,
		j.ID, j.Status, j.Model, j.Key, j.WebhookURL, ms(j.CreatedAt), j.Progress.Total); err != nil {
		return err
	}
	for i, it := range j.items {
		if _, err := tx.ExecContext(ctx, `INSERT INTO items(job_id, idx, custom_id, request) VALUES (?,?,?,?)`, j.ID, i, it.customID, it.body); err != nil {
			return err
		}
	}
	return tx.Commit()
}

const jobColumns = `id, status, model, key_name, webhook_url, created_at, started_at, finished_at, total, done, failed, error_code, error_message`

func scanJob(row interface{ Scan(...any) error }) (*Job, error) {
	var j Job
	var created int64
	var started, finished sql.NullInt64
	var code, msg sql.NullString
	if err := row.Scan(&j.ID, &j.Status, &j.Model, &j.Key, &j.WebhookURL, &created, &started, &finished,
		&j.Progress.Total, &j.Progress.Done, &j.Progress.Failed, &code, &msg); err != nil {
		return nil, err
	}
	j.CreatedAt = time.UnixMilli(created).UTC()
	j.StartedAt, j.FinishedAt = fromMS(started), fromMS(finished)
	if code.Valid {
		j.Error = &ErrorBody{Code: code.String, Message: msg.String}
	}
	return &j, nil
}

var errNotFound = errors.New("job not found")

func (s *store) get(ctx context.Context, id string) (*Job, error) {
	j, err := scanJob(s.db.QueryRowContext(ctx, `SELECT `+jobColumns+` FROM jobs WHERE id = ?`, id))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, errNotFound
	}
	return j, err
}

// list pages newest first; cursor is the last id of the previous page.
func (s *store) list(ctx context.Context, status, cursor, key string, limit int) ([]*Job, string, error) {
	q := `SELECT ` + jobColumns + ` FROM jobs WHERE 1=1`
	var args []any
	if status != "" {
		q += ` AND status = ?`
		args = append(args, status)
	}
	if key != "" {
		q += ` AND key_name = ?`
		args = append(args, key)
	}
	if cursor != "" {
		c, err := s.get(ctx, cursor)
		if err != nil {
			return nil, "", fmt.Errorf("bad cursor")
		}
		q += ` AND (created_at < ? OR (created_at = ? AND id < ?))`
		args = append(args, ms(c.CreatedAt), ms(c.CreatedAt), c.ID)
	}
	q += ` ORDER BY created_at DESC, id DESC LIMIT ?`
	args = append(args, limit+1)
	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, "", err
	}
	defer rows.Close()
	var out []*Job
	for rows.Next() {
		j, err := scanJob(rows)
		if err != nil {
			return nil, "", err
		}
		out = append(out, j)
	}
	next := ""
	if len(out) > limit {
		out = out[:limit]
		next = out[limit-1].ID
	}
	return out, next, rows.Err()
}

func (s *store) results(ctx context.Context, id string) ([]ItemResult, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT idx, custom_id, result, error_code, error_message FROM items WHERE job_id = ? ORDER BY idx`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ItemResult
	for rows.Next() {
		var r ItemResult
		var res []byte
		var code, msg sql.NullString
		if err := rows.Scan(&r.Index, &r.CustomID, &res, &code, &msg); err != nil {
			return nil, err
		}
		if len(res) > 0 {
			r.Result = &core.ChatResult{}
			_ = json.Unmarshal(res, r.Result)
		}
		if code.Valid {
			r.Error = &ErrorBody{Code: code.String, Message: msg.String}
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// pending lists a job's lines without an outcome, in order.
func (s *store) pending(ctx context.Context, id string) ([]int, [][]byte, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT idx, request FROM items WHERE job_id = ? AND result IS NULL AND error_code IS NULL ORDER BY idx`, id)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	var idx []int
	var reqs [][]byte
	for rows.Next() {
		var i int
		var b []byte
		if err := rows.Scan(&i, &b); err != nil {
			return nil, nil, err
		}
		idx, reqs = append(idx, i), append(reqs, b)
	}
	return idx, reqs, rows.Err()
}

func (s *store) saveItem(ctx context.Context, id string, idx int, res *core.ChatResult, e *ErrorBody) error {
	if res != nil {
		b, _ := json.Marshal(res)
		if _, err := s.db.ExecContext(ctx, `UPDATE items SET result = ? WHERE job_id = ? AND idx = ?`, b, id, idx); err != nil {
			return err
		}
		_, err := s.db.ExecContext(ctx, `UPDATE jobs SET done = done + 1 WHERE id = ?`, id)
		return err
	}
	if _, err := s.db.ExecContext(ctx, `UPDATE items SET error_code = ?, error_message = ? WHERE job_id = ? AND idx = ?`, e.Code, e.Message, id, idx); err != nil {
		return err
	}
	_, err := s.db.ExecContext(ctx, `UPDATE jobs SET done = done + 1, failed = failed + 1 WHERE id = ?`, id)
	return err
}

// setStatus moves a job to status only from one of the from statuses, so a
// cancel and the worker cannot overwrite each other; ok reports the move.
func (s *store) setStatus(ctx context.Context, id, status string, e *ErrorBody, from ...string) (bool, error) {
	now := ms(time.Now())
	where, args := ` WHERE id = ?`, []any{id}
	if len(from) > 0 {
		where += ` AND status IN (?` + strings.Repeat(`, ?`, len(from)-1) + `)`
		for _, f := range from {
			args = append(args, f)
		}
	}
	var r sql.Result
	var err error
	switch status {
	case StatusRunning:
		r, err = s.db.ExecContext(ctx, `UPDATE jobs SET status = ?, started_at = COALESCE(started_at, ?)`+where, append([]any{status, now}, args...)...)
	case StatusQueued:
		r, err = s.db.ExecContext(ctx, `UPDATE jobs SET status = ?`+where, append([]any{status}, args...)...)
	default:
		var code, msg any
		if e != nil {
			code, msg = e.Code, e.Message
		}
		r, err = s.db.ExecContext(ctx, `UPDATE jobs SET status = ?, finished_at = ?, error_code = ?, error_message = ?`+where, append([]any{status, now, code, msg}, args...)...)
	}
	if err != nil {
		return false, err
	}
	n, err := r.RowsAffected()
	return n > 0, err
}

// nextQueued is the oldest queued job.
func (s *store) nextQueued(ctx context.Context) (*Job, error) {
	j, err := scanJob(s.db.QueryRowContext(ctx, `SELECT `+jobColumns+` FROM jobs WHERE status = ? ORDER BY created_at, id LIMIT 1`, StatusQueued))
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	return j, err
}

func (s *store) countActive(ctx context.Context) (int, error) {
	var n int
	err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM jobs WHERE status IN (?, ?)`, StatusQueued, StatusRunning).Scan(&n)
	return n, err
}

// requeueRunning returns jobs a crash or restart left running to the queue;
// their finished lines keep their results.
func (s *store) requeueRunning(ctx context.Context) (int64, error) {
	r, err := s.db.ExecContext(ctx, `UPDATE jobs SET status = ? WHERE status = ?`, StatusQueued, StatusRunning)
	if err != nil {
		return 0, err
	}
	return r.RowsAffected()
}

// purgeFinishedBefore deletes the finished jobs (succeeded, failed or
// canceled) whose finish time is before cut; their lines go with them.
func (s *store) purgeFinishedBefore(ctx context.Context, cut time.Time) (int64, error) {
	res, err := s.db.ExecContext(ctx, `DELETE FROM jobs WHERE status IN (?, ?, ?) AND COALESCE(finished_at, created_at) < ?`,
		StatusSucceeded, StatusFailed, StatusCanceled, ms(cut))
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}
