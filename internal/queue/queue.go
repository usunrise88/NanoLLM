package queue

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
)

// Options configure the queue.
type Options struct {
	// Path is the SQLite database.
	Path string
	// MaxPending bounds jobs queued or running; beyond it Submit refuses.
	MaxPending int
	// Concurrency is how many lines of the running job are in flight at
	// once; the scheduler batches them on the model's slots.
	Concurrency int
	// WebhookSecret signs webhook bodies (X-NanoLLM-Signature: sha256=…).
	WebhookSecret string
	Log           *slog.Logger
	// HTTP is the webhook client.
	HTTP *http.Client
}

// Queue runs batch jobs: one at a time, oldest first, each line through
// core.Service as a batch (non-interactive) request, so interactive traffic
// overtakes it in the scheduler.
type Queue struct {
	opts  Options
	st    *store
	svc   core.Service
	wake  chan struct{}
	close chan struct{}
	done  chan struct{}

	mu      sync.Mutex
	cancel  map[string]context.CancelFunc
	subs    map[string]map[chan *Job]struct{}
	running string
}

// Open opens the database, puts jobs a restart interrupted back in the queue
// and starts the worker.
func Open(svc core.Service, opts Options) (*Queue, error) {
	if opts.MaxPending <= 0 {
		opts.MaxPending = 1000
	}
	if opts.Concurrency <= 0 {
		opts.Concurrency = 4
	}
	if opts.Log == nil {
		opts.Log = slog.Default()
	}
	if opts.HTTP == nil {
		opts.HTTP = &http.Client{Timeout: 30 * time.Second}
	}
	st, err := openStore(opts.Path)
	if err != nil {
		return nil, err
	}
	if n, err := st.requeueRunning(context.Background()); err == nil && n > 0 {
		opts.Log.Info("jobs resumed after restart", "jobs", n)
	}
	q := &Queue{opts: opts, st: st, svc: svc, wake: make(chan struct{}, 1), close: make(chan struct{}), done: make(chan struct{}),
		cancel: map[string]context.CancelFunc{}, subs: map[string]map[chan *Job]struct{}{}}
	go q.worker()
	return q, nil
}

// Close stops the worker; a job in progress is left running in the database
// and resumes on the next start.
func (q *Queue) Close() error {
	close(q.close)
	q.mu.Lock()
	for _, c := range q.cancel {
		c()
	}
	q.mu.Unlock()
	<-q.done
	return q.st.close()
}

// line is one JSONL line: a native chat request with an optional custom_id.
type line struct {
	core.ChatRequest
	CustomID string `json:"custom_id,omitempty"`
}

// Submit validates every line before anything is queued: a batch with a bad
// line is refused with the line number.
func (q *Queue) Submit(ctx context.Context, jsonl []byte, model, webhook, key string) (*Job, error) {
	if webhook != "" {
		if err := checkWebhook(webhook); err != nil {
			return nil, core.Errorf(core.CodeInvalidRequest, "webhook_url: %v", err).WithParam("webhook_url")
		}
	}
	var items []itemRequest
	for n, raw := range bytes.Split(jsonl, []byte("\n")) {
		raw = bytes.TrimSpace(raw)
		if len(raw) == 0 {
			continue
		}
		var l line
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&l); err != nil {
			return nil, core.Errorf(core.CodeInvalidRequest, "line %d: %v", n+1, err)
		}
		if len(l.Messages) == 0 {
			return nil, core.Errorf(core.CodeInvalidRequest, "line %d: messages must not be empty", n+1)
		}
		if model != "" && l.Model == "" {
			l.Model = model
		}
		body, _ := json.Marshal(l.ChatRequest)
		items = append(items, itemRequest{customID: l.CustomID, body: body})
	}
	if len(items) == 0 {
		return nil, core.Errorf(core.CodeInvalidRequest, "the batch has no requests")
	}
	if n, err := q.st.countActive(ctx); err != nil {
		return nil, err
	} else if n >= q.opts.MaxPending {
		return nil, core.Errorf(core.CodeQueueFull, "%d jobs are pending, the limit is queue.max_pending=%d", n, q.opts.MaxPending)
	}
	var b [10]byte
	_, _ = rand.Read(b[:])
	j := &Job{ID: "job_" + hex.EncodeToString(b[:]), Status: StatusQueued, Model: model, CreatedAt: time.Now().UTC(),
		Progress: Progress{Total: len(items)}, WebhookURL: webhook, Key: key, items: items}
	if err := q.st.insert(ctx, j); err != nil {
		return nil, err
	}
	q.publish(j)
	select {
	case q.wake <- struct{}{}:
	default:
	}
	return j, nil
}

// Get returns a job; a finished one carries its results.
func (q *Queue) Get(ctx context.Context, id string) (*Job, error) {
	j, err := q.st.get(ctx, id)
	if errors.Is(err, errNotFound) {
		return nil, core.Errorf(core.CodeJobNotFound, "no job %q", id)
	}
	if err != nil {
		return nil, err
	}
	if j.terminal() {
		if j.Results, err = q.st.results(ctx, id); err != nil {
			return nil, err
		}
	}
	return j, nil
}

// List pages jobs newest first. A non-admin key sees only its own jobs.
func (q *Queue) List(ctx context.Context, status, cursor, key string, limit int) ([]*Job, string, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	jobs, next, err := q.st.list(ctx, status, cursor, key, limit)
	if err != nil && err.Error() == "bad cursor" {
		return nil, "", core.Errorf(core.CodeInvalidRequest, "unknown cursor").WithParam("cursor")
	}
	return jobs, next, err
}

// Cancel stops a queued or running job; lines already done keep results.
func (q *Queue) Cancel(ctx context.Context, id string) (*Job, error) {
	j, err := q.Get(ctx, id)
	if err != nil {
		return nil, err
	}
	if j.terminal() {
		return j, nil
	}
	moved, err := q.st.setStatus(ctx, id, StatusCanceled, nil, StatusQueued, StatusRunning)
	if err != nil {
		return nil, err
	}
	q.mu.Lock()
	if c := q.cancel[id]; c != nil {
		c()
	}
	q.mu.Unlock()
	j, err = q.Get(ctx, id)
	if err == nil && moved {
		q.publish(j)
		q.notifyWebhook(j)
	}
	return j, err
}

// Subscribe follows a job's updates until it ends or ctx does.
func (q *Queue) Subscribe(id string) (<-chan *Job, func()) {
	ch := make(chan *Job, 16)
	q.mu.Lock()
	if q.subs[id] == nil {
		q.subs[id] = map[chan *Job]struct{}{}
	}
	q.subs[id][ch] = struct{}{}
	q.mu.Unlock()
	return ch, func() {
		q.mu.Lock()
		delete(q.subs[id], ch)
		if len(q.subs[id]) == 0 {
			delete(q.subs, id)
		}
		q.mu.Unlock()
	}
}

func (q *Queue) publish(j *Job) {
	q.mu.Lock()
	defer q.mu.Unlock()
	for ch := range q.subs[j.ID] {
		select {
		case ch <- j:
		default: // a slow reader gets the next update
		}
	}
}

func (q *Queue) worker() {
	defer close(q.done)
	for {
		select {
		case <-q.close:
			return
		default:
		}
		j, err := q.st.nextQueued(context.Background())
		if err != nil {
			q.opts.Log.Error("queue", "err", err)
		}
		if j == nil {
			select {
			case <-q.wake:
			case <-q.close:
				return
			case <-time.After(30 * time.Second):
			}
			continue
		}
		q.run(j)
	}
}

// run works one job's pending lines, Concurrency at a time.
func (q *Queue) run(j *Job) {
	ctx, cancel := context.WithCancel(core.WithInteractive(context.Background(), false))
	q.mu.Lock()
	q.cancel[j.ID] = cancel
	q.running = j.ID
	q.mu.Unlock()
	defer func() {
		cancel()
		q.mu.Lock()
		delete(q.cancel, j.ID)
		q.running = ""
		q.mu.Unlock()
	}()
	bg := context.Background()
	if ok, err := q.st.setStatus(bg, j.ID, StatusRunning, nil, StatusQueued); err != nil || !ok {
		if err != nil {
			q.opts.Log.Error("queue", "job", j.ID, "err", err)
			time.Sleep(time.Second) // the database is failing: do not spin
		}
		return // canceled before it started
	}
	idx, reqs, err := q.st.pending(bg, j.ID)
	if err != nil {
		q.opts.Log.Error("queue", "job", j.ID, "err", err)
		return
	}
	if cur, _ := q.st.get(bg, j.ID); cur != nil {
		q.publish(cur)
	}
	sem := make(chan struct{}, q.opts.Concurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex
	for k := range idx {
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
		}
		if ctx.Err() != nil {
			break
		}
		wg.Add(1)
		go func(i int, body []byte) {
			defer wg.Done()
			defer func() { <-sem }()
			var req core.ChatRequest
			_ = json.Unmarshal(body, &req)
			res, err := q.svc.Chat(ctx, req, nil)
			if ctx.Err() != nil {
				return // canceled or shutting down: the line stays pending
			}
			var e *ErrorBody
			if err != nil {
				ce := core.AsError(err)
				e = &ErrorBody{Code: string(ce.Code), Message: ce.Message}
				res = nil
			}
			mu.Lock()
			defer mu.Unlock()
			if err := q.st.saveItem(bg, j.ID, i, res, e); err != nil {
				q.opts.Log.Error("queue", "job", j.ID, "err", err)
			}
			if cur, _ := q.st.get(bg, j.ID); cur != nil {
				q.publish(cur)
			}
		}(idx[k], reqs[k])
	}
	wg.Wait()
	select {
	case <-q.close:
		return // resumes on the next start
	default:
	}
	cur, err := q.st.get(bg, j.ID)
	if err != nil || cur.Status == StatusCanceled {
		return
	}
	status, eb := StatusSucceeded, (*ErrorBody)(nil)
	if cur.Progress.Failed == cur.Progress.Total {
		status, eb = StatusFailed, &ErrorBody{Code: "all_failed", Message: "every request in the batch failed"}
	}
	if ok, err := q.st.setStatus(bg, j.ID, status, eb, StatusRunning); err != nil || !ok {
		if err != nil {
			q.opts.Log.Error("queue", "job", j.ID, "err", err)
		}
		return
	}
	if final, err := q.Get(bg, j.ID); err == nil {
		q.publish(final)
		q.notifyWebhook(final)
	}
}

// checkWebhook keeps webhooks to https, or http on loopback for local
// tools, and away from link-local addresses (cloud metadata endpoints).
func checkWebhook(raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return errors.New("not a URL")
	}
	host := u.Hostname()
	ip := net.ParseIP(host)
	loopback := host == "localhost" || (ip != nil && ip.IsLoopback())
	switch {
	case u.Scheme == "https":
	case u.Scheme == "http" && loopback:
	default:
		return errors.New("must be https (http only on loopback)")
	}
	if ip != nil && (ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified()) {
		return errors.New("link-local and unspecified addresses are refused")
	}
	return nil
}

// notifyWebhook posts the finished job, signed, with a few retries. It runs
// in the background and never fails the job.
func (q *Queue) notifyWebhook(j *Job) {
	if j.WebhookURL == "" {
		return
	}
	payload := *j
	payload.Results = nil // the receiver fetches them; a webhook stays small
	body, _ := json.Marshal(map[string]any{"type": "job." + j.Status, "job": payload})
	go func() {
		for attempt := range 4 {
			if attempt > 0 {
				time.Sleep(time.Duration(1<<attempt) * time.Second)
			}
			req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, j.WebhookURL, bytes.NewReader(body))
			if err != nil {
				return
			}
			ts := strconv.FormatInt(time.Now().Unix(), 10)
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-NanoLLM-Timestamp", ts)
			if q.opts.WebhookSecret != "" {
				req.Header.Set("X-NanoLLM-Signature", "sha256="+Sign(q.opts.WebhookSecret, ts, body))
			}
			resp, err := q.opts.HTTP.Do(req)
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode/100 == 2 {
					return
				}
				err = fmt.Errorf("status %s", resp.Status)
			}
			q.opts.Log.Warn("webhook failed", "job", j.ID, "attempt", attempt+1, "err", err)
		}
	}()
}

// Sign is the webhook signature: HMAC-SHA256 over "<timestamp>.<body>", so a
// replayed body with a fresh timestamp does not verify.
func Sign(secret, timestamp string, body []byte) string {
	m := hmac.New(sha256.New, []byte(secret))
	m.Write([]byte(timestamp + "."))
	m.Write(body)
	return hex.EncodeToString(m.Sum(nil))
}
