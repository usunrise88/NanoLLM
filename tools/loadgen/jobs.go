package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"sync"
	"time"
)

// The queue part of the mix (docs/STAGE2.md, plan task 2): batch jobs of a few
// lines, each with a webhook that loadgen itself receives and verifies. A job
// is followed by polling or by its event stream, one in five is cancelled
// part-way, and a tenth of the webhook deliveries are answered with a 500 so
// that the server's retries run too. What the server owes is that every job
// reaches a terminal state, every terminal state is announced exactly in the
// signed webhook, and its database does not grow with the traffic.

// hookMsg is what the receiver took from one verified delivery.
type hookMsg struct {
	Type   string
	JobID  string
	Status string
}

// hooks is the webhook receiver. Deliveries are kept by job id until the job's
// waiter collects them, so one that arrives before the waiter is not lost.
type hooks struct {
	secret string
	url    string

	mu       sync.Mutex
	waiting  map[string]chan hookMsg
	refused  map[string]bool // jobs whose first delivery was answered with a 500
	badSig   int
	received int
}

// startHooks listens on addr (port 0 picks one) and serves POST /hook.
func startHooks(addr, secret string) (*hooks, func(), error) {
	ln, err := (&net.ListenConfig{}).Listen(context.Background(), "tcp", addr)
	if err != nil {
		return nil, nil, err
	}
	h := &hooks{secret: secret, waiting: map[string]chan hookMsg{}, refused: map[string]bool{}}
	h.url = "http://" + ln.Addr().String() + "/hook"
	mux := http.NewServeMux()
	mux.HandleFunc("POST /hook", h.handle)
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = srv.Serve(ln) }()
	return h, func() { _ = srv.Close() }, nil
}

// sign is the server's webhook signature (queue.Sign): HMAC-SHA256 over
// "<timestamp>.<body>".
func sign(secret, timestamp string, body []byte) string {
	m := hmac.New(sha256.New, []byte(secret))
	m.Write([]byte(timestamp + "."))
	m.Write(body)
	return hex.EncodeToString(m.Sum(nil))
}

func (h *hooks) handle(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
	ts := r.Header.Get("X-NanoLLM-Timestamp")
	want := "sha256=" + sign(h.secret, ts, body)
	var m struct {
		Type string `json:"type"`
		Job  struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		} `json:"job"`
	}
	ok := hmac.Equal([]byte(want), []byte(r.Header.Get("X-NanoLLM-Signature"))) && json.Unmarshal(body, &m) == nil && m.Job.ID != ""
	h.mu.Lock()
	if !ok {
		h.badSig++
		h.mu.Unlock()
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	// A tenth of the jobs get a 500 on their first delivery; the retry follows.
	if hash := sha256.Sum256([]byte(m.Job.ID)); hash[0]%10 == 0 && !h.refused[m.Job.ID] {
		h.refused[m.Job.ID] = true
		h.mu.Unlock()
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	h.received++
	ch := h.box(m.Job.ID)
	h.mu.Unlock()
	select {
	case ch <- hookMsg{Type: m.Type, JobID: m.Job.ID, Status: m.Job.Status}:
	default: // a repeated delivery of a job already announced
	}
	w.WriteHeader(http.StatusNoContent)
}

// box is the job's mailbox; the caller holds h.mu.
func (h *hooks) box(id string) chan hookMsg {
	ch, ok := h.waiting[id]
	if !ok {
		ch = make(chan hookMsg, 1)
		h.waiting[id] = ch
	}
	return ch
}

// await waits for the job's webhook; ok is false when none came in time.
func (h *hooks) await(ctx context.Context, id string, wait time.Duration) (hookMsg, bool) {
	h.mu.Lock()
	ch := h.box(id)
	h.mu.Unlock()
	defer func() {
		h.mu.Lock()
		delete(h.waiting, id)
		delete(h.refused, id)
		h.mu.Unlock()
	}()
	select {
	case m := <-ch:
		return m, true
	case <-time.After(wait):
	case <-ctx.Done():
	}
	return hookMsg{}, false
}

// jobStatus is the part of a job the checks read.
type jobStatus struct {
	ID       string `json:"id"`
	Status   string `json:"status"`
	Progress struct {
		Done, Total, Failed int
	} `json:"progress"`
	Results []struct {
		Error *struct{ Code string } `json:"error"`
	} `json:"results"`
}

func (j jobStatus) terminal() bool {
	return j.Status == "succeeded" || j.Status == "failed" || j.Status == "canceled"
}

// jobTimeout bounds one job: its lines queue behind the interactive load.
const jobTimeout = 20 * time.Minute

func (g *gen) rawDo(ctx context.Context, method, path, contentType string, body []byte) (*http.Response, error) {
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, g.base+path, rd)
	if err != nil {
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if g.key != "" {
		req.Header.Set("Authorization", "Bearer "+g.key)
	}
	return g.http.Do(req)
}

func (g *gen) getJob(ctx context.Context, id string) (jobStatus, int, error) {
	resp, err := g.rawDo(ctx, http.MethodGet, "/api/v1/jobs/"+id, "", nil)
	if err != nil {
		return jobStatus{}, 0, err
	}
	defer resp.Body.Close()
	var j jobStatus
	err = json.NewDecoder(resp.Body).Decode(&j)
	return j, resp.StatusCode, err
}

// follow waits for a terminal state: by the event stream, which the server
// ends with that state, or by polling every two seconds. cancelAfter, when
// positive, sends a DELETE after that long.
func (g *gen) follow(ctx context.Context, id string, stream bool, cancelAfter time.Duration) (jobStatus, error) {
	ctx, cancel := context.WithTimeout(ctx, jobTimeout)
	defer cancel()
	if cancelAfter > 0 {
		t := time.AfterFunc(cancelAfter, func() {
			if resp, err := g.rawDo(context.Background(), http.MethodDelete, "/api/v1/jobs/"+id, "", nil); err == nil {
				resp.Body.Close()
			}
		})
		defer t.Stop()
	}
	if stream {
		if resp, err := g.rawDo(ctx, http.MethodGet, "/api/v1/jobs/"+id+"/events", "", nil); err == nil {
			sc := bufio.NewScanner(resp.Body)
			for sc.Scan() { // until the server ends the stream
			}
			resp.Body.Close()
		}
	}
	for {
		j, code, err := g.getJob(ctx, id)
		switch {
		case err != nil && ctx.Err() != nil:
			return j, ctx.Err()
		case err == nil && code == http.StatusOK && j.terminal():
			return j, nil
		}
		select {
		case <-time.After(2 * time.Second):
		case <-ctx.Done():
			return j, ctx.Err()
		}
	}
}

// job submits one batch and judges the whole life of it.
func (g *gen) job(ctx context.Context, rng *rand.Rand) {
	lines := 2 + rng.IntN(4)
	var body bytes.Buffer
	for i := range lines {
		l, _ := json.Marshal(map[string]any{
			"custom_id": "l" + strconv.Itoa(i), "max_tokens": 24, "temperature": 0.7,
			"messages": []chatMsg{{"user", "Name one thing about " + topics[rng.IntN(len(topics))] + ", in six words."}},
		})
		body.Write(l)
		body.WriteByte('\n')
	}
	t0 := time.Now()
	resp, err := g.rawDo(ctx, http.MethodPost, "/api/v1/jobs?webhook_url="+url.QueryEscape(g.hooks.url), "application/x-ndjson", body.Bytes())
	if err != nil {
		if ctx.Err() == nil {
			g.add(sample{kind: "job", outcome: "error"})
		}
		return
	}
	var sub jobStatus
	_ = json.NewDecoder(resp.Body).Decode(&sub)
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted || sub.ID == "" {
		g.add(sample{kind: "job", outcome: "http_" + strconv.Itoa(resp.StatusCode)})
		return
	}
	var cancelAfter time.Duration
	if rng.IntN(5) == 0 {
		cancelAfter = time.Duration(2+rng.IntN(20)) * time.Second
	}
	j, err := g.follow(ctx, sub.ID, rng.IntN(2) == 0, cancelAfter)
	if err != nil {
		if ctx.Err() == nil {
			g.add(sample{kind: "job", outcome: "timeout"})
		}
		return
	}
	total := float64(time.Since(t0).Microseconds()) / 1000
	// The retries of a webhook wait 2, 4 and 8 s.
	hook, got := g.hooks.await(ctx, sub.ID, 40*time.Second)
	switch {
	case ctx.Err() != nil:
	case !got:
		g.add(sample{kind: "job", outcome: "no_webhook"})
		g.recordBad("job", map[string]any{"job": sub.ID, "status": j.Status}, nil)
	case hook.Status != j.Status:
		g.add(sample{kind: "job", outcome: "webhook_disagrees"})
	case j.Status == "canceled":
		g.add(sample{kind: "job", outcome: "cancelled", total: total})
	case j.Status != "succeeded" || len(j.Results) != lines || failedLines(j) > 0:
		g.add(sample{kind: "job", outcome: "failed"})
		g.recordBad("job", map[string]any{"job": sub.ID, "lines": lines}, j)
	default:
		g.add(sample{kind: "job", outcome: "ok", total: total})
	}
}

func failedLines(j jobStatus) int {
	n := 0
	for _, r := range j.Results {
		if r.Error != nil {
			n++
		}
	}
	return n
}

// recordBad keeps a job that went wrong, for -bad-out.
func (g *gen) recordBad(kind string, request map[string]any, resp any) {
	if g.bad == nil {
		return
	}
	line, _ := json.Marshal(map[string]any{"kind": kind, "why": kind, "request": request, "response": resp})
	g.mu.Lock()
	defer g.mu.Unlock()
	fmt.Fprintln(g.bad, string(line))
}

// dbBytes is the size of the queue's SQLite database with its journal files.
func dbBytes(path string) int64 {
	var n int64
	for _, suffix := range []string{"", "-wal", "-shm"} {
		if fi, err := os.Stat(path + suffix); err == nil {
			n += fi.Size()
		}
	}
	return n
}

// hookStats says how many deliveries were verified and how many were refused
// for a bad signature.
func (h *hooks) stats() (received, badSig int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.received, h.badSig
}
