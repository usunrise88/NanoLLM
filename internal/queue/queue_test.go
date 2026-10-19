package queue

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
)

// stubService answers every chat with the last user message upper-cased;
// "fail" fails, "block" waits for release or cancellation.
type stubService struct {
	core.Service
	release chan struct{}
	mu      sync.Mutex
	calls   int
	inter   []bool
}

func (s *stubService) Chat(ctx context.Context, req core.ChatRequest, _ core.Sink) (*core.ChatResult, error) {
	s.mu.Lock()
	s.calls++
	s.inter = append(s.inter, core.Interactive(ctx))
	s.mu.Unlock()
	text := req.Messages[len(req.Messages)-1].Content
	switch text {
	case "fail":
		return nil, core.Errorf(core.CodeContextOverflow, "too long")
	case "block":
		select {
		case <-s.release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return &core.ChatResult{Message: core.Message{Role: core.RoleAssistant, Content: strings.ToUpper(text)}, FinishReason: core.FinishStop}, nil
}

func open(t *testing.T, svc core.Service, opts Options) *Queue {
	t.Helper()
	if opts.Path == "" {
		opts.Path = filepath.Join(t.TempDir(), "q.db")
	}
	q, err := Open(svc, opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = q.Close() })
	return q
}

func batch(texts ...string) []byte {
	var b strings.Builder
	for i, t := range texts {
		line, _ := json.Marshal(map[string]any{"custom_id": "c" + string(rune('0'+i)), "messages": []map[string]string{{"role": "user", "content": t}}})
		b.Write(line)
		b.WriteByte('\n')
	}
	return []byte(b.String())
}

func wait(t *testing.T, q *Queue, id string, done func(*Job) bool) *Job {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		j, err := q.Get(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		if done(j) {
			return j
		}
		time.Sleep(10 * time.Millisecond)
	}
	j, _ := q.Get(context.Background(), id)
	t.Fatalf("job %s did not get there: %+v", id, j)
	return nil
}

func TestRunsAJob(t *testing.T) {
	svc := &stubService{}
	q := open(t, svc, Options{})
	j, err := q.Submit(context.Background(), batch("a", "fail", "c"), "gemma", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if j.Status != StatusQueued || j.Progress.Total != 3 {
		t.Fatalf("submitted: %+v", j)
	}
	j = wait(t, q, j.ID, (*Job).terminal)
	if j.Status != StatusSucceeded || j.Progress.Done != 3 || j.Progress.Failed != 1 || j.StartedAt == nil || j.FinishedAt == nil {
		t.Fatalf("finished: %+v", j)
	}
	if len(j.Results) != 3 || j.Results[0].Result.Message.Content != "A" || j.Results[0].CustomID != "c0" ||
		j.Results[1].Error == nil || j.Results[1].Error.Code != string(core.CodeContextOverflow) || j.Results[2].Index != 2 {
		t.Fatalf("results: %+v", j.Results)
	}
	for _, in := range svc.inter {
		if in {
			t.Fatal("a batch line ran as interactive")
		}
	}
}

func TestAllFailed(t *testing.T) {
	q := open(t, &stubService{}, Options{})
	j, _ := q.Submit(context.Background(), batch("fail", "fail"), "", "", "")
	j = wait(t, q, j.ID, (*Job).terminal)
	if j.Status != StatusFailed || j.Error == nil {
		t.Fatalf("%+v", j)
	}
}

func TestSubmitRefusesBadLines(t *testing.T) {
	q := open(t, &stubService{}, Options{})
	for body, want := range map[string]string{
		"":                    "no requests",
		"{\"messages\":[]}\n": "line 1: messages",
		"\n{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]}\n{oops}\n": "line 3",
		"{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"bogus\":1}": "line 1",
	} {
		_, err := q.Submit(context.Background(), []byte(body), "", "", "")
		if err == nil || !strings.Contains(err.Error(), want) || core.AsError(err).Code != core.CodeInvalidRequest {
			t.Errorf("%q: %v, want %q", body, err, want)
		}
	}
	for _, hook := range []string{"http://example.com/h", "ftp://x", "https://169.254.169.254/latest"} {
		if _, err := q.Submit(context.Background(), batch("a"), "", hook, ""); err == nil {
			t.Errorf("webhook %s accepted", hook)
		}
	}
}

func TestMaxPending(t *testing.T) {
	svc := &stubService{release: make(chan struct{})}
	q := open(t, svc, Options{MaxPending: 2})
	for range 2 {
		if _, err := q.Submit(context.Background(), batch("block"), "", "", ""); err != nil {
			t.Fatal(err)
		}
	}
	_, err := q.Submit(context.Background(), batch("a"), "", "", "")
	if core.AsError(err).Code != core.CodeQueueFull {
		t.Fatalf("got %v", err)
	}
	close(svc.release)
}

func TestCancel(t *testing.T) {
	svc := &stubService{release: make(chan struct{})}
	q := open(t, svc, Options{Concurrency: 1})
	j, _ := q.Submit(context.Background(), batch("a", "block", "c"), "", "", "")
	wait(t, q, j.ID, func(j *Job) bool { return j.Progress.Done == 1 })
	j, err := q.Cancel(context.Background(), j.ID)
	if err != nil || j.Status != StatusCanceled {
		t.Fatalf("%+v %v", j, err)
	}
	time.Sleep(50 * time.Millisecond)
	j, _ = q.Get(context.Background(), j.ID)
	if j.Status != StatusCanceled || j.Progress.Done != 1 || len(j.Results) != 3 || j.Results[0].Result == nil || j.Results[1].Result != nil {
		t.Fatalf("after cancel: %+v", j)
	}
	// A queued job is canceled before it starts.
	j2, _ := q.Submit(context.Background(), batch("a"), "", "", "")
	if j2, _ = q.Cancel(context.Background(), j2.ID); j2.Status != StatusCanceled {
		t.Fatalf("%+v", j2)
	}
	if _, err := q.Cancel(context.Background(), "job_nope"); core.AsError(err).Code != core.CodeJobNotFound {
		t.Fatal(err)
	}
}

func TestResumesAfterRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "q.db")
	svc := &stubService{release: make(chan struct{})}
	q, err := Open(svc, Options{Path: path, Concurrency: 1})
	if err != nil {
		t.Fatal(err)
	}
	j, _ := q.Submit(context.Background(), batch("a", "block", "c"), "", "", "")
	wait(t, q, j.ID, func(j *Job) bool { return j.Progress.Done == 1 })
	if err := q.Close(); err != nil {
		t.Fatal(err)
	}
	svc2 := &stubService{release: make(chan struct{})}
	close(svc2.release)
	q2 := open(t, svc2, Options{Path: path})
	j = wait(t, q2, j.ID, (*Job).terminal)
	if j.Status != StatusSucceeded || j.Results[1].Result.Message.Content != "BLOCK" || svc2.calls != 2 {
		t.Fatalf("%+v, %d calls after the restart", j, svc2.calls)
	}
}

func TestListPages(t *testing.T) {
	q := open(t, &stubService{}, Options{})
	var ids []string
	for range 5 {
		j, _ := q.Submit(context.Background(), batch("a"), "", "", "k1")
		ids = append(ids, j.ID)
		time.Sleep(2 * time.Millisecond)
	}
	if _, err := q.Submit(context.Background(), batch("a"), "", "", "k2"); err != nil {
		t.Fatal(err)
	}
	var got []string
	cursor := ""
	for {
		page, next, err := q.List(context.Background(), "", cursor, "k1", 2)
		if err != nil {
			t.Fatal(err)
		}
		for _, j := range page {
			got = append(got, j.ID)
		}
		if next == "" {
			break
		}
		cursor = next
	}
	if len(got) != 5 || got[0] != ids[4] || got[4] != ids[0] {
		t.Fatalf("got %v, want %v reversed", got, ids)
	}
	if _, _, err := q.List(context.Background(), "", "job_nope", "", 2); core.AsError(err).Code != core.CodeInvalidRequest {
		t.Fatal(err)
	}
}

func TestWebhookSigned(t *testing.T) {
	type hit struct {
		sig, ts string
		body    []byte
	}
	hits := make(chan hit, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		hits <- hit{r.Header.Get("X-NanoLLM-Signature"), r.Header.Get("X-NanoLLM-Timestamp"), b}
	}))
	defer srv.Close()
	q := open(t, &stubService{}, Options{WebhookSecret: "s3cret"})
	j, err := q.Submit(context.Background(), batch("a"), "", srv.URL+"/hook", "")
	if err != nil {
		t.Fatal(err)
	}
	select {
	case h := <-hits:
		if h.sig != "sha256="+Sign("s3cret", h.ts, h.body) {
			t.Fatalf("signature %s does not verify", h.sig)
		}
		var ev struct {
			Type string `json:"type"`
			Job  Job    `json:"job"`
		}
		if err := json.Unmarshal(h.body, &ev); err != nil || ev.Type != "job.succeeded" || ev.Job.ID != j.ID {
			t.Fatalf("%s %v", h.body, err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no webhook")
	}
}

func TestHTTP(t *testing.T) {
	svc := &stubService{release: make(chan struct{})}
	q := open(t, svc, Options{})
	mux := http.NewServeMux()
	q.MountNative(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/api/v1/jobs?model=gemma", "application/x-ndjson", strings.NewReader(string(batch("a", "block"))))
	if err != nil {
		t.Fatal(err)
	}
	var j Job
	_ = json.NewDecoder(resp.Body).Decode(&j)
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted || j.ID == "" || j.Model != "gemma" {
		t.Fatalf("%d %+v", resp.StatusCode, j)
	}

	resp, err = http.Post(srv.URL+"/api/v1/jobs", "application/x-ndjson", strings.NewReader("{bad}\n"))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad line: %d", resp.StatusCode)
	}

	// The event stream starts with the current state and ends terminal.
	events, err := http.Get(srv.URL + "/api/v1/jobs/" + j.ID + "/events")
	if err != nil {
		t.Fatal(err)
	}
	defer events.Body.Close()
	close(svc.release)
	sc := bufio.NewScanner(events.Body)
	sc.Buffer(nil, 1<<20)
	var last Job
	for sc.Scan() {
		if data, ok := strings.CutPrefix(sc.Text(), "data: "); ok {
			if err := json.Unmarshal([]byte(data), &last); err != nil {
				t.Fatal(err)
			}
		}
	}
	if last.Status != StatusSucceeded || len(last.Results) != 2 {
		t.Fatalf("last event: %+v", last)
	}

	resp, _ = http.Get(srv.URL + "/api/v1/jobs?limit=10")
	var page struct {
		Data []Job `json:"data"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&page)
	resp.Body.Close()
	if len(page.Data) != 1 || page.Data[0].ID != j.ID {
		t.Fatalf("%+v", page)
	}

	req, _ := http.NewRequest(http.MethodDelete, srv.URL+"/api/v1/jobs/job_missing", nil)
	resp, _ = http.DefaultClient.Do(req)
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("missing job: %d", resp.StatusCode)
	}
	if errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
}

// flakyService refuses the first n calls with a transient error, as a server
// whose wait queue is full or whose model is loading does.
type flakyService struct {
	stubService
	mu2   sync.Mutex
	fails int
	code  core.Code
}

func (s *flakyService) Chat(ctx context.Context, req core.ChatRequest, sink core.Sink) (*core.ChatResult, error) {
	s.mu2.Lock()
	if s.fails > 0 {
		s.fails--
		s.mu2.Unlock()
		return nil, core.Errorf(s.code, "try again")
	}
	s.mu2.Unlock()
	return s.stubService.Chat(ctx, req, sink)
}

// A line the server could not take for a moment is tried again, not recorded
// as failed: an overload that passes must not turn queued work into errors.
func TestTransientErrorsAreRetried(t *testing.T) {
	for _, code := range []core.Code{core.CodeQueueFull, core.CodeModelUnavailable} {
		svc := &flakyService{fails: 3, code: code}
		q := open(t, svc, Options{Concurrency: 1, RetryWait: time.Millisecond})
		j, _ := q.Submit(context.Background(), batch("a"), "", "", "")
		j = wait(t, q, j.ID, (*Job).terminal)
		if j.Status != StatusSucceeded || j.Progress.Failed != 0 || j.Results[0].Result == nil || j.Results[0].Result.Message.Content != "A" {
			t.Errorf("%s: %+v", code, j)
		}
	}
	// A real error is not retried, and neither is one that never clears.
	svc := &flakyService{fails: 1 << 20, code: core.CodeQueueFull}
	q := open(t, svc, Options{Concurrency: 1, RetryWait: time.Millisecond, RetryMax: 2})
	j, _ := q.Submit(context.Background(), batch("a"), "", "", "")
	j = wait(t, q, j.ID, (*Job).terminal)
	if j.Status != StatusFailed || j.Results[0].Error == nil || j.Results[0].Error.Code != string(core.CodeQueueFull) {
		t.Errorf("a line that never clears: %+v", j)
	}
}

// Finished jobs older than the retention are deleted with their lines;
// newer ones and jobs still queued stay.
func TestRetentionPurgesOldJobs(t *testing.T) {
	q := open(t, &stubService{}, Options{})
	j, _ := q.Submit(context.Background(), batch("a"), "", "", "")
	j = wait(t, q, j.ID, (*Job).terminal)
	if n, err := q.st.purgeFinishedBefore(context.Background(), time.Now().Add(-time.Hour)); err != nil || n != 0 {
		t.Fatalf("a fresh job was purged: %d, %v", n, err)
	}
	if n, err := q.st.purgeFinishedBefore(context.Background(), time.Now().Add(time.Hour)); err != nil || n != 1 {
		t.Fatalf("an old job was kept: %d, %v", n, err)
	}
	if _, err := q.Get(context.Background(), j.ID); err == nil {
		t.Error("the purged job can still be read")
	}
	var items int
	if err := q.st.db.QueryRow(`SELECT COUNT(*) FROM items`).Scan(&items); err != nil || items != 0 {
		t.Errorf("lines of a purged job remain: %d, %v", items, err)
	}
}
