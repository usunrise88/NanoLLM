package queue

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/engine"
	"github.com/usunrise88/nanollm/internal/pool"
)

// The queue over the real pipeline (internal/engine, its pool and scheduler)
// with the fake backend, as `nanollm serve` assembles them: a line is a chat
// request that meets the same slots, errors and cancellation a client's does.

func newEngine(t *testing.T, opts fake.Options, spec backend.LoadSpec, maxPending int) (*engine.Service, *pool.Pool) {
	t.Helper()
	p := pool.New(func(ctx context.Context, k pool.Key) (backend.Model, any, error) {
		m, err := fake.New(opts).Load(ctx, spec)
		return m, nil, err
	}, 1)
	p.MaxPending = maxPending
	t.Cleanup(p.Close) // after the queue's Close: it was registered later
	svc := engine.New(engine.Config{DefaultModel: "tiny", PromptCache: true}, p,
		func(_ context.Context, model, _ string) (engine.Target, error) {
			if model != "tiny" {
				return engine.Target{}, core.Errorf(core.CodeModelNotFound, "no model %q", model)
			}
			return engine.Target{Key: pool.Key{Model: model, Backend: "fake", Device: "cpu", NCtx: 4096}}, nil
		})
	return svc, p
}

// jsonl makes a batch from lines given as objects, for fields batch() has no
// place for (a line's own model, stop sequences).
func jsonl(t *testing.T, objs ...map[string]any) []byte {
	t.Helper()
	var b strings.Builder
	for _, o := range objs {
		line, err := json.Marshal(o)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	return []byte(b.String())
}

func msg(text string) []map[string]string {
	return []map[string]string{{"role": "user", "content": text}}
}

// refusing counts the lines the pipeline turned away for a moment.
type refusing struct {
	core.Service
	refused atomic.Int32
}

func (s *refusing) Chat(ctx context.Context, req core.ChatRequest, sink core.Sink) (*core.ChatResult, error) {
	res, err := s.Service.Chat(ctx, req, sink)
	if err != nil && transient(err) {
		s.refused.Add(1)
	}
	return res, err
}

func TestJobRunsThroughTheEngine(t *testing.T) {
	svc, _ := newEngine(t, fake.Options{}, backend.LoadSpec{Slots: 2}, 0)
	q := open(t, svc, Options{Concurrency: 3})
	j, err := q.Submit(context.Background(), jsonl(t,
		map[string]any{"custom_id": "echo", "messages": msg("hello batch [[fake:echo]]")},
		map[string]any{"custom_id": "stop", "messages": msg("alpha beta gamma [[fake:echo]]"), "stop": []string{" gam"}},
		map[string]any{"custom_id": "toolong", "messages": msg(strings.Repeat("x", 5000))},
		map[string]any{"custom_id": "nomodel", "model": "nope", "messages": msg("x")},
		map[string]any{"custom_id": "json", "messages": msg("[[fake:json]]")},
	), "tiny", "", "")
	if err != nil {
		t.Fatal(err)
	}
	j = wait(t, q, j.ID, (*Job).terminal)
	if j.Status != StatusSucceeded || j.Progress.Done != 5 || j.Progress.Failed != 2 {
		out, _ := json.Marshal(j)
		t.Fatalf("finished: %s", out)
	}
	byID := map[string]ItemResult{}
	for _, r := range j.Results {
		byID[r.CustomID] = r
	}
	for id, want := range map[string]string{"echo": "hello batch", "stop": "alpha beta", "json": `{"ok":true,"items":[1,2,3],"name":"fake"}`} {
		r := byID[id]
		if r.Result == nil || r.Result.Message.Content != want || r.Result.Usage.CompletionTokens == 0 {
			t.Errorf("%s: %+v", id, r)
		}
	}
	// A failure is a line's own: recorded with the pipeline's code, not retried.
	if r := byID["toolong"]; r.Error == nil || r.Error.Code != string(core.CodeContextOverflow) {
		t.Errorf("toolong: %+v", r)
	}
	if r := byID["nomodel"]; r.Error == nil || r.Error.Code != string(core.CodeModelNotFound) {
		t.Errorf("nomodel: %+v", r)
	}
	for i, r := range j.Results {
		if r.Index != i {
			t.Errorf("result %d has index %d", i, r.Index)
		}
	}
}

// More lines than the model can take at once: the pipeline refuses the extra
// ones with queue_full while slots and the wait queue are taken, and the
// queue tries them again instead of recording them as failed.
func TestEngineRefusalsAreRetried(t *testing.T) {
	inner, _ := newEngine(t, fake.Options{DecodeRate: 100}, backend.LoadSpec{Slots: 1}, 1)
	svc := &refusing{Service: inner}
	q := open(t, svc, Options{Concurrency: 4, RetryWait: 2 * time.Millisecond, RetryMax: 12})
	var lines []map[string]any
	for i := range 6 {
		lines = append(lines, map[string]any{"custom_id": string(rune('a' + i)), "messages": msg("line " + string(rune('a'+i)) + " [[fake:echo]]")})
	}
	j, err := q.Submit(context.Background(), jsonl(t, lines...), "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	j = wait(t, q, j.ID, (*Job).terminal)
	if j.Status != StatusSucceeded || j.Progress.Failed != 0 || j.Progress.Done != 6 {
		t.Fatalf("finished: %+v", j)
	}
	for i, r := range j.Results {
		want := "line " + string(rune('a'+i))
		if r.Result == nil || r.Result.Message.Content != want {
			t.Errorf("line %d: %+v, want %q", i, r, want)
		}
	}
	if svc.refused.Load() == 0 {
		t.Error("the model never turned a line away: nothing was retried")
	}
}

// Cancelling a job reaches the lines being generated: their requests are
// cancelled in the scheduler, the slots are freed, and the model goes on
// serving other work.
func TestCancelStopsLinesInTheEngine(t *testing.T) {
	svc, p := newEngine(t, fake.Options{DecodeRate: 500}, backend.LoadSpec{Slots: 2}, 0)
	q := open(t, svc, Options{Concurrency: 2})
	j, err := q.Submit(context.Background(), batch("one [[fake:loop]]", "two [[fake:loop]]", "three [[fake:loop]]"), "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	busy := func() int {
		n := 0
		for _, s := range p.Status() {
			n += s.Loop.Busy
		}
		return n
	}
	eventually(t, "both slots to decode", func() bool { return busy() == 2 })
	j, err = q.Cancel(context.Background(), j.ID)
	if err != nil || j.Status != StatusCanceled {
		t.Fatalf("%+v %v", j, err)
	}
	eventually(t, "the slots to free", func() bool { return busy() == 0 })
	j, _ = q.Get(context.Background(), j.ID)
	if j.Status != StatusCanceled || j.Progress.Done != 0 {
		t.Fatalf("after the cancel: %+v", j)
	}
	res, err := svc.Chat(context.Background(), core.ChatRequest{Messages: []core.Message{{Role: core.RoleUser, Content: "still serving [[fake:echo]]"}}}, nil)
	if err != nil || res.Message.Content != "still serving" {
		t.Fatalf("after the cancel: %v %+v", err, res)
	}
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

// Jobs that finished before the retention are deleted when the queue starts
// (and hourly after), with their lines; newer ones stay, and a queue with no
// retention keeps everything.
func TestRetentionPurgesOnStart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "q.db")
	svc, _ := newEngine(t, fake.Options{}, backend.LoadSpec{}, 0)
	q, err := Open(svc, Options{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	old, _ := q.Submit(context.Background(), batch("old [[fake:echo]]", "older [[fake:echo]]"), "", "", "")
	fresh, _ := q.Submit(context.Background(), batch("fresh [[fake:echo]]"), "", "", "")
	wait(t, q, old.ID, (*Job).terminal)
	wait(t, q, fresh.ID, (*Job).terminal)
	if err := q.Close(); err != nil {
		t.Fatal(err)
	}
	// Age one job by 40 days, behind the queue's back.
	st, err := openStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.db.Exec(`UPDATE jobs SET finished_at = ? WHERE id = ?`, ms(time.Now().AddDate(0, 0, -40)), old.ID); err != nil {
		t.Fatal(err)
	}
	if err := st.close(); err != nil {
		t.Fatal(err)
	}

	forever, err := Open(svc, Options{Path: path})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := forever.Get(context.Background(), old.ID); err != nil {
		t.Fatalf("a queue with no retention deleted a job: %v", err)
	}
	if err := forever.Close(); err != nil {
		t.Fatal(err)
	}

	thirty, err := Open(svc, Options{Path: path, RetentionDays: 30})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = thirty.Close() })
	eventually(t, "the old job to be purged", func() bool {
		_, err := thirty.Get(context.Background(), old.ID)
		return err != nil
	})
	if j, err := thirty.Get(context.Background(), fresh.ID); err != nil || len(j.Results) != 1 {
		t.Fatalf("the fresh job did not survive the purge: %+v, %v", j, err)
	}
	var items int
	if err := thirty.st.db.QueryRow(`SELECT COUNT(*) FROM items WHERE job_id = ?`, old.ID).Scan(&items); err != nil || items != 0 {
		t.Fatalf("lines of the purged job remain: %d, %v", items, err)
	}
}

// A job that finishes right before the queue is closed still gets its
// webhook: Close waits for the delivery in flight, and gives up the retries
// rather than wait out their pauses.
func TestCloseWaitsForWebhooks(t *testing.T) {
	var delivered atomic.Int32
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		time.Sleep(150 * time.Millisecond)
		delivered.Add(1)
	}))
	defer slow.Close()
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer failing.Close()

	t.Run("an attempt in flight completes", func(t *testing.T) {
		q, err := Open(&stubService{}, Options{Path: filepath.Join(t.TempDir(), "q.db")})
		if err != nil {
			t.Fatal(err)
		}
		j, err := q.Submit(context.Background(), batch("a"), "", slow.URL+"/hook", "")
		if err != nil {
			t.Fatal(err)
		}
		wait(t, q, j.ID, (*Job).terminal)
		if err := q.Close(); err != nil {
			t.Fatal(err)
		}
		if got := delivered.Load(); got != 1 {
			t.Fatalf("%d webhooks delivered by the time Close returned, want 1", got)
		}
	})
	t.Run("retries are given up", func(t *testing.T) {
		q, err := Open(&stubService{}, Options{Path: filepath.Join(t.TempDir(), "q.db")})
		if err != nil {
			t.Fatal(err)
		}
		j, err := q.Submit(context.Background(), batch("a"), "", failing.URL+"/hook", "")
		if err != nil {
			t.Fatal(err)
		}
		wait(t, q, j.ID, (*Job).terminal)
		start := time.Now()
		if err := q.Close(); err != nil {
			t.Fatal(err)
		}
		// The first retry would wait 2 s.
		if d := time.Since(start); d > time.Second {
			t.Fatalf("Close took %v: it waited for a webhook retry", d)
		}
	})
}
