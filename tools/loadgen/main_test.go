package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/queue"
)

func TestAnswerAndWhy(t *testing.T) {
	for _, c := range []struct {
		name, body string
		calls      int
		why        string
	}{
		{"a call", `{"choices":[{"finish_reason":"tool_calls","message":{"content":"","tool_calls":[{"id":"1"}]}}]}`, 1, "empty"},
		{"a null list", `{"choices":[{"finish_reason":"stop","message":{"content":"Sunny.","tool_calls":null}}]}`, 0, "text instead of a call"},
		{"reasoning without room left", `{"choices":[{"finish_reason":"length","message":{"content":"","reasoning_content":"1. Analyse"}}]}`, 0, "budget spent on reasoning"},
		{"an answer cut short", `{"choices":[{"finish_reason":"length","message":{"content":"The weather"}}]}`, 0, "budget spent on the answer"},
		{"not JSON", `oops`, 0, "unreadable"},
		{"no choices", `{"choices":[]}`, 0, "unreadable"},
	} {
		a := answerOf([]byte(c.body))
		if a.toolCalls != c.calls || a.why() != c.why {
			t.Errorf("%s: %d calls, why %q", c.name, a.toolCalls, a.why())
		}
	}
}

func TestClaimSharesTheBudget(t *testing.T) {
	g := &gen{limit: 3}
	n := 0
	for g.claim() {
		n++
	}
	if n != 3 || g.claim() {
		t.Fatalf("claimed %d", n)
	}
	g = &gen{} // no -n: never runs out
	for range 1000 {
		if !g.claim() {
			t.Fatal("an unlimited run ran out")
		}
	}
}

func post(t *testing.T, h *hooks, id, status, signature string) int {
	t.Helper()
	body := []byte(`{"type":"job.` + status + `","job":{"id":"` + id + `","status":"` + status + `"}}`)
	req, _ := http.NewRequest(http.MethodPost, h.url, bytes.NewReader(body))
	req.Header.Set("X-NanoLLM-Timestamp", "1700000000")
	if signature == "" {
		signature = "sha256=" + sign(h.secret, "1700000000", body)
	}
	req.Header.Set("X-NanoLLM-Signature", signature)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	return resp.StatusCode
}

// The receiver verifies the signature the server puts on a webhook, hands the
// delivery to whoever waits for the job (even if it came first), and answers a
// tenth of the jobs' first delivery with a 500 so that the retry runs.
func TestWebhookReceiver(t *testing.T) {
	h, stop, err := startHooks("127.0.0.1:0", "s3cret")
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	var plain, refused string
	for i := 0; plain == "" || refused == ""; i++ {
		id := "job_" + strconv.Itoa(i)
		if sum := sha256.Sum256([]byte(id)); sum[0]%10 == 0 {
			refused = id
		} else {
			plain = id
		}
	}
	if code := post(t, h, plain, "succeeded", "sha256=00"); code != http.StatusUnauthorized {
		t.Fatalf("a wrong signature: %d", code)
	}
	if code := post(t, h, plain, "succeeded", ""); code != http.StatusNoContent {
		t.Fatalf("a good delivery: %d", code)
	}
	m, ok := h.await(context.Background(), plain, time.Second) // it came before the wait began
	if !ok || m.Status != "succeeded" || m.JobID != plain {
		t.Fatalf("%+v %v", m, ok)
	}
	if code := post(t, h, refused, "canceled", ""); code != http.StatusInternalServerError {
		t.Fatalf("the first delivery of %s: %d", refused, code)
	}
	if code := post(t, h, refused, "canceled", ""); code != http.StatusNoContent {
		t.Fatalf("its retry: %d", code)
	}
	if m, ok = h.await(context.Background(), refused, time.Second); !ok || m.Status != "canceled" {
		t.Fatalf("%+v %v", m, ok)
	}
	if _, ok = h.await(context.Background(), "job_never", 50*time.Millisecond); ok {
		t.Fatal("a job nobody announced")
	}
	if recv, bad := h.stats(); recv != 2 || bad != 1 {
		t.Fatalf("received %d, refused %d", recv, bad)
	}
}

func TestPickMixWithAndWithoutJobs(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	count := func(g *gen) map[string]int {
		c := map[string]int{}
		for range 4000 {
			c[g.pick(rng)]++
		}
		return c
	}
	if c := count(&gen{}); c["job"] != 0 || c["chat"] < 2000 {
		t.Errorf("without a receiver: %v", c)
	}
	if c := count(&gen{hooks: &hooks{}}); c["job"] < 150 || c["job"] > 330 {
		t.Errorf("with a receiver jobs are about six percent: %v", c)
	}
	if c := count(&gen{only: "tool"}); c["tool"] != 4000 {
		t.Errorf("-only: %v", c)
	}
}

// echoService answers every chat with its last message.
type echoService struct{ core.Service }

func (echoService) Chat(_ context.Context, req core.ChatRequest, _ core.Sink) (*core.ChatResult, error) {
	return &core.ChatResult{Message: core.Message{Role: core.RoleAssistant, Content: req.Messages[len(req.Messages)-1].Content}, FinishReason: core.FinishStop}, nil
}

// A job the generator builds is a batch the real queue accepts, runs and
// announces in a webhook that verifies: the format of the lines, the signature,
// the retry after the receiver's first 500.
func TestJobAgainstTheRealQueue(t *testing.T) {
	h, stop, err := startHooks("127.0.0.1:0", "s3cret")
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	q, err := queue.Open(echoService{}, queue.Options{
		Path: filepath.Join(t.TempDir(), "q.db"), WebhookSecret: "s3cret", HTTP: http.DefaultClient,
		Log: slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	mux := http.NewServeMux()
	q.MountNative(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()
	g := &gen{base: srv.URL, http: http.DefaultClient, hooks: h}
	rng := rand.New(rand.NewPCG(7, 7))
	for range 12 { // enough jobs that one of them meets the 500 of the receiver
		g.job(context.Background(), rng)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	ok, other := g.totals["job/ok"], 0
	for k, v := range g.totals {
		if k != "job/ok" && k != "job/cancelled" {
			other += v
		}
	}
	if other != 0 || ok+g.totals["job/cancelled"] != 12 || ok == 0 {
		t.Fatalf("totals %v", g.totals)
	}
}
