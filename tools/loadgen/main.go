// Command loadgen drives a running NanoLLM server with a mixed, long-lived
// load and records what an operator would watch: latency percentiles, errors,
// cache hits, the server's memory and goroutines (docs/STAGE2.md 2.8, the
// soak test).
//
//	go run ./tools/loadgen -base http://127.0.0.1:8080 -key KEY -workers 8 -duration 24h -out soak.jsonl -pid $(pgrep -x nanollm)
//	go run ./tools/loadgen -analyze soak.jsonl
//
// The mix per worker: multi-turn conversations that keep a session_id and
// stream (the prompt cache must hit on every turn after the first), requests
// the client abandons mid-stream, tool calls, JSON-schema responses and long
// prompts. One line of JSON per interval goes to -out; -analyze reads such a
// file and says whether memory, goroutines and latency stayed flat.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

func main() {
	base := flag.String("base", "http://127.0.0.1:8080", "server URL")
	key := flag.String("key", os.Getenv("NANOLLM_API_KEY"), "API key (an admin key to read /api/v1/stats)")
	model := flag.String("model", "", "model id (default: the server's)")
	workers := flag.Int("workers", 4, "concurrent clients")
	duration := flag.Duration("duration", time.Hour, "how long to run")
	interval := flag.Duration("interval", time.Minute, "report interval")
	out := flag.String("out", "", "JSON-lines report file (default stdout)")
	pid := flag.Int("pid", 0, "the server's pid, to record its RSS from /proc")
	maxTokens := flag.Int("max-tokens", 96, "tokens per answer")
	seed := flag.Uint64("seed", 1, "random seed of the mix")
	analyze := flag.String("analyze", "", "analyze a report file instead of running")
	only := flag.String("only", "", "run one kind of request only: chat, abandon, tool, json, long or job (default: the mix)")
	hookListen := flag.String("hook-listen", "", "address for the webhook receiver (127.0.0.1:0 picks a port); adds queue jobs to the mix")
	hookSecret := flag.String("hook-secret", "loadgen-secret", "the server's queue.webhook_secret")
	queueDB := flag.String("queue-db", "", "the server's queue database (nanollm.db), to record its size")
	count := flag.Int("n", 0, "stop after this many requests (0: run for -duration)")
	extra := flag.String("extra", "", "JSON object merged into every request body, e.g. '{\"max_reasoning_tokens\":64}'")
	badOut := flag.String("bad-out", "", "append every answer the checker rejected (request and response) to this JSON-lines file")
	flag.Parse()

	if *analyze != "" {
		if err := analyzeFile(*analyze, os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, "loadgen:", err)
			os.Exit(1)
		}
		return
	}
	w := io.Writer(os.Stdout)
	if *out != "" {
		f, err := os.OpenFile(*out, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			fmt.Fprintln(os.Stderr, "loadgen:", err)
			os.Exit(1)
		}
		defer f.Close()
		w = f
	}
	if *only != "" && !slices.Contains(kinds, *only) {
		fmt.Fprintf(os.Stderr, "loadgen: -only is one of %s\n", strings.Join(kinds, ", "))
		os.Exit(2)
	}
	g := &gen{base: strings.TrimRight(*base, "/"), key: *key, model: *model, maxTokens: *maxTokens, pid: *pid,
		http: &http.Client{Timeout: 0}, start: time.Now(), only: *only, limit: *count, queueDB: *queueDB}
	if *hookListen != "" {
		h, stop, err := startHooks(*hookListen, *hookSecret)
		if err != nil {
			fmt.Fprintln(os.Stderr, "loadgen:", err)
			os.Exit(1)
		}
		defer stop()
		g.hooks = h
	}
	if *only == "job" && g.hooks == nil {
		fmt.Fprintln(os.Stderr, "loadgen: -only job needs -hook-listen")
		os.Exit(2)
	}
	if *extra != "" {
		if err := json.Unmarshal([]byte(*extra), &g.extra); err != nil {
			fmt.Fprintln(os.Stderr, "loadgen: -extra:", err)
			os.Exit(2)
		}
	}
	if *badOut != "" {
		f, err := os.OpenFile(*badOut, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			fmt.Fprintln(os.Stderr, "loadgen:", err)
			os.Exit(1)
		}
		defer f.Close()
		g.bad = f
	}
	ctx, cancel := context.WithTimeout(context.Background(), *duration)
	defer cancel()
	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			g.worker(ctx, rand.New(rand.NewPCG(*seed, uint64(i)+1)), i)
		}()
	}
	finished := make(chan struct{}) // every worker has spent its share of -n
	go func() { wg.Wait(); close(finished) }()
	t := time.NewTicker(*interval)
	defer t.Stop()
	for done := false; !done; {
		select {
		case <-ctx.Done():
			done = true
		case <-finished:
			done = true
		case <-t.C:
		}
		line, _ := json.Marshal(g.report())
		fmt.Fprintln(w, string(line))
	}
	wg.Wait()
}

type gen struct {
	base, key, model string
	maxTokens, pid   int
	http             *http.Client
	start            time.Time

	hooks   *hooks         // the webhook receiver; nil leaves queue jobs out of the mix
	queueDB string         // the server's queue database, whose size goes into the report
	extra   map[string]any // merged into every request body
	only    string         // one kind of request, or "" for the mix
	limit   int            // requests to send before stopping; 0 is no limit
	sent    int
	bad     io.Writer // rejected answers, one JSON object per line

	mu     sync.Mutex
	window []sample // since the last report
	totals map[string]int
}

type sample struct {
	kind    string
	outcome string // ok | cancelled | http_<status> | error
	ttft    float64
	total   float64
	cached  int
	prompt  int
}

func (g *gen) add(s sample) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.window = append(g.window, s)
	if g.totals == nil {
		g.totals = map[string]int{}
	}
	g.totals[s.kind+"/"+s.outcome]++
}

// Report is one line of the output.
type Report struct {
	T         string             `json:"t"`
	ElapsedS  float64            `json:"elapsed_s"`
	Requests  int                `json:"requests"`
	Outcomes  map[string]int     `json:"outcomes"`
	Totals    map[string]int     `json:"totals"`
	TTFT      map[string]float64 `json:"ttft_ms"`
	Total     map[string]float64 `json:"total_ms"`
	CacheHit  float64            `json:"cache_hit"`
	RSS       int64              `json:"rss"`
	Server    *serverStats       `json:"server,omitempty"`
	ServerErr string             `json:"server_err,omitempty"`
	// QueueDB is the size of the queue database and its journal files, and
	// Hooks the webhook deliveries verified and refused so far.
	QueueDB int64 `json:"queue_db_bytes,omitempty"`
	Hooks   *struct {
		Received int `json:"received"`
		BadSig   int `json:"bad_signature"`
	} `json:"hooks,omitempty"`
}

type serverStats struct {
	Goroutines int     `json:"goroutines"`
	RSS        int64   `json:"rss"`
	RSSPeak    int64   `json:"rss_peak"`
	Locked     int64   `json:"locked"`
	GoHeap     uint64  `json:"go_heap"`
	Busy       int     `json:"slots_busy"`
	Slots      int     `json:"slots"`
	Pending    int     `json:"pending"`
	HitRate    float64 `json:"cache_hit_rate"`
}

func (g *gen) report() Report {
	g.mu.Lock()
	win := g.window
	g.window = nil
	totals := map[string]int{}
	for k, v := range g.totals {
		totals[k] = v
	}
	g.mu.Unlock()
	r := Report{T: time.Now().UTC().Format(time.RFC3339), ElapsedS: time.Since(g.start).Seconds(), Requests: len(win), Outcomes: map[string]int{}, Totals: totals}
	var ttft, total []float64
	var cached, prompt int
	for _, s := range win {
		r.Outcomes[s.outcome]++
		if s.outcome == "ok" {
			if s.ttft > 0 {
				ttft = append(ttft, s.ttft)
			}
			total = append(total, s.total)
			cached += s.cached
			prompt += s.prompt
		}
	}
	r.TTFT, r.Total = pcts(ttft), pcts(total)
	if prompt > 0 {
		r.CacheHit = float64(cached) / float64(prompt)
	}
	if g.pid > 0 {
		r.RSS = procRSS(g.pid)
	}
	if g.queueDB != "" {
		r.QueueDB = dbBytes(g.queueDB)
	}
	if g.hooks != nil {
		recv, bad := g.hooks.stats()
		r.Hooks = &struct {
			Received int `json:"received"`
			BadSig   int `json:"bad_signature"`
		}{recv, bad}
	}
	if st, err := g.stats(); err != nil {
		r.ServerErr = err.Error()
	} else {
		r.Server = st
	}
	return r
}

func pcts(x []float64) map[string]float64 {
	if len(x) == 0 {
		return nil
	}
	sort.Float64s(x)
	at := func(q float64) float64 { return x[min(len(x)-1, int(q*float64(len(x))))] }
	return map[string]float64{"p50": at(0.5), "p95": at(0.95), "p99": at(0.99), "max": x[len(x)-1]}
}

func procRSS(pid int) int64 {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if v, ok := strings.CutPrefix(line, "VmRSS:"); ok {
			kb, _ := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(v), " kB"), 10, 64)
			return kb << 10
		}
	}
	return 0
}

func (g *gen) do(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, g.base+path, rd)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if g.key != "" {
		req.Header.Set("Authorization", "Bearer "+g.key)
	}
	return g.http.Do(req)
}

func (g *gen) stats() (*serverStats, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	resp, err := g.do(ctx, http.MethodGet, "/api/v1/stats", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("/api/v1/stats: %s", resp.Status)
	}
	var raw struct {
		Process struct {
			Goroutines int    `json:"goroutines"`
			RSS        int64  `json:"rss"`
			RSSPeak    int64  `json:"rss_peak"`
			Locked     int64  `json:"locked"`
			GoHeap     uint64 `json:"go_heap"`
		} `json:"process"`
		Metrics struct {
			CacheHitRate float64 `json:"cache_hit_rate"`
		} `json:"metrics"`
		Instances []struct {
			Loop struct{ Slots, Busy, Pending int } `json:"loop"`
		} `json:"instances"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}
	st := &serverStats{Goroutines: raw.Process.Goroutines, RSS: raw.Process.RSS, RSSPeak: raw.Process.RSSPeak,
		Locked: raw.Process.Locked, GoHeap: raw.Process.GoHeap, HitRate: raw.Metrics.CacheHitRate}
	for _, in := range raw.Instances {
		st.Slots += in.Loop.Slots
		st.Busy += in.Loop.Busy
		st.Pending += in.Loop.Pending
	}
	return st, nil
}

// --- the mix -------------------------------------------------------------------

var topics = []string{
	"how a solar panel works", "why the sky is blue", "the history of the printing press", "how to brew good tea",
	"what a prime number is", "how a bicycle stays upright", "why bees dance", "what causes the tides",
	"как работает холодильник", "почему идёт дождь", "зачем нужен сон", "как устроен компас",
	"что такое электричество", "как делают хлеб", "почему зимой холодно", "как растут деревья",
}

var followups = []string{"Explain it more simply.", "Give one example.", "Summarize in one sentence.", "Объясни проще.", "Приведи пример.", "Сократи до одного предложения."}

// kinds are the request kinds -only picks from.
var kinds = []string{"chat", "abandon", "tool", "json", "long", "job"}

// claim says whether this worker may send another request: the -n budget is
// shared by all workers. A worker that finds it spent leaves its loop; the
// requests the others still have in flight run to their end.
func (g *gen) claim() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.limit > 0 && g.sent >= g.limit {
		return false
	}
	g.sent++
	return true
}

// pick is the kind of the next request: the one -only fixes, else the mix.
func (g *gen) pick(rng *rand.Rand) string {
	if g.only != "" {
		return g.only
	}
	r := rng.Float64()
	if g.hooks != nil { // queue jobs take six percent
		switch {
		case r < 0.50:
			return "chat"
		case r < 0.65:
			return "abandon"
		case r < 0.74:
			return "tool"
		case r < 0.83:
			return "json"
		case r < 0.89:
			return "job"
		}
		return "long"
	}
	switch {
	case r < 0.55:
		return "chat"
	case r < 0.70:
		return "abandon"
	case r < 0.80:
		return "tool"
	case r < 0.90:
		return "json"
	}
	return "long"
}

func (g *gen) worker(ctx context.Context, rng *rand.Rand, id int) {
	for n := 0; ctx.Err() == nil && g.claim(); n++ {
		switch g.pick(rng) {
		case "chat":
			g.conversation(ctx, rng, fmt.Sprintf("lg-%d-%d", id, n))
		case "abandon":
			g.abandoned(ctx, rng)
		case "tool":
			g.toolCall(ctx, rng)
		case "json":
			g.jsonSchema(ctx, rng)
		case "job":
			g.job(ctx, rng)
		default:
			g.longPrompt(ctx, rng)
		}
	}
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (g *gen) body(msgs []chatMsg, session string, stream bool) map[string]any {
	b := map[string]any{"messages": msgs, "max_tokens": g.maxTokens, "temperature": 0.7, "stream": stream}
	if g.model != "" {
		b["model"] = g.model
	}
	maps.Copy(b, g.extra)
	if session != "" {
		b["session_id"] = session
	}
	if stream {
		b["stream_options"] = map[string]any{"include_usage": true}
	}
	return b
}

// stream sends a streaming chat and reads it to the end (or until stopAfter
// content chunks, when positive). It returns the answer.
func (g *gen) stream(ctx context.Context, kind string, body map[string]any, stopAfter int) (string, bool) {
	t0 := time.Now()
	resp, err := g.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		if ctx.Err() == nil {
			g.add(sample{kind: kind, outcome: "error"})
		}
		return "", false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		g.add(sample{kind: kind, outcome: "http_" + strconv.Itoa(resp.StatusCode)})
		return "", false
	}
	var answer strings.Builder
	var ttft float64
	var usage struct{ prompt, cached int }
	chunks := 0
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var ev struct {
			Choices []struct{ Delta struct{ Content string } } `json:"choices"`
			Usage   *struct {
				PromptTokens int `json:"prompt_tokens"`
				Details      struct {
					Cached int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		if len(ev.Choices) > 0 && ev.Choices[0].Delta.Content != "" {
			if ttft == 0 {
				ttft = float64(time.Since(t0).Microseconds()) / 1000
			}
			answer.WriteString(ev.Choices[0].Delta.Content)
			chunks++
			if stopAfter > 0 && chunks >= stopAfter {
				g.add(sample{kind: kind, outcome: "cancelled", ttft: ttft, total: float64(time.Since(t0).Microseconds()) / 1000})
				return answer.String(), false // the deferred Close abandons the stream
			}
		}
		if ev.Usage != nil {
			usage.prompt, usage.cached = ev.Usage.PromptTokens, ev.Usage.Details.Cached
		}
	}
	if err := sc.Err(); err != nil && ctx.Err() == nil {
		g.add(sample{kind: kind, outcome: "error"})
		return answer.String(), false
	}
	g.add(sample{kind: kind, outcome: "ok", ttft: ttft, total: float64(time.Since(t0).Microseconds()) / 1000, cached: usage.cached, prompt: usage.prompt})
	return answer.String(), true
}

func (g *gen) conversation(ctx context.Context, rng *rand.Rand, session string) {
	msgs := []chatMsg{{"system", "You are a concise assistant. Answer in the language of the question."},
		{"user", "Tell me " + topics[rng.IntN(len(topics))] + "."}}
	for turn, turns := 0, 3+rng.IntN(4); turn < turns && ctx.Err() == nil; turn++ {
		ans, ok := g.stream(ctx, "chat", g.body(msgs, session, true), 0)
		if !ok {
			return
		}
		msgs = append(msgs, chatMsg{"assistant", ans}, chatMsg{"user", followups[rng.IntN(len(followups))]})
	}
}

func (g *gen) abandoned(ctx context.Context, rng *rand.Rand) {
	msgs := []chatMsg{{"user", "Write a long story about " + topics[rng.IntN(len(topics))] + "."}}
	b := g.body(msgs, "", true)
	b["max_tokens"] = 400
	g.stream(ctx, "abandon", b, 3+rng.IntN(20))
}

func (g *gen) plain(ctx context.Context, kind string, body map[string]any, check func([]byte) bool) {
	t0 := time.Now()
	resp, err := g.do(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		if ctx.Err() == nil {
			g.add(sample{kind: kind, outcome: "error"})
		}
		return
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	switch {
	case resp.StatusCode != http.StatusOK:
		g.add(sample{kind: kind, outcome: "http_" + strconv.Itoa(resp.StatusCode)})
	case check != nil && !check(data):
		g.add(sample{kind: kind, outcome: "bad_output"})
		g.record(kind, body, data)
	default:
		g.add(sample{kind: kind, outcome: "ok", total: float64(time.Since(t0).Microseconds()) / 1000})
	}
}

// answer is what the checks look at in a chat completion.
type answer struct {
	finish    string
	content   string
	reasoning bool
	toolCalls int
}

// answerOf reads the first choice of a chat completion; an unreadable body is
// the zero answer.
func answerOf(d []byte) answer {
	var r struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
				ToolCalls        []any  `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(d, &r) != nil || len(r.Choices) == 0 {
		return answer{}
	}
	c := r.Choices[0]
	return answer{c.FinishReason, c.Message.Content, c.Message.ReasoningContent != "", len(c.Message.ToolCalls)}
}

// why names the cause of a rejected answer for -bad-out, so that the file can
// be counted instead of read.
func (a answer) why() string {
	switch {
	case a.finish == "":
		return "unreadable"
	case a.finish == "length" && a.reasoning && strings.TrimSpace(a.content) == "":
		return "budget spent on reasoning"
	case a.finish == "length":
		return "budget spent on the answer"
	case strings.TrimSpace(a.content) != "":
		return "text instead of a call"
	}
	return "empty"
}

// record keeps a rejected answer with the request that produced it, to be
// sorted by cause afterwards (-bad-out).
func (g *gen) record(kind string, body map[string]any, resp []byte) {
	if g.bad == nil {
		return
	}
	var parsed any
	if json.Unmarshal(resp, &parsed) != nil {
		parsed = string(resp)
	}
	line, _ := json.Marshal(map[string]any{"kind": kind, "why": answerOf(resp).why(), "request": body, "response": parsed})
	g.mu.Lock()
	defer g.mu.Unlock()
	fmt.Fprintln(g.bad, string(line))
}

func (g *gen) toolCall(ctx context.Context, rng *rand.Rand) {
	b := g.body([]chatMsg{{"user", "What is the weather in " + []string{"Paris", "Moscow", "Tokyo"}[rng.IntN(3)] + "? Use the tool."}}, "", false)
	b["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{
		"name": "get_weather", "description": "Current weather in a city",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []string{"city"}},
	}}}
	b["max_tokens"] = max(128, g.maxTokens) // -max-tokens may raise the floor, never lower it
	g.plain(ctx, "tool", b, func(d []byte) bool { return answerOf(d).toolCalls > 0 })
}

func (g *gen) jsonSchema(ctx context.Context, rng *rand.Rand) {
	b := g.body([]chatMsg{{"user", "Give a person's name and age as JSON. Topic: " + topics[rng.IntN(len(topics))]}}, "", false)
	b["response_format"] = map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "p", "schema": map[string]any{
		"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string"}, "age": map[string]any{"type": "integer"}},
		"required": []string{"name", "age"}, "additionalProperties": false}}}
	b["max_tokens"] = 64
	g.plain(ctx, "json", b, func(d []byte) bool {
		var r struct {
			Choices []struct{ Message struct{ Content string } } `json:"choices"`
		}
		if json.Unmarshal(d, &r) != nil || len(r.Choices) == 0 {
			return false
		}
		var v struct {
			Name string `json:"name"`
			Age  *int   `json:"age"`
		}
		return json.Unmarshal([]byte(r.Choices[0].Message.Content), &v) == nil && v.Name != "" && v.Age != nil
	})
}

func (g *gen) longPrompt(ctx context.Context, rng *rand.Rand) {
	var sb strings.Builder
	for sb.Len() < 9000 {
		sb.WriteString("The old library smelled of paper and dust, and the clock on the wall was five minutes slow. ")
	}
	fmt.Fprintf(&sb, "\n\nThe code word is %d. What is the code word? Answer with the number.", 1000+rng.IntN(9000))
	b := g.body([]chatMsg{{"user", sb.String()}}, "", false)
	b["max_tokens"] = 16
	g.plain(ctx, "long", b, nil)
}

// --- analysis -----------------------------------------------------------------

func analyzeFile(path string, w io.Writer) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var reps []Report
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var r Report
		if json.Unmarshal(sc.Bytes(), &r) == nil && r.T != "" {
			reps = append(reps, r)
		}
	}
	if len(reps) < 4 {
		return errors.New("fewer than four reports; run longer")
	}
	last := reps[len(reps)-1]
	fmt.Fprintf(w, "%d reports over %.1f h\n", len(reps), last.ElapsedS/3600)

	// Memory and goroutines after a warm-up of the first quarter.
	from := len(reps) / 4
	var xs, rss, gor []float64
	for _, r := range reps[from:] {
		rssv := float64(r.RSS)
		if r.Server != nil && rssv == 0 {
			rssv = float64(r.Server.RSS)
		}
		xs = append(xs, r.ElapsedS/3600)
		rss = append(rss, rssv/(1<<20))
		if r.Server != nil {
			gor = append(gor, float64(r.Server.Goroutines))
		}
	}
	slope, _ := fit(xs, rss)
	fmt.Fprintf(w, "RSS after warm-up: %.0f MiB at the start, %.0f at the end, trend %+.1f MiB/h\n", rss[0], rss[len(rss)-1], slope)
	if len(gor) == len(xs) {
		gs, _ := fit(xs, gor)
		fmt.Fprintf(w, "goroutines: %.0f → %.0f, trend %+.2f/h\n", gor[0], gor[len(gor)-1], gs)
	}

	// Errors and latency, first third against the last.
	totals := map[string]int{}
	for k, v := range last.Totals {
		totals[k] = v
	}
	var bad []string
	for k, v := range totals {
		if !strings.HasSuffix(k, "/ok") && !strings.HasSuffix(k, "/cancelled") {
			bad = append(bad, fmt.Sprintf("%s=%d", k, v))
		}
	}
	sort.Strings(bad)
	fmt.Fprintf(w, "requests: %v\nnot ok or cancelled: %s\n", totals, orNone(bad))
	if last.QueueDB > 0 {
		var dx, dy []float64
		for _, r := range reps[from:] {
			dx = append(dx, r.ElapsedS/3600)
			dy = append(dy, float64(r.QueueDB)/(1<<20))
		}
		ds, _ := fit(dx, dy)
		fmt.Fprintf(w, "queue database after warm-up: %.2f MiB at the start, %.2f at the end, trend %+.3f MiB/h\n", dy[0], dy[len(dy)-1], ds)
	}
	if last.Hooks != nil {
		fmt.Fprintf(w, "webhooks: %d verified, %d refused for a bad signature\n", last.Hooks.Received, last.Hooks.BadSig)
	}
	third := len(reps) / 3
	p95 := func(rs []Report, key string) float64 {
		var v []float64
		for _, r := range rs {
			if x, ok := r.TTFT[key]; ok {
				v = append(v, x)
			}
		}
		sort.Float64s(v)
		if len(v) == 0 {
			return math.NaN()
		}
		return v[len(v)/2]
	}
	fmt.Fprintf(w, "TTFT p95 (median of windows): first third %.0f ms, last third %.0f ms\n", p95(reps[:third], "p95"), p95(reps[len(reps)-third:], "p95"))
	return nil
}

func orNone(s []string) string {
	if len(s) == 0 {
		return "none"
	}
	return strings.Join(s, " ")
}

// fit is the least-squares line y = a·x + b.
func fit(x, y []float64) (a, b float64) {
	n := float64(len(x))
	var sx, sy, sxx, sxy float64
	for i := range x {
		sx += x[i]
		sy += y[i]
		sxx += x[i] * x[i]
		sxy += x[i] * y[i]
	}
	d := n*sxx - sx*sx
	if d == 0 {
		return 0, sy / n
	}
	a = (n*sxy - sx*sy) / d
	return a, (sy - a*sx) / n
}
