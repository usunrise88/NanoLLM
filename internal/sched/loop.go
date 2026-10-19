package sched

import (
	"context"
	"errors"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
)

// Request is one generation on a loaded model: a prompt already tokenized and
// a sampler already built. The loop knows nothing of templates or dialects.
type Request struct {
	Prompt    []backend.Token
	Sampler   backend.SamplerSpec
	MaxTokens int
	// Logprobs is the number of alternatives per token: -1 for none.
	Logprobs int
	// SessionID keeps a conversation on the slot that holds its cache.
	SessionID string
	// NoCache opts out of prefix reuse: the slot is cleared first.
	NoCache bool
	// Interactive requests go first when the batch is full.
	Interactive bool
	// NoDraft turns MTP drafting off, for a constrained (grammar) request
	// where a rejected draft would have advanced the grammar.
	NoDraft bool
}

// Finish reasons.
const (
	FinishEOG    = "eog"
	FinishLength = "length"
)

// Event is what a request receives: tokens as they are accepted, then one
// final event with the outcome.
type Event struct {
	Tokens   []backend.Token
	Logprobs []backend.TokenLogprobs
	// Done is set on the last event; Err or Finish says why.
	Done   bool
	Finish string
	Err    error
	Stats  *Stats
}

// Stats is a finished request's account of its time in the loop.
type Stats struct {
	Slot         int
	PromptTokens int
	CachedTokens int
	Completion   int
	Drafted      int
	Accepted     int
	Queued       time.Duration // submit to slot assignment
	Prefill      time.Duration // assignment to first token
	Decode       time.Duration // first token to the end
}

// ErrClosed is returned for requests on a loop that has stopped.
var ErrClosed = errors.New("sched: model is unloading")

// ErrOverloaded is returned when the requests waiting for a slot are already
// as many as the loop was told to hold (SetMaxPending): an unbounded queue
// only turns an overload into a growing wait for everyone.
var ErrOverloaded = errors.New("sched: too many requests are waiting for a slot")

// Loop is the scheduling loop of one loaded model. It is the only goroutine
// that calls the model's Step and changes slot state: requests join between
// steps, and every step batches all active slots — prefill chunks and decode
// steps together — up to the model's batch size.
type Loop struct {
	model backend.Model
	caps  backend.Capabilities

	stop chan struct{}
	done chan struct{}

	mu      sync.Mutex
	pending []*Handle
	closed  bool
	wake    chan struct{}
	// maxPending bounds the requests waiting for a slot; zero is no bound.
	maxPending int

	slots []slot
	clock uint64

	// Read from other goroutines (Snapshot), written by the loop.
	busy       atomic.Int32
	steps      atomic.Int64
	stepTokens atomic.Int64
}

// LoopStats is the loop's state at an instant, for the operator.
type LoopStats struct {
	Slots   int `json:"slots"`
	Busy    int `json:"busy"`    // slots running a request
	Pending int `json:"pending"` // requests waiting for a slot
	// Steps is the model steps taken and StepTokens the tokens in their
	// batches; their ratio is how full the batches run.
	Steps      int64 `json:"steps"`
	StepTokens int64 `json:"step_tokens"`
}

// Snapshot reads the loop's counters; it is safe from any goroutine.
func (l *Loop) Snapshot() LoopStats {
	l.mu.Lock()
	pending := len(l.pending)
	l.mu.Unlock()
	return LoopStats{
		Slots: len(l.slots), Busy: int(l.busy.Load()), Pending: pending,
		Steps: l.steps.Load(), StepTokens: l.stepTokens.Load(),
	}
}

type slot struct {
	// cached mirrors the KV cache: the tokens evaluated in this slot.
	cached  []backend.Token
	session string
	lastUse uint64
	h       *Handle // nil when free
}

// New starts the loop for a model. Close stops it.
func New(m backend.Model) *Loop {
	caps := m.Capabilities()
	if caps.Slots <= 0 {
		caps.Slots = 1
	}
	if caps.NBatch <= 0 {
		caps.NBatch = 512
	}
	l := &Loop{
		model: m,
		caps:  caps,
		stop:  make(chan struct{}),
		done:  make(chan struct{}),
		wake:  make(chan struct{}, 1),
		slots: make([]slot, caps.Slots),
	}
	go l.run()
	return l
}

// SetMaxPending bounds the requests that may wait for a slot (not those
// running in one); a Submit past it fails with ErrOverloaded. Zero removes
// the bound.
func (l *Loop) SetMaxPending(n int) {
	l.mu.Lock()
	l.maxPending = n
	l.mu.Unlock()
}

// Handle is one submitted request.
type Handle struct {
	ctx context.Context
	req Request

	mu     sync.Mutex
	events []Event
	notify chan struct{}
	ended  bool

	// Owned by the loop.
	seq       int
	pos       int // tokens of the prompt evaluated so far, reused prefix included
	next      backend.Token
	hasNext   bool
	generated int
	stats     Stats
	submitted time.Time
	assigned  time.Time
	firstTok  time.Time
}

// Submit queues a request. Cancelling ctx ends it at the next step boundary.
func (l *Loop) Submit(ctx context.Context, req Request) (*Handle, error) {
	if len(req.Prompt) == 0 {
		return nil, errors.New("sched: empty prompt")
	}
	if len(req.Prompt) >= l.caps.NCtx {
		return nil, ContextError{Prompt: len(req.Prompt), NCtx: l.caps.NCtx}
	}
	h := &Handle{ctx: ctx, req: req, notify: make(chan struct{}, 1), submitted: time.Now(), seq: -1}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil, ErrClosed
	}
	if l.maxPending > 0 && len(l.pending) >= l.maxPending {
		return nil, ErrOverloaded
	}
	l.pending = append(l.pending, h)
	select {
	case l.wake <- struct{}{}:
	default:
	}
	return h, nil
}

// ContextError is a prompt that does not fit a slot.
type ContextError struct{ Prompt, NCtx int }

func (e ContextError) Error() string {
	return "sched: prompt of " + strconv.Itoa(e.Prompt) + " tokens does not fit a context of " + strconv.Itoa(e.NCtx)
}

// Next blocks for the next event. After the Done event it returns it again.
func (h *Handle) Next(ctx context.Context) (Event, error) {
	for {
		h.mu.Lock()
		if len(h.events) > 0 {
			ev := h.events[0]
			if !ev.Done || len(h.events) > 1 {
				h.events = h.events[1:]
			}
			h.mu.Unlock()
			return ev, nil
		}
		h.mu.Unlock()
		select {
		case <-h.notify:
		case <-ctx.Done():
			return Event{}, ctx.Err()
		}
	}
}

// push queues an event without ever blocking the loop on a slow reader.
func (h *Handle) push(ev Event) {
	h.mu.Lock()
	if h.ended {
		h.mu.Unlock()
		return
	}
	if ev.Done {
		h.ended = true
	}
	// Consecutive token events merge, so a reader that falls behind sees
	// fewer, larger events rather than a growing queue.
	if n := len(h.events); n > 0 && !ev.Done && !h.events[n-1].Done {
		last := &h.events[n-1]
		last.Tokens = append(last.Tokens, ev.Tokens...)
		last.Logprobs = append(last.Logprobs, ev.Logprobs...)
	} else {
		h.events = append(h.events, ev)
	}
	h.mu.Unlock()
	select {
	case h.notify <- struct{}{}:
	default:
	}
}

// Close stops the loop. Running and queued requests end with ErrClosed; the
// model itself is the caller's to close afterwards.
func (l *Loop) Close() {
	l.mu.Lock()
	if !l.closed {
		l.closed = true
		close(l.stop)
	}
	l.mu.Unlock()
	<-l.done
}

// Busy reports the number of requests running or queued.
func (l *Loop) Busy() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := len(l.pending)
	for _, s := range l.slots {
		if s.h != nil {
			n++
		}
	}
	return n
}

func (l *Loop) run() {
	defer close(l.done)
	for {
		l.admit()
		active := l.active()
		if len(active) == 0 {
			select {
			case <-l.wake:
				continue
			case <-l.stop:
				l.shutdown()
				return
			}
		}
		select {
		case <-l.stop:
			l.shutdown()
			return
		default:
		}
		l.step(active)
	}
}

func (l *Loop) shutdown() {
	l.mu.Lock()
	pending := l.pending
	l.pending = nil
	l.mu.Unlock()
	for _, h := range pending {
		h.push(Event{Done: true, Err: ErrClosed})
	}
	for i := range l.slots {
		if h := l.slots[i].h; h != nil {
			l.finish(i, Event{Err: ErrClosed})
		}
	}
}

// admit moves queued requests into free slots, interactive ones first.
func (l *Loop) admit() {
	l.mu.Lock()
	pending := l.pending
	l.pending = nil
	l.mu.Unlock()
	sort.SliceStable(pending, func(i, j int) bool {
		return pending[i].req.Interactive && !pending[j].req.Interactive
	})
	var waiting []*Handle
	for _, h := range pending {
		if err := h.ctx.Err(); err != nil {
			h.push(Event{Done: true, Err: err})
			continue
		}
		seq := l.pickSlot(h.req)
		if seq < 0 {
			waiting = append(waiting, h)
			continue
		}
		l.assign(seq, h)
	}
	if len(waiting) > 0 {
		l.mu.Lock()
		l.pending = append(waiting, l.pending...)
		l.mu.Unlock()
	}
}

// pickSlot chooses a free slot: the one holding the request's session, else
// the one sharing the longest prefix with its prompt, else the least recently
// used.
func (l *Loop) pickSlot(req Request) int {
	best, bestLCP, bestUse := -1, -1, uint64(0)
	for i := range l.slots {
		s := &l.slots[i]
		if s.h != nil {
			continue
		}
		if req.SessionID != "" && s.session == req.SessionID {
			return i
		}
		n := 0
		if !req.NoCache {
			n = lcp(s.cached, req.Prompt)
		}
		if n > bestLCP || (n == bestLCP && s.lastUse < bestUse) {
			best, bestLCP, bestUse = i, n, s.lastUse
		}
	}
	return best
}

func (l *Loop) assign(seq int, h *Handle) {
	s := &l.slots[seq]
	h.seq = seq
	h.assigned = time.Now()
	reuse := 0
	if !h.req.NoCache {
		reuse = lcp(s.cached, h.req.Prompt)
	}
	// At least one prompt token is evaluated, so there are logits to sample.
	if reuse >= len(h.req.Prompt) {
		reuse = len(h.req.Prompt) - 1
	}
	if reuse < len(s.cached) {
		if !l.model.TrimSeq(seq, reuse) {
			// The cache cannot go back that far (sliding-window layers
			// without swa_full): start the slot over.
			l.model.ClearSeq(seq)
			reuse = 0
		}
	}
	s.cached = s.cached[:reuse]
	s.h = h
	l.busy.Add(1)
	s.session = h.req.SessionID
	h.pos = reuse
	h.stats = Stats{Slot: seq, PromptTokens: len(h.req.Prompt), CachedTokens: reuse}
	if err := l.model.SetSampler(seq, h.req.Sampler, h.req.Prompt); err != nil {
		l.finish(seq, Event{Err: err})
	}
}

func (l *Loop) active() []int {
	var out []int
	for i := range l.slots {
		if l.slots[i].h != nil {
			out = append(out, i)
		}
	}
	return out
}

// step builds one batch: decode steps first (interactive before batch), each
// with as many draft tokens as the budget allows, then prefill chunks in the
// order slots were assigned.
func (l *Loop) step(active []int) {
	budget := l.caps.NBatch
	var steps []backend.SeqStep
	var decode, prefill []int
	for _, seq := range active {
		h := l.slots[seq].h
		if err := h.ctx.Err(); err != nil {
			// Cancelled between steps: the cache is exactly what the slot
			// mirrors, so it stays. A client that goes away, a stop sequence
			// and a reasoning budget all end here, and the next request —
			// the continuation, the next turn — reuses the prefix.
			l.release(seq, Event{Err: err}, false)
			continue
		}
		if h.hasNext {
			if len(l.slots[seq].cached)+1 >= l.caps.NCtx {
				// The slot's context is full: the answer ends here.
				l.finish(seq, Event{Finish: FinishLength})
				continue
			}
			decode = append(decode, seq)
		} else {
			prefill = append(prefill, seq)
		}
	}
	sort.SliceStable(decode, func(i, j int) bool {
		return l.slots[decode[i]].h.req.Interactive && !l.slots[decode[j]].h.req.Interactive
	})
	sort.SliceStable(prefill, func(i, j int) bool {
		return l.slots[prefill[i]].h.assigned.Before(l.slots[prefill[j]].h.assigned)
	})

	// Every decoding slot gets its one token before any gets drafts, so a
	// full batch slows drafting down rather than starving a slot.
	var chosen []int
	for _, seq := range decode {
		if budget < 1 {
			break
		}
		budget--
		chosen = append(chosen, seq)
	}
	for _, seq := range chosen {
		h := l.slots[seq].h
		draft := 0
		if !h.req.NoDraft && l.caps.MaxDraft > 0 {
			left := l.caps.NCtx
			if h.req.MaxTokens > 0 {
				left = h.req.MaxTokens - h.generated - 1
			}
			draft = min(l.caps.MaxDraft, budget, max(left, 0), l.caps.NCtx-len(l.slots[seq].cached)-2)
			draft = max(draft, 0)
			budget -= draft
		}
		steps = append(steps, backend.SeqStep{
			Seq: seq, Tokens: []backend.Token{h.next}, Pos: len(l.slots[seq].cached),
			Sample: true, MaxDraft: draft, Logprobs: h.req.Logprobs,
		})
	}
	for _, seq := range prefill {
		if budget < 1 {
			break
		}
		h := l.slots[seq].h
		rest := h.req.Prompt[h.pos:]
		n := min(len(rest), budget)
		budget -= n
		steps = append(steps, backend.SeqStep{
			Seq: seq, Tokens: rest[:n], Pos: len(l.slots[seq].cached),
			Sample: n == len(rest), Logprobs: h.req.Logprobs,
		})
	}
	if len(steps) == 0 {
		return
	}

	// The step's own context: cancelling it would abandon every slot in the
	// batch, so it is never tied to one request.
	l.steps.Add(1)
	for _, st := range steps {
		l.stepTokens.Add(int64(len(st.Tokens)))
	}
	outs, err := l.model.Step(context.Background(), steps)
	if err != nil {
		for _, st := range steps {
			l.finish(st.Seq, Event{Err: err})
		}
		return
	}
	byseq := map[int]backend.SeqOut{}
	for _, o := range outs {
		byseq[o.Seq] = o
	}
	l.clock++
	for _, st := range steps {
		s := &l.slots[st.Seq]
		h := s.h
		s.cached = append(s.cached, st.Tokens...)
		s.lastUse = l.clock
		if !st.Sample {
			h.pos += len(st.Tokens)
			continue
		}
		h.pos = len(h.req.Prompt)
		out, ok := byseq[st.Seq]
		if !ok || len(out.Tokens) == 0 {
			l.finish(st.Seq, Event{Err: errors.New("sched: backend returned no tokens for a sampling slot")})
			continue
		}
		l.deliver(st.Seq, out)
	}
}

func (l *Loop) deliver(seq int, out backend.SeqOut) {
	s := &l.slots[seq]
	h := s.h
	if h.firstTok.IsZero() {
		h.firstTok = time.Now()
	}
	h.stats.Drafted += out.Drafted
	h.stats.Accepted += out.Accepted

	// All but the last returned token are in the cache now.
	s.cached = append(s.cached, out.Tokens[:len(out.Tokens)-1]...)

	toks, lps := out.Tokens, out.Logprobs
	finish := ""
	if out.EOG {
		// The end-of-generation token ends the answer but is not part of it.
		toks = toks[:len(toks)-1]
		lps = lps[:min(len(lps), len(toks))]
		finish = FinishEOG
	}
	if left := h.req.MaxTokens - h.generated; h.req.MaxTokens > 0 && len(toks) >= left {
		if len(toks) > left || finish == "" {
			finish = FinishLength
		}
		toks = toks[:left]
		lps = lps[:min(len(lps), left)]
	}
	h.generated += len(toks)
	if len(toks) > 0 {
		h.push(Event{Tokens: toks, Logprobs: lps})
	}
	if finish != "" {
		l.finish(seq, Event{Finish: finish})
		return
	}
	h.next, h.hasNext = out.Tokens[len(out.Tokens)-1], true
}

// finish ends the request in a slot and frees it. The slot keeps its cache
// for the next request that shares the prefix, unless the request failed: an
// error may have struck mid-step, and the cache can then differ from what the
// slot mirrors.
func (l *Loop) finish(seq int, ev Event) {
	l.release(seq, ev, ev.Err != nil)
}

// release is finish with the cache decision made by the caller.
func (l *Loop) release(seq int, ev Event, clear bool) {
	s := &l.slots[seq]
	h := s.h
	if h == nil {
		return
	}
	s.h = nil
	l.busy.Add(-1)
	now := time.Now()
	st := h.stats
	st.Completion = h.generated
	st.Queued = h.assigned.Sub(h.submitted)
	if !h.firstTok.IsZero() {
		st.Prefill = h.firstTok.Sub(h.assigned)
		st.Decode = now.Sub(h.firstTok)
	} else if !h.assigned.IsZero() {
		st.Prefill = now.Sub(h.assigned)
	}
	ev.Done = true
	ev.Stats = &st
	if clear {
		l.model.ClearSeq(seq)
		s.cached = nil
		s.session = ""
	}
	h.push(ev)
}

func lcp(a, b []backend.Token) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
