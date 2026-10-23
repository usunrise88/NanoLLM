package server

import (
	"net/http"
	"sync"

	"github.com/usunrise88/nanollm/internal/httpx"
)

// eventLog is the journal of one long run (a benchmark, an optimization): the
// events so far, kept for a client that connects late, and a channel that wakes
// the ones following it. An event named done or error ends the run.
type eventLog struct {
	mu     sync.Mutex
	events []benchEvent
	notify chan struct{}
	done   bool
}

type benchEvent struct {
	name    string
	payload any
}

func newEventLog() *eventLog { return &eventLog{notify: make(chan struct{})} }

func (l *eventLog) push(name string, payload any) {
	l.mu.Lock()
	l.events = append(l.events, benchEvent{name, payload})
	if name == "done" || name == "error" {
		l.done = true
	}
	old := l.notify
	l.notify = make(chan struct{})
	l.mu.Unlock()
	close(old)
}

// stream replays everything so far, then follows the run until it ends or the
// client goes away.
func (l *eventLog) stream(w http.ResponseWriter, req *http.Request) {
	st, err := httpx.NewStream(w)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	defer st.Close()
	sent := 0
	for {
		l.mu.Lock()
		pending := l.events[sent:]
		done, wait := l.done, l.notify
		l.mu.Unlock()
		for _, e := range pending {
			if st.Send("", e.name, e.payload) != nil {
				return
			}
			sent++
		}
		if done {
			return
		}
		select {
		case <-wait:
		case <-st.Heartbeat():
			if st.Comment("keep-alive") != nil {
				return
			}
		case <-req.Context().Done():
			return
		}
	}
}

// runs keeps the last few runs of one kind, so that a client can ask for the
// result of one that has ended.
type runs[T any] struct {
	mu    sync.Mutex
	byID  map[string]T
	order []string
}

// maxRuns is how many finished runs are kept.
const maxRuns = 8

func newRuns[T any]() *runs[T] { return &runs[T]{byID: map[string]T{}} }

func (r *runs[T]) add(id string, v T) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.byID[id] = v
	r.order = append(r.order, id)
	for len(r.order) > maxRuns {
		delete(r.byID, r.order[0])
		r.order = r.order[1:]
	}
}

func (r *runs[T]) get(id string) (T, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	v, ok := r.byID[id]
	return v, ok
}
