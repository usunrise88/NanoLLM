package queue

import (
	"io"
	"net/http"
	"strconv"

	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
)

// MaxBatchBytes bounds one submitted JSONL body.
const MaxBatchBytes = 64 << 20

// MountNative adds the /api/v1/jobs routes; it implements native.Mounter.
func (q *Queue) MountNative(mux *http.ServeMux) {
	mux.HandleFunc("POST /api/v1/jobs", q.handleSubmit)
	mux.HandleFunc("GET /api/v1/jobs", q.handleList)
	mux.HandleFunc("GET /api/v1/jobs/{id}", q.handleGet)
	mux.HandleFunc("GET /api/v1/jobs/{id}/events", q.handleEvents)
	mux.HandleFunc("DELETE /api/v1/jobs/{id}", q.handleCancel)
}

// owner is the key a job belongs to: "" in open mode and for admin keys,
// which see every job.
func owner(r *http.Request) (name string, all bool) {
	k, ok := httpx.KeyFrom(r.Context())
	if !ok {
		return "", true
	}
	return k.Name, k.Admin
}

// visible loads a job the caller may see; another key's job is reported as
// missing rather than forbidden, so ids cannot be probed.
func (q *Queue) visible(w http.ResponseWriter, r *http.Request) (*Job, bool) {
	j, err := q.Get(r.Context(), r.PathValue("id"))
	if err != nil {
		httpx.WriteProblem(w, err)
		return nil, false
	}
	if name, all := owner(r); !all && j.Key != name {
		httpx.WriteProblem(w, core.Errorf(core.CodeJobNotFound, "no job %q", j.ID))
		return nil, false
	}
	return j, true
}

func (q *Queue) handleSubmit(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, MaxBatchBytes))
	if err != nil {
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "read the batch: %v", err))
		return
	}
	name, _ := owner(r)
	j, err := q.Submit(r.Context(), body, r.URL.Query().Get("model"), r.URL.Query().Get("webhook_url"), name)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	w.Header().Set("Location", "/api/v1/jobs/"+j.ID)
	httpx.WriteJSON(w, http.StatusAccepted, j)
}

func (q *Queue) handleList(w http.ResponseWriter, r *http.Request) {
	v := r.URL.Query()
	limit := 0
	if s := v.Get("limit"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "limit must be a positive integer").WithParam("limit"))
			return
		}
		limit = n
	}
	status := v.Get("status")
	switch status {
	case "", StatusQueued, StatusRunning, StatusSucceeded, StatusFailed, StatusCanceled:
	default:
		httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "unknown status %q", status).WithParam("status"))
		return
	}
	name, all := owner(r)
	if all {
		name = ""
	}
	jobs, next, err := q.List(r.Context(), status, v.Get("cursor"), name, limit)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	if jobs == nil {
		jobs = []*Job{}
	}
	page := map[string]any{"data": jobs}
	if next != "" {
		page["next_cursor"] = next
	}
	httpx.WriteJSON(w, http.StatusOK, page)
}

func (q *Queue) handleGet(w http.ResponseWriter, r *http.Request) {
	if j, ok := q.visible(w, r); ok {
		httpx.WriteJSON(w, http.StatusOK, j)
	}
}

func (q *Queue) handleCancel(w http.ResponseWriter, r *http.Request) {
	j, ok := q.visible(w, r)
	if !ok {
		return
	}
	j, err := q.Cancel(r.Context(), j.ID)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, j)
}

// handleEvents sends the job as it is now, then on every change, and ends
// with the terminal state.
func (q *Queue) handleEvents(w http.ResponseWriter, r *http.Request) {
	j, ok := q.visible(w, r)
	if !ok {
		return
	}
	updates, stop := q.Subscribe(j.ID)
	defer stop()
	st, err := httpx.NewStream(w)
	if err != nil {
		httpx.WriteProblem(w, err)
		return
	}
	defer st.Close()
	// Re-read after subscribing so an update between the first read and the
	// subscription is not lost.
	if cur, err := q.Get(r.Context(), j.ID); err == nil {
		j = cur
	}
	if st.Send("", "progress", j) != nil || j.terminal() {
		return
	}
	for {
		select {
		case <-r.Context().Done():
			return
		case u := <-updates:
			if st.Send("", "progress", u) != nil || u.terminal() {
				return
			}
		case <-st.Heartbeat():
			if st.Comment("keep-alive") != nil {
				return
			}
		}
	}
}
