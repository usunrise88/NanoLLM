package server

import (
	"net/http"

	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/pool"
	"github.com/usunrise88/nanollm/internal/stats"
)

// StatsReport is GET /api/v1/stats: the process, the requests served and the
// resident models with their scheduling loops.
type StatsReport struct {
	Process   stats.Process  `json:"process"`
	Metrics   stats.Snapshot `json:"metrics"`
	Instances []pool.Status  `json:"instances"`
}

// Stats reads the server's state.
func (r *Runtime) Stats() StatsReport {
	return StatsReport{Process: stats.ReadProcess(), Metrics: r.Engine.Metrics.Snapshot(), Instances: r.Pool.Status()}
}

// MountStats adds the operator's views of the server, admin-only:
// GET /api/v1/stats as JSON and GET /metrics in the Prometheus text format.
func (r *Runtime) MountStats(mux *http.ServeMux) {
	admin := httpx.RequireAdmin
	mux.Handle("GET /api/v1/stats", admin(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, r.Stats())
	})))
	mux.Handle("GET /metrics", admin(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		rep := r.Stats()
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		stats.WritePrometheus(w, rep.Metrics, gauges(rep))
	})))
}

func gauges(rep StatsReport) []stats.Gauge {
	var out []stats.Gauge
	add := func(name, help string, v float64, labels map[string]string) {
		out = append(out, stats.Gauge{Name: name, Help: help, Value: v, Labels: labels})
	}
	p := rep.Process
	add("nanollm_uptime_seconds", "Seconds since the server started.", p.UptimeS, nil)
	add("nanollm_goroutines", "Goroutines.", float64(p.Goroutines), nil)
	add("nanollm_rss_bytes", "Resident set size of the server process.", float64(p.RSS), nil)
	add("nanollm_rss_peak_bytes", "Peak resident set size.", float64(p.RSSPeak), nil)
	add("nanollm_locked_bytes", "Memory locked with mlock.", float64(p.Locked), nil)
	add("nanollm_go_heap_bytes", "Go heap in use.", float64(p.GoHeap), nil)
	for _, in := range rep.Instances {
		l := map[string]string{"model": in.Key.Model, "device": in.Key.Device}
		add("nanollm_slots", "Slots of a loaded model.", float64(in.Loop.Slots), l)
		add("nanollm_slots_busy", "Slots running a request.", float64(in.Loop.Busy), l)
		add("nanollm_pending_requests", "Requests waiting for a slot.", float64(in.Loop.Pending), l)
		add("nanollm_steps_total", "Model steps taken.", float64(in.Loop.Steps), l)
		add("nanollm_step_tokens_total", "Tokens in those steps' batches.", float64(in.Loop.StepTokens), l)
		add("nanollm_model_references", "Requests holding the model.", float64(in.Refs), l)
	}
	return out
}
