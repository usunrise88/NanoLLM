//go:build linux

package bench

import (
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// cgroupWatch samples the memory of the cgroup this process runs in. A strict
// worker is alone in a scope of its own, so the numbers are the run's:
// memory.peak is what the limit is compared against, and the page cache
// (memory.stat "file") is what a file-backed mode spends instead of RSS.
type cgroupWatch struct {
	dir  string
	stop chan struct{}
	done sync.WaitGroup

	mu        sync.Mutex
	pageCache int64
}

// watchCgroup starts sampling; the zero watch (no cgroup v2, no access) does
// nothing and reports zeros.
func watchCgroup() *cgroupWatch {
	w := &cgroupWatch{}
	b, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return w
	}
	for _, line := range strings.Split(string(b), "\n") {
		if rest, ok := strings.CutPrefix(line, "0::"); ok {
			w.dir = "/sys/fs/cgroup" + strings.TrimSpace(rest)
		}
	}
	if _, err := os.Stat(w.dir + "/memory.peak"); err != nil {
		w.dir = ""
		return w
	}
	w.stop = make(chan struct{})
	w.done.Add(1)
	go func() {
		defer w.done.Done()
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for {
			w.sample()
			select {
			case <-w.stop:
				return
			case <-t.C:
			}
		}
	}()
	return w
}

func (w *cgroupWatch) sample() {
	file := statField(w.dir+"/memory.stat", "file")
	w.mu.Lock()
	w.pageCache = max(w.pageCache, file)
	w.mu.Unlock()
}

// finish stops sampling and returns memory.peak and the page cache's peak.
func (w *cgroupWatch) finish() (peak, pageCache int64) {
	if w.dir == "" {
		return 0, 0
	}
	close(w.stop)
	w.done.Wait()
	w.sample()
	b, _ := os.ReadFile(w.dir + "/memory.peak")
	peak, _ = strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	w.mu.Lock()
	defer w.mu.Unlock()
	return peak, w.pageCache
}

// statField reads one key of a flat-keyed cgroup file ("file 1234").
func statField(path, key string) int64 {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if k, v, ok := strings.Cut(line, " "); ok && k == key {
			n, _ := strconv.ParseInt(v, 10, 64)
			return n
		}
	}
	return 0
}
