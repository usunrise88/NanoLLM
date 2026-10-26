//go:build !windows

package posture

import (
	"os"
	"strconv"
	"sync"

	"golang.org/x/sys/unix"
)

var (
	mu  sync.Mutex
	low bool
)

// Apply lowers the priority of every thread of the process to Nice when low is
// set; threads started later inherit it from the thread that starts them. A
// process cannot raise its niceness back without CAP_SYS_NICE, so Apply(false)
// after Apply(true) returns ErrRestart and the server stays quiet until it is
// restarted.
func Apply(want bool) error {
	mu.Lock()
	defer mu.Unlock()
	if !want {
		if low {
			return ErrRestart
		}
		return nil
	}
	low = true
	return reapply()
}

// Reapply lowers the threads the process has started since Apply: the worker
// threads of a model loaded later. It does nothing while the posture is normal.
func Reapply() error {
	mu.Lock()
	defer mu.Unlock()
	if !low {
		return nil
	}
	return reapply()
}

func reapply() error {
	var first error
	set := func(tid int) {
		if err := unix.Setpriority(unix.PRIO_PROCESS, tid, Nice); err != nil && first == nil {
			first = err
		}
	}
	ents, err := os.ReadDir("/proc/self/task")
	if err != nil {
		// No /proc (macOS): the whole process.
		set(0)
		return first
	}
	for _, e := range ents {
		if tid, err := strconv.Atoi(e.Name()); err == nil {
			set(tid)
		}
	}
	return first
}

// Current reads the posture back from the main thread's niceness.
func Current() State {
	n, err := unix.Getpriority(unix.PRIO_PROCESS, 0)
	// getpriority's raw return is 20-nice on Linux; x/sys hands it back so.
	return State{Low: err == nil && n < 20-5}
}
