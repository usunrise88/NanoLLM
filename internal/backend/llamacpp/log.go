package llamacpp

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// llama.cpp and ggml log through one callback, which goes to slog. It is
// registered once per process: purego callbacks are a fixed, never-freed
// resource, and the callback is global C state anyway.
//
// ggml's INFO level is a load-time narrative (every GGUF key, every tensor
// buffer), so it maps to slog's Debug; WARN and ERROR keep their level. A
// message arrives in pieces — a line, or a CONT fragment such as the dots
// of the load progress — and is emitted per complete line.
//
// Some facts are only in the log: the CPU variant ggml picked, the buffer
// sizes per device, the KV cache cells. Code that needs one adds a tap,
// which sees every complete line.

var (
	logOnce sync.Once
	logMu   sync.Mutex
	logBuf  strings.Builder
	logLvl  GgmlLogLevel
	taps    = map[int]func(GgmlLogLevel, string){}
	tapNext int
)

func installLogger() {
	logOnce.Do(func() {
		// Every argument is pointer-sized: a Windows callback takes nothing
		// narrower, and must return a value.
		cb := purego.NewCallback(func(level uintptr, text *byte, _ uintptr) uintptr {
			onLog(GgmlLogLevel(int32(level)), cString(text))
			return 0
		})
		llamaLogSet(cb, 0)
	})
}

func onLog(level GgmlLogLevel, text string) {
	logMu.Lock()
	if level != GgmlLogLevelCont {
		if logBuf.Len() > 0 {
			emitLocked(logLvl, logBuf.String())
			logBuf.Reset()
		}
		logLvl = level
	}
	logBuf.WriteString(text)
	if s := logBuf.String(); strings.HasSuffix(s, "\n") {
		logBuf.Reset()
		for _, line := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
			emitLocked(logLvl, line)
		}
	}
	logMu.Unlock()
}

func emitLocked(level GgmlLogLevel, line string) {
	line = strings.TrimRight(line, " \r")
	if line == "" || strings.Trim(line, ".") == "" {
		return // load progress dots
	}
	for _, t := range taps {
		t(level, line)
	}
	sl := slog.LevelDebug
	switch level {
	case GgmlLogLevelWarn:
		sl = slog.LevelWarn
	case GgmlLogLevelError:
		sl = slog.LevelError
	}
	if slog.Default().Enabled(context.Background(), sl) {
		slog.Log(context.Background(), sl, line, "component", "llama.cpp")
	}
}

// addLogTap calls fn with every complete log line until the returned
// function is called. fn runs with the log lock held and must not log
// through llama.cpp itself.
func addLogTap(fn func(GgmlLogLevel, string)) (remove func()) {
	logMu.Lock()
	id := tapNext
	tapNext++
	taps[id] = fn
	logMu.Unlock()
	return func() {
		logMu.Lock()
		delete(taps, id)
		logMu.Unlock()
	}
}

// cString copies a NUL-terminated C string; nil is "".
func cString(p *byte) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}
