package managed

import (
	"encoding/json"
	"io"
	"sync"
)

// Events are the lines the instance writes on its standard output, one JSON
// object each, for the application that started it. Nothing else is written
// there: the log goes to a file, so the application can read this stream
// line by line without filtering it.
//
//	{"event":"ready","url":"http://127.0.0.1:53124","token":"…",…}
//	{"event":"tune_started","run":"…","mode":"silent","model":"gemma-4-e4b-qat","reason":"missing","estimate_s":60}
//	{"event":"tune_done","run":"…","mode":"silent","profile":{…}}
//	{"event":"warning","code":"tune_failed","message":"…"}
//	{"event":"restart_required","reasons":["priority"]}
//	{"event":"stopping","reason":"stdin closed"}
//	{"event":"fatal","code":"data_dir_locked","message":"…"}
type Events struct {
	mu sync.Mutex
	w  io.Writer
}

// NewEvents writes events to w.
func NewEvents(w io.Writer) *Events { return &Events{w: w} }

// Event names.
const (
	EventReady           = "ready"
	EventTuneStarted     = "tune_started"
	EventTuneDone        = "tune_done"
	EventRestartRequired = "restart_required"
	EventWarning         = "warning"
	EventStopping        = "stopping"
	EventFatal           = "fatal"
)

// Warning codes.
const (
	// WarnSettingsReset is a settings file that could not be read: it was set
	// aside as settings.json.invalid and the instance starts unconfigured.
	WarnSettingsReset = "settings_reset"
	// WarnTuneFailed is a measurement that failed; the mode's limits hold
	// without a profile.
	WarnTuneFailed = "tune_failed"
	// WarnTuneSkipped is a measurement not started because the last one failed
	// on this very machine, model and build.
	WarnTuneSkipped = "tune_skipped"
)

// Emit writes one event. A write that fails is dropped: an application that
// stopped reading has nothing to hear, and the instance must not stop over it.
func (e *Events) Emit(name string, fields map[string]any) {
	if e == nil {
		return
	}
	body, err := json.Marshal(fields)
	if err != nil {
		return
	}
	head, _ := json.Marshal(name)
	line := append([]byte(`{"event":`), head...)
	if len(fields) > 0 {
		line = append(line, ',')
		line = append(line, body[1:]...) // the fields, without their opening brace
	} else {
		line = append(line, '}')
	}
	line = append(line, '\n')

	e.mu.Lock()
	defer e.mu.Unlock()
	_, _ = e.w.Write(line)
}

// Warn reports something the application may want to show or log; the
// instance goes on.
func (e *Events) Warn(code, message string, fields map[string]any) {
	all := map[string]any{"code": code, "message": message}
	for k, v := range fields {
		all[k] = v
	}
	e.Emit(EventWarning, all)
}

// Fatal reports why the instance cannot run. The process exits after it.
func (e *Events) Fatal(code, message string) {
	e.Emit(EventFatal, map[string]any{"code": code, "message": message})
}
