package httpx

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// NewNDJSON is NewStream for newline-delimited JSON, the framing of Ollama's
// streaming endpoints: one JSON value per line, no event names and no comment
// lines. The headers, the flush after every line and the per-write deadline are
// the event stream's; Send, Data and Comment are not meant for it, Line is.
//
// A blank line is not a keep-alive here: Ollama clients parse every line they
// read, so a silent stream is kept open with a valid, empty chunk instead, and
// that is the caller's to choose.
func NewNDJSON(w http.ResponseWriter) (*Stream, error) {
	return newStream(w, "application/x-ndjson")
}

// Line writes v as one line of JSON and flushes it.
func (s *Stream) Line(v any) error {
	// json.Marshal escapes newlines inside strings, so a value is one line.
	body, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("ndjson: encode: %w", err)
	}
	if err := s.write(append(body, '\n')); err != nil {
		return err
	}
	s.flusher.Flush()
	s.resetHeartbeat()
	return nil
}
