package managed

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/tune"
)

// Failures keeps the measurements that failed, one file per model and mode
// beside the profiles: tune/<model>.<mode>.failed.json. A start does not
// measure again while the machine, its drivers, the libraries, the model and
// this binary are what they were when it failed; the application can still
// ask for a measurement (POST /api/v1/managed/tune).
type Failures struct {
	Dir string
	// Version is this binary's; a new one measures again.
	Version string
}

// Failure is one failed measurement.
type Failure struct {
	Version string    `json:"version"`
	Key     tune.Key  `json:"key"`
	Error   string    `json:"error"`
	At      time.Time `json:"at"`
}

func (f Failures) path(model string, mode tune.Mode) string {
	safe := strings.Map(func(r rune) rune {
		if r == '/' || r == '\\' || r == ':' || r == 0 {
			return '_'
		}
		return r
	}, model)
	return filepath.Join(f.Dir, safe+"."+string(mode)+".failed.json")
}

// Record keeps a failure of the model and mode on the machine key.
func (f Failures) Record(model string, mode tune.Mode, key tune.Key, cause error) error {
	if err := os.MkdirAll(f.Dir, 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(Failure{Version: f.Version, Key: key, Error: cause.Error(), At: time.Now().UTC()}, "", "  ")
	if err != nil {
		return err
	}
	path := f.path(model, mode)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// Matches returns the failure recorded for the model and mode when nothing
// has changed since: the same binary and the same machine key.
func (f Failures) Matches(model string, mode tune.Mode, key tune.Key) (*Failure, bool) {
	b, err := os.ReadFile(f.path(model, mode))
	if err != nil {
		return nil, false
	}
	var rec Failure
	if json.Unmarshal(b, &rec) != nil || rec.Version != f.Version || len(rec.Key.Diff(key)) > 0 {
		return nil, false
	}
	return &rec, true
}

// Clear forgets the failure of the model and mode, after a measurement that
// succeeded.
func (f Failures) Clear(model string, mode tune.Mode) { _ = os.Remove(f.path(model, mode)) }
