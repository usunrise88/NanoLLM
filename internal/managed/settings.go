package managed

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/registry"
	"github.com/usunrise88/nanollm/internal/tune"
)

// Settings is what the application chose. It is the whole of what a managed
// instance can be told: the rest of the configuration is either fixed by
// running managed (the address, the key, the paths) or derived from these.
//
// A zero value is an instance nobody has set up yet. It runs anyway, with the
// catalog's default model in the quiet mode.
type Settings struct {
	// DefaultModel answers a request that names no model; empty is the
	// catalog's default, gemma-4-e4b-qat.
	DefaultModel string `json:"default_model"`
	// Mode is the autotuner's mode: silent, optimal or powerful. Empty is
	// silent: a process in the background of an application does not take
	// the machine unless the application says so.
	Mode string `json:"mode"`
	// Autotune is auto (or empty), to measure whenever the mode has no
	// current profile for the model, or off, to run by the mode's limits
	// unless the application asks for a measurement.
	Autotune string `json:"autotune"`
	// Thinking lets a request that does not say think before it answers.
	Thinking bool `json:"thinking"`
	// Mirrors are base URLs to fetch models from when the catalog's own
	// location fails, tried in order.
	Mirrors []string `json:"mirrors"`
	// LogLevel is debug, info, warn or error; empty is info.
	LogLevel string `json:"log_level"`
}

// Autotune values.
const (
	AutotuneAuto = "auto"
	AutotuneOff  = "off"
)

// Model is the model the settings run.
func (s Settings) Model() string {
	if s.DefaultModel == "" {
		return config.DefaultModel
	}
	return s.DefaultModel
}

// TuneMode is the mode the settings run.
func (s Settings) TuneMode() tune.Mode {
	if s.Mode == "" {
		return tune.Silent
	}
	return tune.Mode(s.Mode)
}

// AutotuneOn says whether the instance measures by itself.
func (s Settings) AutotuneOn() bool { return s.Autotune != AutotuneOff }

// LoadSettings reads the settings file. A file that does not exist is not an
// error: it is an instance that has not been set up, and configured is false.
func LoadSettings(path string) (s Settings, configured bool, err error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return Settings{}, false, nil
	}
	if err != nil {
		return Settings{}, false, err
	}
	if err := decodeStrict(b, &s); err != nil {
		return Settings{}, false, fmt.Errorf("%s: %w", path, err)
	}
	return s, true, nil
}

// SaveSettings replaces the settings file whole: a crash leaves the old
// settings or the new, never half of either.
func SaveSettings(path string, s Settings) error {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
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

// Patch applies the fields present in a JSON object to s and leaves the rest.
// A field nobody knows is refused rather than ignored: a misspelt setting
// that silently does nothing is found weeks later.
func (s Settings) Patch(body []byte) (Settings, error) {
	if err := decodeStrict(body, &s); err != nil {
		return s, core.Errorf(core.CodeInvalidRequest, "the settings are not valid: %s", err.Error())
	}
	return s, nil
}

func decodeStrict(b []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return errors.New("trailing data after the JSON object")
	}
	return nil
}

// Lookup describes a model of the catalog by id.
type Lookup func(id string) (registry.Manifest, error)

// Validate refuses settings an instance could not run with, naming the field.
// The model is checked against the catalog rather than the disk: it is
// usually chosen before it is downloaded.
func (s Settings) Validate(lookup Lookup) error {
	if s.DefaultModel != "" {
		if _, err := lookup(s.DefaultModel); err != nil {
			return core.Errorf(core.CodeInvalidRequest, "default_model: %s", core.AsError(err).Message).WithParam("default_model")
		}
	}
	if s.Mode != "" && !tune.Mode(s.Mode).Valid() {
		return core.Errorf(core.CodeInvalidRequest, "mode must be silent, optimal or powerful").WithParam("mode")
	}
	switch s.Autotune {
	case "", AutotuneAuto, AutotuneOff:
	default:
		return core.Errorf(core.CodeInvalidRequest, "autotune must be auto or off").WithParam("autotune")
	}
	for _, m := range s.Mirrors {
		if u, err := url.Parse(m); err != nil || u.Scheme != "https" || u.Host == "" {
			return core.Errorf(core.CodeInvalidRequest, "mirrors: %q is not an https URL", m).WithParam("mirrors")
		}
	}
	switch s.LogLevel {
	case "", "debug", "info", "warn", "error":
	default:
		return core.Errorf(core.CodeInvalidRequest, "log_level must be debug, info, warn or error").WithParam("log_level")
	}
	return nil
}
