package managed

import (
	"io"
	"net/http"
	"sync"

	"github.com/usunrise88/nanollm/internal/core"
	"github.com/usunrise88/nanollm/internal/httpx"
	"github.com/usunrise88/nanollm/internal/tune"
)

// Status is what GET /api/v1/managed answers: what the application chose,
// what this process serves, and what is still to do.
type Status struct {
	Version    string `json:"version"`
	APIVersion int    `json:"api_version"`
	// Configured is false until the application saves settings for the
	// first time; the first start of an instance is its setup.
	Configured bool `json:"configured"`
	// RestartRequired says the next start would run as the settings say and
	// this one cannot: RestartReasons names what.
	RestartRequired bool     `json:"restart_required"`
	RestartReasons  []string `json:"restart_reasons,omitempty"`
	DataDir         string   `json:"data_dir"`
	ModelsDir       string   `json:"models_dir"`
	Settings        Settings `json:"settings"`
	Features        Features `json:"features"`
	// MissingModels is every model the settings need that is not on disk:
	// what the application has to download.
	MissingModels []string `json:"missing_models"`
}

// Features are what the instance serves.
type Features struct {
	Chat     Feature   `json:"chat"`
	Autotune TuneState `json:"autotune"`
}

// Feature is the chat as the status reports it.
type Feature struct {
	State  string   `json:"state"`
	Models []string `json:"models"`
	// Missing are those of them not on disk.
	Missing []string `json:"missing,omitempty"`
}

// Chat states.
const (
	// ChatMissing: the model is not on disk; download it.
	ChatMissing = "missing"
	// ChatLoading: the model is on disk and not loaded yet, or held back by a
	// measurement. A request waits for the load or gets 503 with a
	// Retry-After.
	ChatLoading = "loading"
	// ChatReady: the model is loaded.
	ChatReady = "ready"
)

// Controller holds the settings of a running instance and serves them.
type Controller struct {
	o ControllerOptions

	mu         sync.Mutex
	settings   Settings
	configured bool
}

// ControllerOptions builds a Controller.
type ControllerOptions struct {
	Version    string
	Paths      Paths
	Settings   Settings
	Configured bool
	Lookup     Lookup
	// Present reports whether a model is on disk; Loaded whether the default
	// model is resident.
	Present func(id string) bool
	Loaded  func() bool
	// Apply takes on, in the running process, what changed from prev to next
	// apart from the mode and the model, which the autotuner puts in force:
	// the default model for requests, thinking, the mirrors, the log level.
	Apply    func(prev, next Settings)
	Autotune *Autotune
}

// NewController builds the controller.
func NewController(o ControllerOptions) *Controller {
	return &Controller{o: o, settings: o.Settings, configured: o.Configured}
}

// Settings are the settings in force.
func (c *Controller) Settings() Settings {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.settings
}

// Status reports the instance as it stands now.
func (c *Controller) Status() Status {
	c.mu.Lock()
	s, configured := c.settings, c.configured
	c.mu.Unlock()
	model := s.Model()
	st := Status{
		Version:       c.o.Version,
		APIVersion:    APIVersion,
		Configured:    configured,
		DataDir:       c.o.Paths.Data,
		ModelsDir:     c.o.Paths.Models,
		Settings:      s,
		MissingModels: []string{},
	}
	if st.Settings.Mirrors == nil {
		// A list in JSON either way, so a reader need not handle null.
		st.Settings.Mirrors = []string{}
	}
	chat := Feature{State: ChatReady, Models: []string{model}}
	switch {
	case !c.o.Present(model):
		chat.State, chat.Missing = ChatMissing, []string{model}
		st.MissingModels = append(st.MissingModels, model)
	case !c.o.Loaded():
		chat.State = ChatLoading
	}
	st.Features = Features{Chat: chat, Autotune: c.o.Autotune.State()}
	st.RestartReasons = c.o.Autotune.RestartReasons()
	st.RestartRequired = len(st.RestartReasons) > 0
	return st
}

// maxSettingsBody bounds a settings request; the settings are a few hundred
// bytes.
const maxSettingsBody = 64 << 10

// Mount adds the managed endpoints:
//
//	GET   /api/v1/managed           the status
//	PATCH /api/v1/managed/settings  change and save settings; answers the status
//	POST  /api/v1/managed/tune      measure now and apply → 202 {run, estimate_s}
func (c *Controller) Mount(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/v1/managed", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, c.Status())
	})
	mux.HandleFunc("PATCH /api/v1/managed/settings", func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, maxSettingsBody+1))
		if err != nil {
			httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "cannot read the request body"))
			return
		}
		if len(body) > maxSettingsBody {
			httpx.WriteProblem(w, core.Errorf(core.CodeInvalidRequest, "the settings are too large"))
			return
		}
		st, err := c.Update(body)
		if err != nil {
			httpx.WriteProblem(w, err)
			return
		}
		httpx.WriteJSON(w, http.StatusOK, st)
	})
	mux.HandleFunc("POST /api/v1/managed/tune", func(w http.ResponseWriter, r *http.Request) {
		run, err := c.o.Autotune.Request(r.Context())
		if err != nil {
			httpx.WriteProblem(w, err)
			return
		}
		httpx.WriteJSON(w, http.StatusAccepted, map[string]any{"run": run.ID, "estimate_s": run.EstimateS})
	})
}

// Update applies a settings patch, saves it, takes it on, and reports the new
// status. A new mode or model is put in force by the autotuner, which
// measures first when the mode has no current profile for the model: the
// status says it runs.
func (c *Controller) Update(patch []byte) (Status, error) {
	c.mu.Lock()
	prev := c.settings
	next, err := prev.Patch(patch)
	if err != nil {
		c.mu.Unlock()
		return Status{}, err
	}
	if err := next.Validate(c.o.Lookup); err != nil {
		c.mu.Unlock()
		return Status{}, err
	}
	if err := SaveSettings(c.o.Paths.Settings, next); err != nil {
		c.mu.Unlock()
		return Status{}, core.Errorf(core.CodeInternal, "cannot save the settings").WithCause(err)
	}
	c.settings, c.configured = next, true
	c.mu.Unlock()

	if c.o.Apply != nil {
		c.o.Apply(prev, next)
	}
	if prev.TuneMode() != next.TuneMode() || prev.Model() != next.Model() || (!prev.AutotuneOn() && next.AutotuneOn()) {
		c.o.Autotune.Kick(ReasonSettings)
	}
	return c.Status(), nil
}

// AdoptMode records in the settings the mode of a profile the application
// applied through POST /api/v1/optimize/{id}/apply: the settings are the
// configuration of a managed instance, as tune.mode in the file is the
// service's.
func (c *Controller) AdoptMode(mode tune.Mode) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.settings.TuneMode() == mode {
		return nil
	}
	next := c.settings
	next.Mode = string(mode)
	if err := SaveSettings(c.o.Paths.Settings, next); err != nil {
		return err
	}
	c.settings, c.configured = next, true
	return nil
}

// Downloaded is told of a model that has just arrived on disk: the default
// model's arrival is the moment to measure.
func (c *Controller) Downloaded(id string) {
	if id == c.Settings().Model() {
		c.o.Autotune.Kick(ReasonDownload)
	}
}
