package managed

import (
	"github.com/usunrise88/nanollm/internal/config"
)

// Options is what a plan needs besides the settings.
type Options struct {
	Paths Paths
	// Token is the one key the instance accepts.
	Token string
	// UI mounts the web UI, which is otherwise left out.
	UI bool
	// Backend replaces the enabled backends with one; empty keeps llama.cpp.
	// The tests run on the scripted one.
	Backend string
}

// NewPlan turns settings into the configuration of one start. Nothing else
// goes into it: a managed instance reads no configuration file and no
// NANOLLM_* variable, so what the application chose is all there is.
func NewPlan(s Settings, o Options) (config.Config, error) {
	cfg := config.Default()

	cfg.Server.Addr = "127.0.0.1:0"
	cfg.Server.DataDir = o.Paths.Data
	cfg.Auth = config.Auth{Mode: "apikey", Keys: []config.APIKey{{
		Name: "app", Secret: o.Token, Admin: true, Interactive: true,
	}}}
	cfg.UI.Enabled = o.UI
	if o.Backend != "" {
		cfg.Backends.Enabled = []string{o.Backend}
	}

	cfg.Tune.Mode = string(s.TuneMode())
	cfg.Models.Default = s.Model()
	// The command loads the model itself once it knows whether a measurement
	// comes first: a model loaded only to be unloaded for the measurement is
	// seconds of disk for nothing.
	cfg.Models.Preload = nil
	// A process in the background of an application does not set the memory
	// policy of the machine's NUMA nodes; the autotuner's choice of one is not
	// applied either.
	cfg.Devices.NUMA = "off"
	cfg.Generation.EnableThinking = s.Thinking

	cfg.Registry.Mirrors = []string{}
	if s.Mirrors != nil {
		cfg.Registry.Mirrors = s.Mirrors
	}
	// An application's process never sends a benchmark anywhere.
	cfg.Bench.Submit = "never"

	cfg.Log.File = o.Paths.Log
	cfg.Log.Format = "json"
	if s.LogLevel != "" {
		cfg.Log.Level = s.LogLevel
	}

	if err := cfg.Validate(); err != nil {
		return config.Config{}, err
	}
	return cfg, nil
}
