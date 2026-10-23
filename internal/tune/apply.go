package tune

import (
	"errors"
	"fmt"
	"os"
	"runtime"

	"github.com/usunrise88/nanollm/internal/config"
)

// ApplyOptions say where a profile goes.
type ApplyOptions struct {
	// Store is the directory of profiles.
	Store Store
	// ConfigPath is the configuration file to write tune.mode into; empty
	// leaves the configuration alone.
	ConfigPath string
}

// ApplyResult is what applying a profile did.
type ApplyResult struct {
	ProfileSaved  bool   `json:"profile_saved"`
	ConfigWritten bool   `json:"config_written"`
	ConfigPath    string `json:"config_path,omitempty"`
	// RestartRequired lists what a running server cannot take on without a
	// restart: the memory policy of the NUMA nodes, the priority of a server
	// that was quiet (Linux cannot raise it again).
	RestartRequired []string `json:"restart_required,omitempty"`
	// Note says what could not be done, for the person to do.
	Note string `json:"note,omitempty"`
}

// Apply makes a profile the one in force: saves it (the previous one moves to
// .prev) and writes tune.mode, and devices.numa when the profile chose a memory
// policy, into the configuration. The configuration keeps its comments and its
// order (config.Document). A server that is running takes the profile on
// through the API (the loaded model is reloaded); Apply alone only writes the
// files.
func Apply(p *Profile, o ApplyOptions) (ApplyResult, error) {
	var res ApplyResult
	if p == nil || !p.Mode.Valid() {
		return res, errors.New("tune: no profile to apply")
	}
	if err := o.Store.Save(p); err != nil {
		return res, fmt.Errorf("the profile cannot be saved: %w", err)
	}
	res.ProfileSaved = true
	if o.ConfigPath == "" {
		return res, nil
	}
	b, err := os.ReadFile(o.ConfigPath)
	if err != nil {
		// Not a failure of the run: the profile is saved, and the person is told
		// what to set by hand.
		res.Note = "the configuration cannot be read: " + err.Error()
		return res, nil //nolint:nilerr // the error is reported in Note
	}
	doc, err := config.ParseDocument(b)
	if err != nil {
		res.Note = err.Error()
		return res, nil //nolint:nilerr // the error is reported in Note
	}
	before := doc.TuneMode()
	doc.SetTuneMode(string(p.Mode))
	if p.Chosen.NUMA != nil {
		if policy, ok := p.Chosen.NUMA.Value.(string); ok && policy != "" {
			doc.SetString("devices.numa", policy)
			res.RestartRequired = append(res.RestartRequired, "devices.numa: "+policy)
		}
	}
	if before == string(Silent) && p.Mode != Silent && runtime.GOOS != "windows" {
		res.RestartRequired = append(res.RestartRequired, "priority: a quiet server cannot raise its priority again")
	}
	if err := doc.Save(o.ConfigPath); err != nil {
		res.Note = fmt.Sprintf("the configuration %s cannot be written (%v); set tune.mode: %s there", o.ConfigPath, err, p.Mode)
		res.RestartRequired = nil
		return res, nil
	}
	res.ConfigWritten, res.ConfigPath = true, o.ConfigPath
	return res, nil
}
