package tune

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/placement"
	"github.com/usunrise88/nanollm/internal/registry"
)

// Active is what the server holds of the autotuner: the mode, its limits and
// the profile, if a current one exists. The server places models by it
// (PlaceTune); it never measures by itself (docs/OPTIMIZE.md, §3.7).
type Active struct {
	Mode        Mode        `json:"mode"`
	Constraints Constraints `json:"constraints"`
	// Profile is the profile in force: present only when it is current. A
	// stale or missing profile leaves the server on the limits of the mode and
	// the static rules.
	Profile *Profile `json:"profile,omitempty"`
	// Stale says a profile exists but its key no longer matches; Changes lists
	// what differs. Missing says the mode is set and no profile exists for this
	// machine (the configuration was moved).
	Stale   bool     `json:"stale,omitempty"`
	Missing bool     `json:"missing,omitempty"`
	Changes []Change `json:"changes,omitempty"`
	// Problems are the findings that hold while the server runs, such as the
	// powerful mode that has too little free memory.
	Problems []Problem `json:"problems,omitempty"`
}

// ID names the placement in the pool key: the mode alone, or mode@hash with a
// profile in force.
func (a *Active) ID() string {
	if a == nil || !a.Mode.Valid() {
		return ""
	}
	if a.Profile != nil {
		return a.Profile.ID()
	}
	return string(a.Mode)
}

// DefaultDevice is the device the autotuner picks when the configuration's
// devices.default is auto or empty and the request names none: the CPU in the
// quiet mode, the device a profile in force was measured on; empty leaves the
// usual choice (the first GPU, else the CPU).
func (a *Active) DefaultDevice(configured string) string {
	if a == nil || !a.Mode.Valid() || (configured != "" && configured != config.DeviceAuto) {
		return ""
	}
	if a.Constraints.CPUOnly {
		return config.DeviceCPU
	}
	if a.Profile != nil {
		if sel, ok := a.Profile.Chosen.Device.Value.(string); ok {
			return sel
		}
	}
	return ""
}

// Message is the sentence the log and the API give when the profile is not in
// force, and the empty string when it is.
func (a *Active) Message() string {
	switch {
	case a == nil, a.Profile != nil:
		return ""
	case a.Stale:
		var ch []string
		for _, c := range a.Changes {
			ch = append(ch, c.String())
		}
		return fmt.Sprintf("the %s profile is stale (%s); recheck with: nanollm optimize", a.Mode, strings.Join(ch, "; "))
	case a.Missing:
		return fmt.Sprintf("tune.mode is %s but this machine has no profile; measure with: nanollm optimize", a.Mode)
	}
	return ""
}

// PlaceTune is what the autotuner contributes to a placement on the device: the
// limits of the mode, and the profile's choices if the device is the one it was
// measured on.
func (a *Active) PlaceTune(deviceSelector string) placement.Tune {
	if a == nil || !a.Mode.Valid() {
		return placement.Tune{}
	}
	t := ModeTune(a.Constraints)
	if a.Profile == nil {
		return t
	}
	c := a.Profile.Chosen
	sel, _ := c.Device.Value.(string)
	if sel != "" && sel != deviceSelector {
		return t // another device than the profile measured: the limits only
	}
	t.Source = placement.SourceProfile
	if p, ok := c.Preset.Value.(string); ok && p != "" && a.Constraints.Goal == GoalMinMemory {
		t.Preset = p
	}
	if d, ok := decodeDraft(c.MTP.Value); ok {
		enabled := d.Enabled
		t.MTP = &enabled
		if d.Enabled {
			t.DraftMax, t.DraftPMin = &d.DraftMax, &d.PMin
		}
	}
	if tm, ok := decodeTeam(c.Threads.Value); ok && tm.Threads > 0 {
		t.Threads, t.ThreadsBatch = tm.Threads, tm.ThreadsBatch
	}
	return t
}

// decodeDraft reads a Draft from a value that is one or has come back from
// JSON as a map.
func decodeDraft(v any) (Draft, bool) {
	switch x := v.(type) {
	case Draft:
		return x, true
	case map[string]any:
		d := Draft{}
		if b, ok := x["enabled"].(bool); ok {
			d.Enabled = b
		}
		if f, ok := x["draft_max"].(float64); ok {
			d.DraftMax = int(f)
		}
		if f, ok := x["p_min"].(float64); ok {
			d.PMin = f
		}
		return d, true
	}
	return Draft{}, false
}

func decodeTeam(v any) (Team, bool) {
	switch x := v.(type) {
	case Team:
		return x, true
	case map[string]any:
		t := Team{}
		if f, ok := x["threads"].(float64); ok {
			t.Threads = int(f)
		}
		if f, ok := x["threads_batch"].(float64); ok {
			t.ThreadsBatch = int(f)
		}
		return t, true
	}
	return Team{}, false
}

// ActiveInput is what the server knows at start.
type ActiveInput struct {
	Config   config.Config
	Manifest registry.Manifest
	Devices  []backend.Device
	LibDir   string
	// LibsRelease, LlamaCommit are the builds of this binary.
	LibsRelease string
	LlamaCommit string
}

// LoadActive reads the mode from the configuration and the profile from the
// store, and checks the profile's key against the machine as it is now. It is
// quick: no process is started but nvidia-smi, once. A configuration with no
// mode gives nil.
func LoadActive(ctx context.Context, in ActiveInput) (*Active, error) {
	mode := Mode(in.Config.Tune.Mode)
	if mode == "" {
		return nil, nil
	}
	if !mode.Valid() {
		return nil, fmt.Errorf("tune.mode %q is not silent, optimal or powerful", mode)
	}
	cores := Cores{Physical: hwinfo.PhysicalCores(), Logical: runtime.NumCPU()}
	_, avail := hwinfo.MemoryNow()
	devs := quickDevices(ctx, in.Devices)
	a := &Active{Mode: mode}
	var deg *Problem
	a.Constraints, deg = ConstraintsFor(mode, cores, freeBudgetOf(devs, avail))
	if deg != nil {
		a.Problems = append(a.Problems, *deg)
	}
	st := Store{Dir: in.Config.TunePath()}
	prof, err := st.Load(in.Manifest.ID, mode)
	switch {
	case errors.Is(err, os.ErrNotExist):
		a.Missing = true
		return a, nil
	case err != nil:
		return a, fmt.Errorf("the profile cannot be read: %w", err)
	}
	now := currentKey(prof, mode, devs, in)
	if ch := prof.Key.Diff(now); len(ch) > 0 {
		a.Stale, a.Changes = true, ch
		return a, nil
	}
	a.Profile = prof
	return a, nil
}

// freeBudgetOf is FreeBudget over devices without a Survey.
func freeBudgetOf(devs []Device, avail int64) int64 {
	s := &Survey{Devices: devs}
	return FreeBudget(s)
}

// quickDevices converts the backend devices and fills in their drivers.
func quickDevices(ctx context.Context, in []backend.Device) []Device {
	s := &Survey{GPUs: hwinfo.NVIDIALive(ctx), Adapters: hwinfo.GPUDrivers()}
	for _, d := range in {
		s.Devices = append(s.Devices, deviceOf(d))
	}
	s.attachDrivers()
	return s.Devices
}

// currentKey is the key the profile would have if it were measured now: the
// profile's own device looked up among the devices present.
func currentKey(prof *Profile, mode Mode, devs []Device, in ActiveInput) Key {
	k := Key{
		Mode: mode, Model: in.Manifest.ID, LibsRelease: in.LibsRelease, LlamaCommit: in.LlamaCommit, CUDABuild: cudaBuild(in.LibDir),
		Schema: ProfileSchema, Suite: SuiteVersion, CPU: hwinfo.CPUName(), Cores: hwinfo.PhysicalCores(),
	}
	total, _ := hwinfo.MemoryNow()
	k.RAMGiB = int(total >> 30)
	if f, ok := in.Manifest.File(registry.RoleModel); ok {
		k.ModelSHA256 = f.SHA256
	}
	if f, ok := in.Manifest.File(registry.RoleDrafter); ok {
		k.DrafterSHA256 = f.SHA256
	}
	sel, _ := prof.Chosen.Device.Value.(string)
	for _, d := range devs {
		if d.Selector == sel {
			k.Device = fmt.Sprintf("%s %s %d GiB", d.Selector, d.Name, (d.MemoryTotal+GiB/2)>>30)
			k.Driver = d.Driver
			return k
		}
	}
	// The device is gone: the key names nothing, and the profile differs.
	return k
}

// Staleness is the API's view of the state: whether a recheck is due and with
// which mode.
type Staleness struct {
	Stale   bool     `json:"stale"`
	Missing bool     `json:"missing,omitempty"`
	Changes []Change `json:"changes,omitempty"`
	Suggest *Suggest `json:"suggest,omitempty"`
}

// Suggest is the request that would renew the profile.
type Suggest struct {
	Mode Mode `json:"mode"`
}

// Staleness reports whether a recheck is due.
func (a *Active) Staleness() Staleness {
	if a == nil || (a.Profile != nil) {
		return Staleness{}
	}
	return Staleness{Stale: a.Stale || a.Missing, Missing: a.Missing, Changes: a.Changes, Suggest: &Suggest{Mode: a.Mode}}
}

// age is how long ago a profile was made.
func (p *Profile) age() time.Duration { return time.Since(p.Created) }
