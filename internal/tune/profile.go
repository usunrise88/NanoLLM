package tune

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ProfileSchema names the layout of a profile file; it is part of the key, so
// a layout change makes old profiles stale instead of misread.
const ProfileSchema = "nanollm.tune/v1"

// Key is what a profile was measured on. A change in any part makes the
// profile stale (docs/OPTIMIZE.md, §3.6). Free memory is not part of it: it
// changes all the time and is recorded as a condition of the measurement.
type Key struct {
	Mode          Mode   `json:"mode"`
	Model         string `json:"model"`
	ModelSHA256   string `json:"model_sha256,omitempty"`
	DrafterSHA256 string `json:"drafter_sha256,omitempty"`
	LibsRelease   string `json:"libs_release"`
	LlamaCommit   string `json:"llama_commit,omitempty"`
	CUDABuild     string `json:"cuda_build,omitempty"`
	Schema        string `json:"schema"`
	Suite         int    `json:"suite"`
	// Device is the chosen device: its selector, name and memory.
	Device string `json:"device"`
	Driver string `json:"driver,omitempty"`
	CPU    string `json:"cpu"`
	Cores  int    `json:"cores"`
	RAMGiB int    `json:"ram_gib"`
}

// Change is one part of a key that differs.
type Change struct {
	Field string `json:"field"`
	Old   string `json:"old"`
	New   string `json:"new"`
}

func (c Change) String() string {
	if c.Old == "" {
		return fmt.Sprintf("%s: %s", c.Field, c.New)
	}
	if c.New == "" {
		return fmt.Sprintf("%s: %s removed", c.Field, c.Old)
	}
	return fmt.Sprintf("%s %s → %s", c.Field, c.Old, c.New)
}

// Diff lists the parts of the key that differ from the current one.
func (k Key) Diff(now Key) []Change {
	var out []Change
	add := func(field, old, cur string) {
		if old != cur {
			out = append(out, Change{field, old, cur})
		}
	}
	add("mode", string(k.Mode), string(now.Mode))
	add("model", k.Model, now.Model)
	add("model checksum", short(k.ModelSHA256), short(now.ModelSHA256))
	add("drafter checksum", short(k.DrafterSHA256), short(now.DrafterSHA256))
	add("libraries", k.LibsRelease, now.LibsRelease)
	add("llama.cpp", short(k.LlamaCommit), short(now.LlamaCommit))
	add("CUDA build", k.CUDABuild, now.CUDABuild)
	add("profile format", k.Schema, now.Schema)
	add("probe suite", fmt.Sprint(k.Suite), fmt.Sprint(now.Suite))
	add("device", k.Device, now.Device)
	add("driver", k.Driver, now.Driver)
	add("processor", k.CPU, now.CPU)
	add("cores", fmt.Sprint(k.Cores), fmt.Sprint(now.Cores))
	add("RAM (GiB)", fmt.Sprint(k.RAMGiB), fmt.Sprint(now.RAMGiB))
	return out
}

func short(s string) string {
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

// Value is one lever of the choice: its value, where the value comes from and
// why, with the numbers of the alternatives that were measured.
type Value struct {
	Value any `json:"value"`
	// Source is measured, class, inherited, mechanism, posture, rule or mode
	// (SourceMeasured and the like).
	Source string `json:"source"`
	Why    string `json:"why,omitempty"`
	// Alternatives are the measured options and their score (tokens per second
	// for drafting settings, seconds for devices).
	Alternatives map[string]float64 `json:"alternatives,omitempty"`
	// Confirmed is false for a value taken from a neighbouring class or from the
	// mechanism that the time did not allow to confirm by measuring.
	Confirmed bool `json:"confirmed"`
}

// Draft is the value of the MTP lever.
type Draft struct {
	Enabled  bool    `json:"enabled"`
	DraftMax int     `json:"draft_max,omitempty"`
	PMin     float64 `json:"p_min,omitempty"`
}

// Team is the value of the threads lever.
type Team struct {
	Threads      int `json:"threads"`
	ThreadsBatch int `json:"threads_batch"`
}

// Chosen is the setup the profile picks.
type Chosen struct {
	Device Value `json:"device"`
	// Preset is the memory budget the plan was made under, and PlanMode the
	// memory mode (M1…M9) the planner chose in it.
	Preset   Value `json:"preset"`
	PlanMode Value `json:"plan_mode"`
	NCtx     int   `json:"n_ctx"`
	Slots    int   `json:"slots"`
	MTP      Value `json:"mtp"`
	Threads  Value `json:"threads"`
	// NUMA is the memory policy on a machine with several nodes; empty
	// elsewhere.
	NUMA *Value `json:"numa,omitempty"`
}

// MemoryCheck is the plan against the measurement (docs/OPTIMIZE.md, §3.4).
type MemoryCheck struct {
	PlanBytes      int64 `json:"plan_bytes"`
	PrivateBytes   int64 `json:"private_bytes"`
	VRAMBytes      int64 `json:"vram_bytes"`
	MeasuredBytes  int64 `json:"measured_bytes"`
	LimitBytes     int64 `json:"limit_bytes,omitempty"`
	ToleranceBytes int64 `json:"tolerance_bytes,omitempty"`
	// Source says whether the video memory was read from the process or
	// estimated from the engine's buffers.
	Source string `json:"source,omitempty"`
}

// Record is what happened to one candidate, for the report and the profile.
type Record struct {
	ID            string  `json:"id"`
	Mode          string  `json:"mode"`
	PlanBytes     int64   `json:"plan_bytes"`
	NCtx          int     `json:"n_ctx"`
	EstimateS     float64 `json:"estimate_s,omitempty"`
	Measured      bool    `json:"measured"`
	MeasuredBytes int64   `json:"measured_bytes,omitempty"`
	PP            float64 `json:"pp,omitempty"`
	TG            float64 `json:"tg,omitempty"`
	AnswerS       float64 `json:"answer_s,omitempty"`
	LoadS         float64 `json:"load_s,omitempty"`
	Verdict       string  `json:"verdict"`
	Reason        string  `json:"reason,omitempty"`
	Extrapolated  bool    `json:"extrapolated,omitempty"`
}

// Profile is the result of an optimization: the key, what was found, what was
// chosen and why. It lives in <data_dir>/tune/<model>.<mode>.json.
type Profile struct {
	Schema string `json:"schema"`
	Mode   Mode   `json:"mode"`
	// Constraints are the limits of the mode the profile was made under.
	Constraints Constraints `json:"constraints"`
	Key         Key         `json:"key"`
	Created     time.Time   `json:"created"`
	DurationS   float64     `json:"duration_s"`
	// DryRun marks a look without a measurement; such a profile is never applied.
	Conditions Conditions  `json:"conditions"`
	Survey     *Survey     `json:"survey,omitempty"`
	Candidates []Record    `json:"candidates"`
	Chosen     Chosen      `json:"chosen"`
	Memory     MemoryCheck `json:"memory"`
	// Answer is the typical answer on the chosen setup: its prefill and decode
	// rates and the time of the reference request.
	Answer   AnswerInfo `json:"answer"`
	Problems []Problem  `json:"problems,omitempty"`
	// Skipped names what the time budget left unmeasured.
	Skipped []string `json:"skipped,omitempty"`
	// Verified says how the choice was checked: the comparison with the setup
	// that runs without a profile, and the check that drafting keeps the answer.
	Verified Verified `json:"verified"`
}

// Conditions are the circumstances of the measurement.
type Conditions struct {
	Power      string  `json:"power,omitempty"`
	CPUPercent float64 `json:"cpu_percent"`
	GPUTempC   int     `json:"gpu_temp_c,omitempty"`
	FreeRAM    int64   `json:"free_ram"`
}

// AnswerInfo is the reference request on the chosen setup.
type AnswerInfo struct {
	PP           float64 `json:"pp"`
	TG           float64 `json:"tg"`
	AnswerS      float64 `json:"answer_s"`
	Extrapolated bool    `json:"extrapolated,omitempty"`
	LoadS        float64 `json:"load_s,omitempty"`
}

// Verified is the result of the checks after the choice.
type Verified struct {
	// StaticTokS and ChosenTokS are the decode rates of the setup that runs
	// without a profile and of the chosen one, in the same probe; Gain is their
	// ratio.
	StaticTokS float64 `json:"static_tok_s,omitempty"`
	ChosenTokS float64 `json:"chosen_tok_s,omitempty"`
	Gain       float64 `json:"gain,omitempty"`
	// RolledBack names the levers returned to the rules because the general
	// check found them slower.
	RolledBack []string `json:"rolled_back,omitempty"`
	// DraftChecked and DraftMatch are the correctness check of drafting.
	DraftChecked bool `json:"draft_checked,omitempty"`
	DraftMatch   bool `json:"draft_match,omitempty"`
	// DraftNearTie says drafted and plain decoding parted at a near tie of two
	// words (numerics of the batch kernels), which is not a mismatch.
	DraftNearTie bool `json:"draft_near_tie,omitempty"`
}

// KeyNow builds the key for the machine as surveyed and the device chosen.
func KeyNow(s *Survey, mode Mode, d Device) Key {
	return Key{
		Mode: mode, Model: s.Model.ID, ModelSHA256: s.Model.SHA256, DrafterSHA256: s.Model.DrafterSHA256,
		LibsRelease: s.Versions.LibsRelease, LlamaCommit: s.Versions.LlamaCommit, CUDABuild: s.Versions.CUDABuild,
		Schema: ProfileSchema, Suite: SuiteVersion,
		Device: fmt.Sprintf("%s %s %d GiB", d.Selector, d.Name, (d.MemoryTotal+GiB/2)>>30), Driver: d.Driver,
		CPU: s.CPU.Model, Cores: s.CPU.Physical, RAMGiB: int(s.RAM.Total >> 30),
	}
}

// Hash is a short identity of the choice, part of the pool key: two profiles
// that choose the same make the same instance, whatever the numbers they were
// measured with (Why and Alternatives) and whether a value was measured or
// inherited.
func (p *Profile) Hash() string {
	c := p.Chosen
	var numa any
	if c.NUMA != nil {
		numa = c.NUMA.Value
	}
	b, _ := json.Marshal(struct {
		Mode     Mode `json:"mode"`
		Device   any  `json:"device"`
		Preset   any  `json:"preset"`
		PlanMode any  `json:"plan_mode"`
		NCtx     int  `json:"n_ctx"`
		Slots    int  `json:"slots"`
		MTP      any  `json:"mtp"`
		Threads  any  `json:"threads"`
		NUMA     any  `json:"numa,omitempty"`
	}{p.Mode, c.Device.Value, c.Preset.Value, c.PlanMode.Value, c.NCtx, c.Slots, c.MTP.Value, c.Threads.Value, numa})
	// A profile read back from its file holds maps where the one just made held
	// structs; through a generic value both marshal with sorted keys.
	var generic any
	if json.Unmarshal(b, &generic) == nil {
		b, _ = json.Marshal(generic)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])[:8]
}

// ID is the profile's name for the pool key and the logs: mode@hash.
func (p *Profile) ID() string { return string(p.Mode) + "@" + p.Hash() }

// Store keeps profiles in a directory, <model>.<mode>.json, with the previous
// version of each next to it as .prev.
type Store struct{ Dir string }

func (s Store) path(model string, mode Mode) string {
	return filepath.Join(s.Dir, safeName(model)+"."+string(mode)+".json")
}

func safeName(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '/' || r == '\\' || r == ':' || r == 0 {
			return '_'
		}
		return r
	}, s)
}

// Load reads the profile of a model and mode; os.ErrNotExist when there is none.
func (s Store) Load(model string, mode Mode) (*Profile, error) {
	b, err := os.ReadFile(s.path(model, mode))
	if err != nil {
		return nil, err
	}
	var p Profile
	if err := json.Unmarshal(b, &p); err != nil {
		return nil, fmt.Errorf("tune: %s: %w", s.path(model, mode), err)
	}
	return &p, nil
}

// Save writes the profile, moving the old file to .prev. The write is atomic.
func (s Store) Save(p *Profile) error {
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return err
	}
	path := s.path(p.Key.Model, p.Mode)
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(s.Dir, ".profile-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	if _, err := tmp.Write(append(b, '\n')); err != nil {
		_ = tmp.Close()
		_ = os.Remove(name)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(name)
		return err
	}
	if _, err := os.Stat(path); err == nil {
		_ = os.Remove(path + ".prev")
		if err := os.Rename(path, path+".prev"); err != nil {
			_ = os.Remove(name)
			return err
		}
	}
	if err := os.Rename(name, path); err != nil {
		_ = os.Remove(name)
		return err
	}
	return nil
}

// LoadPrev reads the previous version of a profile.
func (s Store) LoadPrev(model string, mode Mode) (*Profile, error) {
	b, err := os.ReadFile(s.path(model, mode) + ".prev")
	if err != nil {
		return nil, err
	}
	var p Profile
	return &p, json.Unmarshal(b, &p)
}
