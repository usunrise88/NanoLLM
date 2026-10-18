// Package registry knows which models exist, where their weights come from and
// whether the bytes on disk are the bytes that were pinned (SPEC §7).
//
// Weights are never shipped. Every catalog entry is pinned by repository
// commit, size and SHA-256, and a download is verified before use. An entry
// that is not pinned yet is listed but cannot be downloaded: fetching an
// unverified multi-gigabyte file and loading it into a native library is not
// something to do on the strength of a file name.
package registry

import (
	"fmt"
	"regexp"
)

// Formats.
const (
	FormatGGUF     = "gguf"
	FormatORTGenAI = "ortgenai"
)

// File roles inside one entry.
const (
	RoleModel   = "model"
	RoleDrafter = "drafter"
)

// Commercial-use status, honoured by registry.strict_license.
const (
	CommercialPermitted   = "permitted"
	CommercialUnconfirmed = "unconfirmed"
)

// Manifest describes one model. The catalog is a list of manifests;
// `nanollm models inspect` drafts one for a local GGUF.
type Manifest struct {
	ID     string `yaml:"id" json:"id"`
	Family string `yaml:"family" json:"family"`
	// Arch is the GGUF architecture llama.cpp loads it as (gemma4, qwen35).
	Arch   string `yaml:"arch" json:"arch"`
	Params string `yaml:"params" json:"params"`
	Quant  string `yaml:"quant" json:"quant"`
	Format string `yaml:"format" json:"format"`
	Role   string `yaml:"role" json:"role"` // primary | second | fallback

	Files []File `yaml:"files" json:"files"`

	ContextMax int `yaml:"context_max,omitempty" json:"context_max,omitempty"`
	// ChatTemplate overrides the template in the GGUF metadata; empty means
	// use the model's own.
	ChatTemplate     string            `yaml:"chat_template,omitempty" json:"chat_template,omitempty"`
	Reasoning        *Reasoning        `yaml:"reasoning,omitempty" json:"reasoning,omitempty"`
	ToolParser       string            `yaml:"tool_parser,omitempty" json:"tool_parser,omitempty"`
	SamplingDefaults *SamplingDefaults `yaml:"sampling_defaults,omitempty" json:"sampling_defaults,omitempty"`

	Drafter *Drafter `yaml:"drafter,omitempty" json:"drafter,omitempty"`

	License    string `yaml:"license" json:"license"`
	Commercial string `yaml:"commercial" json:"commercial"`

	// Verified lists backend/device pairs that passed the golden tests
	// (SPEC §12). NPU is offered only for a verified pair.
	Verified []string `yaml:"verified,omitempty" json:"verified,omitempty"`
	// Tensors is the measured inventory the planner and range-lock work from.
	// It is filled at catalog build by `nanollm models inspect --tensors`.
	Tensors []Tensor `yaml:"tensors,omitempty" json:"tensors,omitempty"`
	// Presets the entry has been planned and benchmarked under.
	Presets []PresetPlan `yaml:"presets,omitempty" json:"presets,omitempty"`
}

// File is one download. Revision is the repository commit the file is pinned
// to; Size and SHA256 are the bytes expected at that commit.
type File struct {
	Role     string `yaml:"role" json:"role"`
	Repo     string `yaml:"repo" json:"repo"`
	Path     string `yaml:"path" json:"path"`
	Revision string `yaml:"revision" json:"revision"`
	Size     int64  `yaml:"size" json:"size"`
	SHA256   string `yaml:"sha256" json:"sha256"`
	// ApproxSize is the size from the model card, for display until the entry
	// is pinned and Size is exact.
	ApproxSize string `yaml:"approx_size,omitempty" json:"approx_size,omitempty"`
}

var (
	hexSHA256 = regexp.MustCompile(`^[0-9a-f]{64}$`)
	hexCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// Pinned reports whether the file can be downloaded and verified.
func (f File) Pinned() bool {
	return hexCommit.MatchString(f.Revision) && hexSHA256.MatchString(f.SHA256) && f.Size > 0
}

// URL is where the file is fetched from at its pinned revision.
func (f File) URL() (string, error) {
	if !f.Pinned() {
		return "", fmt.Errorf("%s/%s is not pinned to a commit, size and sha256 yet", f.Repo, f.Path)
	}
	return fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s", f.Repo, f.Revision, f.Path), nil
}

type Reasoning struct {
	ThinkingDefault bool   `yaml:"thinking_default" json:"thinking_default"`
	OpenTag         string `yaml:"open_tag,omitempty" json:"open_tag,omitempty"`
	CloseTag        string `yaml:"close_tag,omitempty" json:"close_tag,omitempty"`
}

// SamplingDefaults are the model card's recommended values. A request that
// sets nothing gets these, not the sampler's zero values.
type SamplingDefaults struct {
	Temperature   *float64 `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	TopP          *float64 `yaml:"top_p,omitempty" json:"top_p,omitempty"`
	TopK          *int     `yaml:"top_k,omitempty" json:"top_k,omitempty"`
	MinP          *float64 `yaml:"min_p,omitempty" json:"min_p,omitempty"`
	RepeatPenalty *float64 `yaml:"repeat_penalty,omitempty" json:"repeat_penalty,omitempty"`
}

// Drafter describes the MTP drafter that ships with an entry.
type Drafter struct {
	Arch string `yaml:"arch" json:"arch"` // gemma4-assistant
	// SpecDraftNMax is the recommended draft length per device class
	// (cpu, cuda, vulkan, metal), filled from the O3 sweeps.
	SpecDraftNMax map[string]int `yaml:"spec_draft_n_max,omitempty" json:"spec_draft_n_max,omitempty"`
}

// Tensor is one measured tensor: its placement group (G0–G7), type, size and
// offset in the file.
type Tensor struct {
	Name   string `yaml:"name" json:"name"`
	Group  string `yaml:"group" json:"group"`
	Type   string `yaml:"type" json:"type"`
	Bytes  int64  `yaml:"bytes" json:"bytes"`
	Offset int64  `yaml:"offset" json:"offset"`
}

// PresetPlan records the mode a preset resolved to when it was measured.
type PresetPlan struct {
	Preset string `yaml:"preset" json:"preset"`
	Mode   string `yaml:"mode" json:"mode"`
}

// File returns the entry's file with the given role.
func (m Manifest) File(role string) (File, bool) {
	for _, f := range m.Files {
		if f.Role == role {
			return f, true
		}
	}
	return File{}, false
}

// Pinned reports whether every file of the entry can be downloaded.
func (m Manifest) Pinned() bool {
	if len(m.Files) == 0 {
		return false
	}
	for _, f := range m.Files {
		if !f.Pinned() {
			return false
		}
	}
	return true
}
