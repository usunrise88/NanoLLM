package core

import "context"

// Admin is model and device management (SPEC §6, native dialect, admin keys
// only). Like Service, it is the dialects' only way to reach the registry, the
// pool and the backends.
type Admin interface {
	// Models lists the catalog with what is on disk and what is loaded.
	Models(ctx context.Context) ([]ModelInfo, error)
	// Download makes a model complete on disk. Concurrent calls for one model
	// share the download; progress is called with each update.
	Download(ctx context.Context, id string, progress func(DownloadProgress)) error
	// Load makes a model resident with its configured placement, Unload
	// releases it (after running requests), Pin keeps it resident through
	// eviction, and Reload loads it afresh and swaps it in.
	Load(ctx context.Context, id string) (*ModelInfo, error)
	Unload(ctx context.Context, id string) error
	Pin(ctx context.Context, id string, pin bool) error
	Reload(ctx context.Context, id string) (*ModelInfo, error)
	Devices(ctx context.Context) ([]DeviceInfo, error)
}

// ModelInfo is one catalog entry as GET /api/v1/models reports it.
type ModelInfo struct {
	ID     string `json:"id"`
	Family string `json:"family,omitempty"`
	Params string `json:"params,omitempty"`
	Quant  string `json:"quant,omitempty"`
	// Disk is absent, partial, complete or corrupt.
	Disk      string `json:"disk"`
	SizeBytes int64  `json:"size_bytes"`
	Pinned    bool   `json:"pinned"`
	// Loaded lists the resident instances; Plan is the placement the model
	// loads with by default.
	Loaded []any `json:"loaded,omitempty"`
	Plan   any   `json:"plan,omitempty"`
	// Default marks models.default.
	Default bool `json:"default,omitempty"`
}

// DownloadProgress is one update of a model download.
type DownloadProgress struct {
	File       string  `json:"file"`
	Downloaded int64   `json:"downloaded"`
	Total      int64   `json:"total"`
	Percent    float64 `json:"percent"`
}

// DeviceInfo is one device of one backend.
type DeviceInfo struct {
	Selector    string `json:"selector"`
	Backend     string `json:"backend"`
	ID          string `json:"id"`
	Type        string `json:"type"`
	Name        string `json:"name"`
	MemoryTotal int64  `json:"memory_total"`
	MemoryFree  int64  `json:"memory_free"`
	Unified     bool   `json:"unified,omitempty"`
	// Verified lists the models whose golden tests passed on this device.
	Verified []string `json:"verified,omitempty"`
}
