// Package llamacpp is the core backend: llama.cpp on CPU and GPU on every
// platform (SPEC §3).
//
// It binds libllama through purego, so the binary is built with CGO_ENABLED=0
// and cross-compiles from one runner. ggml compute backends (CPU variants,
// Vulkan, CUDA, Metal, …) are separate libraries in lib/ loaded at runtime; a
// missing one is skipped, never fatal.
//
// The binding is generated against llama.h of one pinned release tag. At
// startup the library's build number is checked against that tag and a
// mismatch is refused with a message, because a C ABI drift between tags does
// not fail loudly — it corrupts memory.
//
// Status: the binding lands with v0.1. Until then the backend registers, finds
// its library, and refuses to load models with ErrNotImplemented, so
// `nanollm devices` can already tell whether scripts/fetch-libs.sh put the
// libraries where the binary looks for them.
package llamacpp

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/usunrise88/nanollm/internal/backend"
)

// Name is the registered backend name and the prefix of its device selectors.
const Name = "llamacpp"

// MinTag is the oldest llama.cpp release that can run the primary
// configuration: Gemma 4 MTP landed in ggml-org/llama.cpp#23398, and E2B/E4B
// drafter conversion was fixed in #24282, which shipped in b9568.
//
// The pinned tag (scripts/llama.cpp.tag) must be at or above it. A later tag
// is preferable — quantized-KV and memory-fitting bugs around MTP were fixed
// after it — but which one is decided by the golden tests, not here.
const MinTag = "b9568"

// PinnedTag is the llama.cpp release this binary's binding was generated
// against, stamped by the linker from scripts/llama.cpp.tag:
// -ldflags "-X github.com/usunrise88/nanollm/internal/backend/llamacpp.PinnedTag=b…".
var PinnedTag = "unpinned"

func init() {
	backend.Register(Name, open)
}

type llamaBackend struct {
	lib string
}

func open(libDir string) (backend.Backend, error) {
	lib := filepath.Join(libDir, backend.SharedLibrary("llama"))
	if _, err := os.Stat(lib); err != nil {
		return nil, fmt.Errorf("%s not found; run scripts/fetch-libs.sh, or point backends.lib_dir at it", lib)
	}
	return &llamaBackend{lib: lib}, nil
}

func (b *llamaBackend) Name() string    { return Name }
func (b *llamaBackend) Version() string { return PinnedTag }

func (b *llamaBackend) Devices(context.Context) ([]backend.Device, error) {
	return nil, fmt.Errorf("llamacpp: device probing: %w (v0.1)", backend.ErrNotImplemented)
}

func (b *llamaBackend) Load(context.Context, backend.LoadSpec) (backend.Model, error) {
	return nil, fmt.Errorf("llamacpp: model loading: %w (v0.1)", backend.ErrNotImplemented)
}

func (b *llamaBackend) Close() error { return nil }
