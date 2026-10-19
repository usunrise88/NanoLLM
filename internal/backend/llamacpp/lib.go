package llamacpp

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/ebitengine/purego"

	"github.com/usunrise88/nanollm/internal/backend"
)

// A process holds one copy of libllama: the libraries are loaded by absolute
// path once, from the first lib_dir asked for, and stay loaded. ggml keeps
// global state (the backend registry, the log callback), and purego
// callbacks are never freed, so there is nothing to gain from unloading.

// cSymbol and cxxSymbol pair a generated function variable with the name the
// loader looks up (zz_generated.go lists them).
type cSymbol struct {
	name string
	fn   any
}

type cxxSymbol struct {
	name    string
	itanium string // Linux, macOS
	msvc    string // Windows (clang-cl)
	fn      any
}

// library is the loaded engine.
type library struct {
	dir     string
	handles []uintptr // ggml-base, ggml, llama
	// commit is what ggml_commit() reports: the short hash of the build.
	commit string

	initOnce sync.Once
	initErr  error
	// cxx records which staging-header functions the library exports.
	cxx map[string]bool
	// cpuVariant is the ggml CPU library ggml picked for this machine, e.g.
	// "sapphirerapids" from libggml-cpu-sapphirerapids.so; empty when the
	// CPU backend is built in (macOS).
	cpuVariant string
	// warnings are backends present in lib/ that did not load, with the
	// reason (SPEC §3: a missing backend is never fatal).
	warnings []string
	// typesByName maps ggml type names (f16, q8_0) to their enum.
	typesByName map[string]GgmlType
}

var (
	libMu  sync.Mutex
	loaded *library
)

// loadLibrary loads ggml-base, ggml and llama from dir, in that order and by
// absolute path, and reads the build commit. Nothing else is bound or called
// yet: if the commit is not the pinned one, the binding's view of every
// struct may be wrong, and the caller refuses before calling anything else.
func loadLibrary(dir string) (*library, error) {
	libMu.Lock()
	defer libMu.Unlock()
	if !filepath.IsAbs(dir) {
		return nil, fmt.Errorf("lib_dir %q is not absolute", dir)
	}
	dir = filepath.Clean(dir)
	if loaded != nil {
		if loaded.dir != dir {
			return nil, fmt.Errorf("llama.cpp is already loaded from %s; one process uses one lib_dir (asked for %s)", loaded.dir, dir)
		}
		return loaded, nil
	}
	if err := prepareLibDir(dir); err != nil {
		return nil, err
	}
	l := &library{dir: dir}
	for _, base := range []string{"ggml-base", "ggml", "llama"} {
		path := filepath.Join(dir, backend.SharedLibrary(base))
		if _, err := os.Stat(path); err != nil {
			return nil, fmt.Errorf("%s not found; run scripts/fetch-libs.sh, or point backends.lib_dir at it", path)
		}
		if err := checkLoaderPaths(path); err != nil {
			return nil, err
		}
		h, err := openLibrary(path)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", path, err)
		}
		l.handles = append(l.handles, h)
	}
	if err := l.bind(cSymbol{"ggml_commit", &ggmlCommit}); err != nil {
		return nil, err
	}
	l.commit = ggmlCommit()
	loaded = l
	return l, nil
}

// lookup finds a symbol in any of the loaded libraries. On Windows
// GetProcAddress does not search a DLL's dependencies, so each is asked.
func (l *library) lookup(name string) (uintptr, error) {
	for i := len(l.handles) - 1; i >= 0; i-- {
		if p, err := symbol(l.handles[i], name); err == nil && p != 0 {
			return p, nil
		}
	}
	return 0, fmt.Errorf("symbol %s not found in %s", name, l.dir)
}

func (l *library) bind(s cSymbol) error {
	p, err := l.lookup(s.name)
	if err != nil {
		return err
	}
	purego.RegisterFunc(s.fn, p)
	return nil
}

// checkCommit compares the build the library reports with the one the binding
// was generated for. ggml_commit() returns an abbreviated hash (9 characters
// at b11600) and the pin holds the full one, so either may be a prefix of the
// other, but never shorter than 7 characters.
func checkCommit(reported, pinned string) error {
	r, p := strings.ToLower(strings.TrimSpace(reported)), strings.ToLower(strings.TrimSpace(pinned))
	if len(r) >= 7 && len(p) >= 7 && (strings.HasPrefix(p, r) || strings.HasPrefix(r, p)) {
		return nil
	}
	return fmt.Errorf("libllama in this lib_dir was built from llama.cpp commit %q, but this binary's binding was generated for %s (%s); "+
		"a different build can change struct layouts and corrupt memory, so it is refused. "+
		"Run scripts/fetch-libs.sh to install the pinned libraries", reported, PinnedTag, pinned)
}

// init binds the rest of the API and starts the engine: called once, after
// the commit check passed.
func (l *library) init() error {
	l.initOnce.Do(func() { l.initErr = l.doInit() })
	return l.initErr
}

func (l *library) doInit() error {
	var missing []string
	for _, s := range cSymbols {
		if s.name == "ggml_commit" {
			continue
		}
		if err := l.bind(s); err != nil {
			missing = append(missing, s.name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("libllama in %s lacks %s; it is not the build this binary was generated for", l.dir, strings.Join(missing, ", "))
	}
	l.cxx = map[string]bool{}
	for _, s := range cxxSymbols {
		name := s.itanium
		if runtime.GOOS == "windows" {
			name = s.msvc
		}
		p, err := l.lookup(name)
		if err != nil {
			continue
		}
		purego.RegisterFunc(s.fn, p)
		l.cxx[s.name] = true
	}

	installLogger()
	// ggml reports the CPU variant it picked only in its log.
	cpu := regexp.MustCompile(`loaded CPU backend from (.+)$`)
	untap := addLogTap(func(_ GgmlLogLevel, line string) {
		if m := cpu.FindStringSubmatch(line); m != nil {
			l.cpuVariant = cpuVariantOf(m[1])
		}
	})
	// The backend libraries (CPU variants, Vulkan, CUDA, …) are found by
	// scanning this directory, not the search path. On macOS they are linked
	// into libggml and this does nothing.
	ggmlBackendLoadAllFromPath(l.dir)
	llamaBackendInit()
	initNUMA()
	untap()

	l.typesByName = map[string]GgmlType{}
	for t := GgmlType(0); t < GgmlTypeCount; t++ {
		if name := ggmlTypeName(t); name != "" && name != "NONE" {
			l.typesByName[strings.ToLower(name)] = t
		}
	}

	regs := map[string]bool{}
	for i := uintptr(0); i < ggmlBackendRegCount(); i++ {
		regs[strings.ToLower(ggmlBackendRegName(ggmlBackendRegGet(i)))] = true
	}
	if !regs["cpu"] {
		return fmt.Errorf("no ggml CPU backend loaded from %s; the CPU libraries are missing or none fits this processor", l.dir)
	}
	l.warnings = probeBackends(l.dir, regs)
	for _, w := range l.warnings {
		slog.Warn("llama.cpp backend not loaded", "detail", w)
	}
	return nil
}

// cpuVariantOf turns ".../libggml-cpu-haswell.so" into "haswell".
func cpuVariantOf(path string) string {
	base := filepath.Base(strings.TrimSpace(path))
	base = strings.TrimSuffix(strings.TrimSuffix(strings.TrimSuffix(base, ".so"), ".dll"), ".dylib")
	base = strings.TrimPrefix(base, "lib")
	return strings.TrimPrefix(strings.TrimPrefix(base, "ggml-cpu"), "-")
}

// backendLib matches a ggml backend library file name and captures the
// backend: libggml-vulkan.so, ggml-cuda.dll. The CPU variants are
// alternatives of which ggml loads only the best one, so they are not listed.
var backendLib = regexp.MustCompile(`^(?:lib)?ggml-([a-z0-9]+)\.(?:so|dll)$`)

// probeBackends names each backend library in dir that did not register,
// with the loader's reason. ggml_backend_load_all_from_path skips a library
// that fails to load without saying why; dlopen-ing it again here recovers
// the error (typically a missing libvulkan.so.1 or CUDA driver).
func probeBackends(dir string, regs map[string]bool) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return []string{fmt.Sprintf("read %s: %v", dir, err)}
	}
	var out []string
	for _, e := range entries {
		m := backendLib.FindStringSubmatch(e.Name())
		if m == nil || m[1] == "base" || m[1] == "cpu" {
			continue
		}
		if regs[m[1]] {
			continue
		}
		path := filepath.Join(dir, e.Name())
		if err := tryOpen(path); err != nil {
			out = append(out, fmt.Sprintf("%s: %v", e.Name(), err))
		} else {
			out = append(out, fmt.Sprintf("%s: loads, but registered no backend (no usable device or driver)", e.Name()))
		}
	}
	sort.Strings(out)
	return out
}
