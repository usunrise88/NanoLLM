package backend

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// ResolveLibDir turns backends.lib_dir into an absolute path. A relative path
// is taken relative to the binary, never to the working directory: native
// libraries are always loaded by absolute path from <binary dir>/lib, so that
// whatever happens to be on the system search path cannot shadow the pinned
// build (SPEC §10).
func ResolveLibDir(dir string) (string, error) {
	if filepath.IsAbs(dir) {
		return filepath.Clean(dir), nil
	}
	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("locate the binary to resolve lib_dir %q: %w", dir, err)
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		exe = resolved
	}
	return filepath.Join(filepath.Dir(exe), dir), nil
}

// SharedLibrary is the file name of library base on this platform:
// libllama.so, libllama.dylib or llama.dll.
func SharedLibrary(base string) string {
	switch runtime.GOOS {
	case "windows":
		return base + ".dll"
	case "darwin":
		return "lib" + base + ".dylib"
	default:
		return "lib" + base + ".so"
	}
}
