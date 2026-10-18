//go:build !windows

package llamacpp

import "github.com/ebitengine/purego"

// prepareLibDir has nothing to do on Unix: each library is opened by absolute
// path, and its dependencies resolve next to it — through RUNPATH=$ORIGIN on
// Linux and @rpath with LC_RPATH @loader_path on macOS (checkLoaderPaths).
// ggml-base is loaded first, so a later DT_NEEDED on libggml-base.so.0 binds
// to the copy already in the process by its soname.
func prepareLibDir(string) error { return nil }

// openLibrary loads with RTLD_GLOBAL: the ggml backend libraries that ggml
// later dlopens resolve ggml-base symbols against the copy loaded here.
func openLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

func symbol(h uintptr, name string) (uintptr, error) {
	return purego.Dlsym(h, name)
}

// tryOpen reports why a library does not load. It stays loaded if it does,
// which is harmless: the same file was already loadable by ggml.
func tryOpen(path string) error {
	_, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	return err
}
