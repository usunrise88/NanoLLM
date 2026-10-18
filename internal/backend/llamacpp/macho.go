package llamacpp

import (
	"debug/macho"
	"fmt"
	"runtime"
	"strings"
)

// checkLoaderPaths makes sure a macOS dylib resolves its llama.cpp
// dependencies next to itself (SPEC §10: dylibs use @loader_path). A
// dependency named by an absolute path outside the system directories, or an
// @rpath without an @loader_path run path, would let dyld pick up another
// copy of ggml — from Homebrew, say — and mix two builds in one process.
// Elsewhere it does nothing: ELF has RUNPATH=$ORIGIN, PE has the DLL
// directory.
func checkLoaderPaths(path string) error {
	if runtime.GOOS != "darwin" {
		return nil
	}
	return checkMachO(path)
}

// checkMachO is checkLoaderPaths without the platform test, so it can be
// tested on any host against the macOS archive.
func checkMachO(path string) error {
	var files []*macho.File
	if fat, err := macho.OpenFat(path); err == nil {
		defer func() { _ = fat.Close() }()
		for _, a := range fat.Arches {
			files = append(files, a.File)
		}
	} else {
		f, err := macho.Open(path)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		defer func() { _ = f.Close() }()
		files = append(files, f)
	}
	for _, f := range files {
		loaderRpath := false
		var deps []string
		for _, l := range f.Loads {
			switch c := l.(type) {
			case *macho.Rpath:
				if strings.HasPrefix(c.Path, "@loader_path") {
					loaderRpath = true
				}
			case *macho.Dylib:
				deps = append(deps, c.Name)
			}
		}
		// debug/macho reports LC_LOAD_DYLIB as *macho.Dylib; the library's
		// own LC_ID_DYLIB is not among them.
		for _, d := range deps {
			switch {
			case strings.HasPrefix(d, "/usr/lib/"), strings.HasPrefix(d, "/System/"):
			case strings.HasPrefix(d, "@loader_path/"):
			case strings.HasPrefix(d, "@rpath/"):
				if !loaderRpath {
					return fmt.Errorf("%s loads %s but has no @loader_path run path; it could resolve outside lib/", path, d)
				}
			default:
				return fmt.Errorf("%s loads %s by a path outside lib/", path, d)
			}
		}
	}
	return nil
}
