//go:build windows

package llamacpp

import (
	"fmt"
	"sync"

	"golang.org/x/sys/windows"
)

var dllDirOnce sync.Once

// prepareLibDir restricts DLL resolution for the whole process to the system
// directories plus lib/ (SPEC §10: AddDllDirectory on lib/ before loading).
// Without it, a dependency such as ggml-base.dll resolves through the
// current directory and PATH first, where another llama.cpp build may sit.
// The vendor runtimes ggml loads later (vulkan-1.dll, nvcuda.dll) live in the
// system directory, which stays on the list.
func prepareLibDir(dir string) error {
	var err error
	dllDirOnce.Do(func() {
		if e := windows.SetDefaultDllDirectories(windows.LOAD_LIBRARY_SEARCH_DEFAULT_DIRS); e != nil {
			err = fmt.Errorf("SetDefaultDllDirectories: %w", e)
			return
		}
		p, e := windows.UTF16PtrFromString(dir)
		if e != nil {
			err = e
			return
		}
		if _, e := windows.AddDllDirectory(p); e != nil {
			err = fmt.Errorf("AddDllDirectory %s: %w", dir, e)
		}
	})
	return err
}

// openLibrary loads by absolute path; LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR lets
// the DLL's own directory satisfy its imports.
func openLibrary(path string) (uintptr, error) {
	h, err := windows.LoadLibraryEx(path, 0,
		windows.LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|windows.LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
	return uintptr(h), err
}

func symbol(h uintptr, name string) (uintptr, error) {
	return windows.GetProcAddress(windows.Handle(h), name)
}

func tryOpen(path string) error {
	_, err := openLibrary(path)
	return err
}
