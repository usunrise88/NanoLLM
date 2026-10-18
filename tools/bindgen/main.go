// Command bindgen generates the purego binding of libllama from the headers of
// the pinned llama.cpp release (SPEC §10: bumping llama.cpp means changing one
// tag, regenerating the binding and passing the golden tests).
//
//	go run ./tools/bindgen -src /path/to/llama.cpp            # regenerate
//	go run ./tools/bindgen -src /path/to/llama.cpp -check-exports dist/linux_amd64/lib
//
// -src is a checkout of the tag in scripts/llama.cpp.tag. The generator writes
//
//	internal/backend/llamacpp/zz_generated.go       structs, enums, function variables
//	internal/backend/llamacpp/zz_generated_test.go  Go offsets == computed C offsets
//	internal/backend/llamacpp/testdata/layout.c     _Static_assert per target
//
// and refuses to: bind a struct by value above 208 bytes, use a type whose
// size differs between the targets, or bind a name the headers lack.
//
// -check-exports reads the export table of the libraries in each directory
// (ELF, PE or Mach-O, whatever the directory holds) and checks that every
// bound C function and the platform's decorated name of every staging-header
// function is exported. It runs on any host, so one Linux job checks the
// Windows and macOS archives too.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func main() {
	src := flag.String("src", "", "llama.cpp checkout at the pinned tag")
	tag := flag.String("tag", "", "llama.cpp tag (default: scripts/llama.cpp.tag)")
	commit := flag.String("commit", "", "llama.cpp commit (default: scripts/llama.cpp.commit)")
	out := flag.String("out", "internal/backend/llamacpp", "package directory to write")
	var checks multiFlag
	flag.Var(&checks, "check-exports", "library directory whose export tables to check (repeatable); skips generation")
	flag.Parse()
	if err := run(*src, *tag, *commit, *out, checks); err != nil {
		fmt.Fprintln(os.Stderr, "bindgen:", err)
		os.Exit(1)
	}
}

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

func run(src, tag, commit, out string, checks []string) error {
	if src == "" {
		return errors.New("-src is required: a llama.cpp checkout at the pinned tag")
	}
	var err error
	if tag == "" {
		if tag, err = readTrimmed("scripts/llama.cpp.tag"); err != nil {
			return err
		}
	}
	if commit == "" {
		if commit, err = readTrimmed("scripts/llama.cpp.commit"); err != nil {
			return err
		}
	}
	h := newHeader()
	for _, hd := range headers {
		if err := h.parseFile(filepath.Join(src, hd.path), hd.cxx); err != nil {
			return err
		}
	}
	g := newGen(h, tag, commit)
	if len(checks) > 0 {
		for _, dir := range checks {
			if err := checkExports(g, dir); err != nil {
				return err
			}
		}
		return nil
	}
	goSrc, err := g.goFile()
	if err != nil {
		return err
	}
	testSrc, err := g.layoutTestFile()
	if err != nil {
		return err
	}
	cSrc, err := g.cLayoutFile()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(out, "testdata"), 0o755); err != nil {
		return err
	}
	for name, data := range map[string][]byte{
		"zz_generated.go":      goSrc,
		"zz_generated_test.go": testSrc,
		"testdata/layout.c":    cSrc,
	} {
		if err := os.WriteFile(filepath.Join(out, name), data, 0o644); err != nil {
			return err
		}
	}
	for _, name := range allowStructs {
		l, _ := h.layout(name)
		fmt.Printf("%-34s %4d B\n", name, l.size)
	}
	fmt.Printf("%d functions, %d staging functions, %d structs, %d enums\n",
		len(allowFuncs), len(allowCxx), len(allowStructs), len(allowEnums))
	return nil
}

// checkExports checks one library directory. The decoration is chosen by
// the file format: PE means MSVC, anything else Itanium.
func checkExports(g *gen, dir string) error {
	all := map[string]bool{}
	var llama map[string]bool
	msvc := false
	for _, base := range []string{"ggml-base", "ggml", "llama"} {
		var path string
		for _, cand := range []string{"lib" + base + ".so", "lib" + base + ".dylib", base + ".dll"} {
			if _, err := os.Stat(filepath.Join(dir, cand)); err == nil {
				path = filepath.Join(dir, cand)
				msvc = strings.HasSuffix(cand, ".dll")
				break
			}
		}
		if path == "" {
			return fmt.Errorf("%s: no %s library", dir, base)
		}
		syms, err := exports(path)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		for s := range syms {
			all[s] = true
		}
		if base == "llama" {
			llama = syms
		}
	}
	var missing []string
	for _, name := range allowFuncs {
		if !all[name] {
			missing = append(missing, name)
		}
	}
	for _, name := range allowCxx {
		f, ok := g.h.funcs[name]
		if !ok {
			return g.missing("C++ function", name)
		}
		var sym string
		var err error
		if msvc {
			sym, err = g.h.msvc(f)
		} else {
			sym, err = g.h.itanium(f)
		}
		if err != nil {
			return err
		}
		if !llama[sym] {
			missing = append(missing, sym)
		} else {
			fmt.Printf("%s: %s\n", dir, sym)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("%s: not exported: %s", dir, strings.Join(missing, " "))
	}
	fmt.Printf("%s: all %d C and %d decorated symbols exported\n", dir, len(allowFuncs), len(allowCxx))
	return nil
}

func readTrimmed(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("%w (run from the repository root, or pass the value)", err)
	}
	return strings.TrimSpace(string(b)), nil
}
