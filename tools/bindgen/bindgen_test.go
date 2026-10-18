package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const testHeader = `
#ifndef X_H
#define X_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
    typedef int32_t llama_token;
    typedef struct ggml_backend_device * ggml_backend_dev_t;
    typedef bool (*cb_t)(float progress, void * user_data);
    struct llama_model;
    struct llama_context;

    enum flags {
        F_A = 1 << 0,
        F_B = 1 << 3,
        F_AB = F_A | F_B,
    };
    enum neg { N_AUTO = -1, N_OFF, N_ON };

    struct inner { bool a; bool b; };

    // A comment with a ; and a { brace.
    struct params {
        ggml_backend_dev_t * devices;
        int32_t n;          /* trailing */
        enum neg mode;
        uint8_t small;
        double d;
        const char * name;
        cb_t cb;
        struct inner in;
        char key[16];
        bool last;
    };

    typedef struct pair { llama_token t; float bias; } pair;

    struct with_union { int tag; union { int64_t i; double f; }; };

    LLAMA_API struct params default_params(void);
    LLAMA_API struct llama_model * load(const char * path, struct params p);
    LLAMA_API int32_t tokenize(const struct llama_model * m, const char * text, llama_token * out, bool special);
    DEPRECATED(LLAMA_API int32_t old_api(void), "use new_api");
#ifdef __cplusplus
}
#endif
#endif
`

const testCxxHeader = `
#pragma once
#include "x.h"
#include <vector>
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);
LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);
LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);
LLAMA_API void two(llama_context * a, llama_context * b);
LLAMA_API void twoconst(const llama_model * a, const llama_model * b, uint32_t n);
LLAMA_API std::vector<int> cxx_only(const llama_model * m);
struct cxx_struct {
    size_t a = 0;
    size_t total() const { return a; }
};
`

func parseTest(t *testing.T) *header {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "x.h"), []byte(testHeader), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "x-ext.h"), []byte(testCxxHeader), 0o644); err != nil {
		t.Fatal(err)
	}
	h := newHeader()
	if err := h.parseFile(filepath.Join(dir, "x.h"), false); err != nil {
		t.Fatal(err)
	}
	if err := h.parseFile(filepath.Join(dir, "x-ext.h"), true); err != nil {
		t.Fatal(err)
	}
	return h
}

func TestParseEnums(t *testing.T) {
	h := parseTest(t)
	for name, want := range map[string]int64{"F_A": 1, "F_B": 8, "F_AB": 9, "N_AUTO": -1, "N_OFF": 0, "N_ON": 1} {
		if got, ok := h.consts[name]; !ok || got != want {
			t.Errorf("%s = %d (%v), want %d", name, got, ok, want)
		}
	}
}

func TestLayoutPadding(t *testing.T) {
	h := parseTest(t)
	l, err := h.layout("params")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]int{"devices": 0, "n": 8, "mode": 12, "small": 16, "d": 24, "name": 32, "cb": 40, "in": 48, "key": 50, "last": 66}
	for _, f := range l.fields {
		if want[f.name] != f.offset {
			t.Errorf("offsetof(%s) = %d, want %d", f.name, f.offset, want[f.name])
		}
	}
	if l.size != 72 || l.align != 8 {
		t.Errorf("size %d align %d, want 72 8", l.size, l.align)
	}
	if p, err := h.layout("pair"); err != nil || p.size != 8 {
		t.Errorf("typedef'd struct pair: %+v %v", p, err)
	}
	if _, err := h.layout("with_union"); err == nil {
		t.Error("a struct with a union got a layout")
	}
	if _, err := h.layout("cxx_struct"); err == nil {
		t.Error("a C++ struct got a layout")
	}
}

func TestFunctions(t *testing.T) {
	h := parseTest(t)
	if _, ok := h.funcs["old_api"]; ok {
		t.Error("a deprecated function was parsed")
	}
	if _, ok := h.funcs["cxx_only"]; ok {
		t.Error("a function returning a C++ type was parsed")
	}
	f := h.funcs["tokenize"]
	if f == nil || len(f.params) != 4 || f.cxx {
		t.Fatalf("tokenize: %+v", f)
	}
	g := newGen(h, "bTEST", "0000000")
	g.wantStruct["params"] = true
	g.wantEnum["neg"] = true
	for i, want := range []string{"LlamaModel", "string", "*int32", "bool"} {
		got, err := g.goType(f.params[i].typ, inParam)
		if err != nil || got != want {
			t.Errorf("tokenize param %d: %q %v, want %q", i, got, err, want)
		}
	}
	l, _ := h.layout("params")
	for name, want := range map[string]string{"devices": "*GgmlBackendDevice", "name": "*byte", "cb": "uintptr", "mode": "Neg", "key": "[16]byte"} {
		for _, fl := range l.fields {
			if fl.name == name {
				got, err := g.goType(fl.typ, inField)
				if err != nil || got != want {
					t.Errorf("field %s: %q %v, want %q", name, got, err, want)
				}
			}
		}
	}
}

// The decorated names below are copied from the export tables of the
// b11600 libllama.so and llama.dll.
func TestMangle(t *testing.T) {
	h := parseTest(t)
	for name, want := range map[string][2]string{
		"llama_set_embeddings_nextn":     {"_Z26llama_set_embeddings_nextnP13llama_contextbb", "?llama_set_embeddings_nextn@@YAXPEAUllama_context@@_N1@Z"},
		"llama_get_embeddings_nextn_ith": {"_Z30llama_get_embeddings_nextn_ithP13llama_contexti", "?llama_get_embeddings_nextn_ith@@YAPEAMPEAUllama_context@@H@Z"},
		"llama_get_ctx_other":            {"_Z19llama_get_ctx_otherP13llama_context", "?llama_get_ctx_other@@YAPEAUllama_context@@PEAU1@@Z"},
		"llama_model_n_devices":          {"_Z21llama_model_n_devicesPK11llama_model", "?llama_model_n_devices@@YAHPEBUllama_model@@@Z"},
		"llama_model_get_device":         {"_Z22llama_model_get_devicePK11llama_modeli", "?llama_model_get_device@@YAPEAUggml_backend_device@@PEBUllama_model@@H@Z"},
		// Substitutions: a repeated pointer type is S0_ (S_ is the class).
		"two":      {"_Z3twoP13llama_contextS0_", "?two@@YAXPEAUllama_context@@0@Z"},
		"twoconst": {"_Z8twoconstPK11llama_modelS1_j", "?twoconst@@YAXPEBUllama_model@@0I@Z"},
	} {
		f := h.funcs[name]
		if f == nil || !f.cxx {
			t.Fatalf("%s not parsed as C++", name)
		}
		it, err := h.itanium(f)
		if err != nil || it != want[0] {
			t.Errorf("%s itanium: %s %v, want %s", name, it, err, want[0])
		}
		ms, err := h.msvc(f)
		if err != nil || ms != want[1] {
			t.Errorf("%s msvc: %s %v, want %s", name, ms, err, want[1])
		}
	}
}

func TestCamel(t *testing.T) {
	for in, want := range map[string]string{
		"llama_model_params": "LlamaModelParams", "GGML_TYPE_Q8_0": "GgmlTypeQ8_0", "n_gpu_layers": "NGpuLayers",
		"GGML_TYPE_F16": "GgmlTypeF16", "LLAMA_LOAD_MODE_MMAP_MLOCK": "LlamaLoadModeMmapMlock",
	} {
		if got := camel(in, true); got != want {
			t.Errorf("%s: %s, want %s", in, got, want)
		}
	}
	if got := camel("llama_tokenize", false); got != "llamaTokenize" {
		t.Errorf("unexported: %s", got)
	}
}

// TestGeneratedIsCurrent regenerates from a llama.cpp checkout when one is
// given (NANOLLM_LLAMA_SRC) and compares with the committed files.
func TestGeneratedIsCurrent(t *testing.T) {
	src := os.Getenv("NANOLLM_LLAMA_SRC")
	if src == "" {
		t.Skip("set NANOLLM_LLAMA_SRC to a llama.cpp checkout of the pinned tag")
	}
	root := filepath.Join("..", "..")
	read := func(p string) string {
		b, err := os.ReadFile(filepath.Join(root, p))
		if err != nil {
			t.Fatal(err)
		}
		return strings.TrimSpace(string(b))
	}
	h := newHeader()
	for _, hd := range headers {
		if err := h.parseFile(filepath.Join(src, hd.path), hd.cxx); err != nil {
			t.Fatal(err)
		}
	}
	g := newGen(h, read("scripts/llama.cpp.tag"), read("scripts/llama.cpp.commit"))
	goSrc, err := g.goFile()
	if err != nil {
		t.Fatal(err)
	}
	if string(goSrc) != read("internal/backend/llamacpp/zz_generated.go")+"\n" {
		t.Error("zz_generated.go is stale; run go run ./tools/bindgen -src " + src)
	}
	_ = runtime.GOOS
}
