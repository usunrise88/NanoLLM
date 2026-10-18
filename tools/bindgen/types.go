package main

import (
	"fmt"
	"strings"
)

// Layout is computed for the 64-bit targets NanoLLM ships: linux, windows and
// darwin on amd64 and arm64. On all six, pointers and size_t are 8 bytes,
// bool is 1, enums are int-sized, and every scalar is aligned to its size, so
// one layout serves them all. The one type that differs between LP64 (Linux,
// macOS) and LLP64 (Windows) is long, and a struct that uses it is refused.
// The generated _Static_assert file checks this claim per target.

const ptrSize = 8

// maxByValue is the largest struct passed or returned by value. purego puts a
// by-value argument in at most 26 eight-byte words on SysV amd64; the bound
// leaves headroom over llama_context_params (168 B at b11600) and fails the
// generator, not the process, when upstream grows it.
const maxByValue = 208

var primSize = map[string]int{
	"bool": 1, "_Bool": 1, "char": 1, "signed char": 1, "unsigned char": 1,
	"int8_t": 1, "uint8_t": 1,
	"short": 2, "unsigned short": 2, "int16_t": 2, "uint16_t": 2,
	"int": 4, "unsigned": 4, "unsigned int": 4, "signed": 4, "signed int": 4,
	"int32_t": 4, "uint32_t": 4, "float": 4,
	"long long": 8, "unsigned long long": 8, "int64_t": 8, "uint64_t": 8,
	"size_t": 8, "intptr_t": 8, "uintptr_t": 8, "double": 8,
}

var primGo = map[string]string{
	"bool": "bool", "_Bool": "bool", "char": "byte", "signed char": "int8", "unsigned char": "uint8",
	"int8_t": "int8", "uint8_t": "uint8",
	"short": "int16", "unsigned short": "uint16", "int16_t": "int16", "uint16_t": "uint16",
	"int": "int32", "unsigned": "uint32", "unsigned int": "uint32", "signed": "int32", "signed int": "int32",
	"int32_t": "int32", "uint32_t": "uint32", "float": "float32",
	"long long": "int64", "unsigned long long": "uint64", "int64_t": "int64", "uint64_t": "uint64",
	"size_t": "uintptr", "intptr_t": "uintptr", "uintptr_t": "uintptr", "double": "float64",
}

// resolve follows typedef names to the type they stand for, keeping const.
func (h *header) resolve(t *cType) (*cType, error) {
	for i := 0; t.kind == kNamed && t.tag == "" && !isPrim(t.name); i++ {
		if i > 16 {
			return nil, fmt.Errorf("typedef loop at %s", t.name)
		}
		td, ok := h.typedefs[t.name]
		if !ok {
			// An unknown name in C++ code is a struct used without its tag
			// (llama-ext.h writes "llama_context *").
			return &cType{kind: kNamed, tag: "struct", name: t.name, konst: t.konst}, nil
		}
		r := *td.typ
		r.konst = r.konst || t.konst
		t = &r
	}
	return t, nil
}

// sizeAlign is the C size and alignment of t on the 64-bit targets.
func (h *header) sizeAlign(t *cType) (int, int, error) {
	t, err := h.resolve(t)
	if err != nil {
		return 0, 0, err
	}
	switch t.kind {
	case kPrim:
		if t.name == "long" || t.name == "unsigned long" {
			return 0, 0, fmt.Errorf("%s differs between LP64 and LLP64", t.name)
		}
		n, ok := primSize[t.name]
		if !ok {
			return 0, 0, fmt.Errorf("no size for %s", t.name)
		}
		return n, n, nil
	case kPointer, kFuncPtr:
		return ptrSize, ptrSize, nil
	case kArray:
		n, a, err := h.sizeAlign(t.elem)
		return n * t.n, a, err
	case kNamed:
		switch t.tag {
		case "enum":
			return 4, 4, nil
		case "struct":
			l, err := h.layout(t.name)
			if err != nil {
				return 0, 0, err
			}
			return l.size, l.align, nil
		}
	}
	return 0, 0, fmt.Errorf("no layout for %s", t)
}

type fieldLayout struct {
	cField
	offset int
	size   int
}

type structLayout struct {
	name   string
	size   int
	align  int
	fields []fieldLayout
}

func (h *header) layout(name string) (*structLayout, error) {
	s, ok := h.structs[name]
	if !ok {
		if err, broken := h.broken[name]; broken {
			return nil, fmt.Errorf("struct %s: %w", name, err)
		}
		return nil, fmt.Errorf("struct %s is not defined", name)
	}
	if s.union {
		return nil, fmt.Errorf("struct %s has a union or C++ members; bind it by pointer only", name)
	}
	l := &structLayout{name: name, align: 1}
	off := 0
	for _, f := range s.fields {
		n, a, err := h.sizeAlign(f.typ)
		if err != nil {
			return nil, fmt.Errorf("struct %s.%s: %w", name, f.name, err)
		}
		off = alignUp(off, a)
		l.fields = append(l.fields, fieldLayout{f, off, n})
		off += n
		l.align = max(l.align, a)
	}
	l.size = alignUp(off, l.align)
	return l, nil
}

func alignUp(n, a int) int { return (n + a - 1) / a * a }

// camel turns snake_case or SCREAMING_CASE into CamelCase. Two digit runs
// that would touch keep their underscore, so GGML_TYPE_Q8_0 is GgmlTypeQ8_0,
// not the ambiguous GgmlTypeQ80.
func camel(s string, exported bool) string {
	var b strings.Builder
	parts := strings.Split(strings.ToLower(s), "_")
	prevDigit := false
	for i, p := range parts {
		if p == "" {
			continue
		}
		if prevDigit && p[0] >= '0' && p[0] <= '9' {
			b.WriteByte('_')
		}
		if i == 0 && !exported {
			b.WriteString(p)
		} else {
			b.WriteString(strings.ToUpper(p[:1]) + p[1:])
		}
		last := p[len(p)-1]
		prevDigit = last >= '0' && last <= '9'
	}
	out := b.String()
	if exported && out != "" && out[0] >= '0' && out[0] <= '9' {
		out = "X" + out
	}
	return out
}

// goTypeCtx says where a type appears: struct fields hold raw pointers, while
// a function parameter of type const char * is a Go string that purego copies
// into a NUL-terminated buffer for the duration of the call.
type goTypeCtx int

const (
	inField goTypeCtx = iota
	inParam
	inResult
)

// goType maps a C type to its Go spelling for the generated binding.
func (g *gen) goType(t *cType, ctx goTypeCtx) (string, error) {
	r, err := g.h.resolve(t)
	if err != nil {
		return "", err
	}
	switch r.kind {
	case kPrim:
		if r.name == "void" {
			if ctx == inResult {
				return "", nil
			}
			return "", fmt.Errorf("void value")
		}
		if r.name == "long" || r.name == "unsigned long" {
			return "", fmt.Errorf("%s differs between LP64 and LLP64", r.name)
		}
		s, ok := primGo[r.name]
		if !ok {
			return "", fmt.Errorf("no Go type for %s", r.name)
		}
		return s, nil
	case kFuncPtr:
		return "uintptr", nil
	case kArray:
		e, err := g.goType(r.elem, inField)
		return fmt.Sprintf("[%d]%s", r.n, e), err
	case kNamed:
		switch r.tag {
		case "enum":
			if !g.wantEnum[r.name] {
				return "int32", nil
			}
			return camel(r.name, true), nil
		case "struct":
			if !g.wantStruct[r.name] {
				return "", fmt.Errorf("struct %s by value is not in the allowlist", r.name)
			}
			return camel(r.name, true), nil
		}
	case kPointer:
		e, err := g.h.resolve(r.elem)
		if err != nil {
			return "", err
		}
		switch {
		case e.kind == kPrim && e.name == "char":
			if ctx == inParam && e.konst {
				return "string", nil
			}
			if ctx == inResult && e.konst {
				return "string", nil
			}
			return "*byte", nil
		case e.kind == kPrim && e.name == "void":
			return "uintptr", nil
		case e.kind == kPrim:
			s, err := g.goType(e, inField)
			return "*" + s, err
		case e.kind == kNamed && e.tag == "struct":
			if g.wantStruct[e.name] {
				return "*" + camel(e.name, true), nil
			}
			g.handles[e.name] = true
			return camel(e.name, true), nil
		case e.kind == kNamed && e.tag == "enum":
			s, err := g.goType(e, inField)
			return "*" + s, err
		case e.kind == kPointer:
			inner, err := g.goType(e, inField)
			if err != nil {
				return "", err
			}
			if inner == "uintptr" {
				return "*uintptr", nil
			}
			return "*" + inner, nil
		case e.kind == kFuncPtr:
			return "*uintptr", nil
		}
	}
	return "", fmt.Errorf("no Go type for %s", t)
}
