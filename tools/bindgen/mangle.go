package main

import (
	"fmt"
	"strconv"
	"strings"
)

// The staging header src/llama-ext.h declares its functions with LLAMA_API
// but outside extern "C", so they are exported under C++ decorated names:
// Itanium on Linux and macOS, MSVC on Windows (the release is built with
// clang-cl). purego looks symbols up by their exported name, so the generator
// decorates the few functions NanoLLM needs itself. It covers what those
// signatures use — free functions in the global namespace taking and
// returning void, bool, int, unsigned int, float and pointers to (const)
// structs or floats — and refuses anything else rather than guess. Every name
// is checked against the export tables of the real libraries by
// `bindgen -check-exports`.

// itanium decorates f per the Itanium C++ ABI (§5.1): _Z <name> <params>,
// return type omitted for a non-template function.
func (h *header) itanium(f *cFunc) (string, error) {
	var b strings.Builder
	b.WriteString("_Z")
	b.WriteString(strconv.Itoa(len(f.name)))
	b.WriteString(f.name)
	if len(f.params) == 0 {
		b.WriteString("v")
		return b.String(), nil
	}
	var subs []string // substitution candidates, in order of appearance
	for _, p := range f.params {
		s, err := h.itaniumType(p.typ, &subs)
		if err != nil {
			return "", fmt.Errorf("%s(%s): %w", f.name, p.name, err)
		}
		b.WriteString(s)
	}
	return b.String(), nil
}

var itaniumBuiltin = map[string]string{
	"void": "v", "bool": "b", "char": "c", "signed char": "a", "unsigned char": "h",
	"int8_t": "a", "uint8_t": "h", "short": "s", "unsigned short": "t", "int16_t": "s", "uint16_t": "t",
	"int": "i", "unsigned": "j", "unsigned int": "j", "int32_t": "i", "uint32_t": "j",
	"float": "f", "double": "d", "long long": "x", "unsigned long long": "y",
}

// itaniumType encodes t, using and extending the substitution table: a class
// name, a qualified type and a pointer type are each substitutable once seen.
// The table holds canonical keys, not encodings, because the second time a
// type appears its encoding already contains a substitution.
func (h *header) itaniumType(t *cType, subs *[]string) (string, error) {
	enc, _, err := h.itaniumEnc(t, subs)
	return enc, err
}

// itaniumEnc returns the encoding of t and its canonical key.
func (h *header) itaniumEnc(t *cType, subs *[]string) (string, string, error) {
	t, err := h.resolve(t)
	if err != nil {
		return "", "", err
	}
	// sub returns the back-reference for key, or registers key and returns enc.
	sub := func(key, enc string) string {
		if i := indexOf(*subs, key); i >= 0 {
			if i == 0 {
				return "S_"
			}
			return "S" + strings.ToUpper(strconv.FormatInt(int64(i-1), 36)) + "_"
		}
		*subs = append(*subs, key)
		return enc
	}
	var enc, key string
	switch t.kind {
	case kPrim:
		s, ok := itaniumBuiltin[t.name]
		if !ok {
			return "", "", fmt.Errorf("type %s has a platform-dependent or unsupported encoding", t.name)
		}
		enc, key = s, s
	case kNamed:
		if t.tag != "enum" && t.tag != "struct" {
			return "", "", fmt.Errorf("type %s is not supported", t)
		}
		key = strconv.Itoa(len(t.name)) + t.name
		enc = sub(key, key)
	case kPointer:
		e, err := h.resolve(t.elem)
		if err != nil {
			return "", "", err
		}
		// const below a pointer is significant: K<type>, itself substitutable.
		// A by-value parameter's top-level const is not part of the signature.
		plain := *e
		plain.konst = false
		inner, innerKey, err := h.itaniumEnc(&plain, subs)
		if err != nil {
			return "", "", err
		}
		if e.konst {
			innerKey = "K" + innerKey
			inner = sub(innerKey, "K"+inner)
		}
		key = "P" + innerKey
		enc = sub(key, "P"+inner)
	default:
		return "", "", fmt.Errorf("type %s is not supported", t)
	}
	return enc, key, nil
}

var msvcBuiltin = map[string]string{
	"void": "X", "bool": "_N", "char": "D", "signed char": "C", "unsigned char": "E",
	"int8_t": "C", "uint8_t": "E", "short": "F", "unsigned short": "G", "int16_t": "F", "uint16_t": "G",
	"int": "H", "unsigned": "I", "unsigned int": "I", "int32_t": "H", "uint32_t": "I",
	"float": "M", "double": "N", "long long": "_J", "unsigned long long": "_K",
	"int64_t": "_J", "uint64_t": "_K",
}

// msvc decorates f the way MSVC (and clang-cl) does for a free __cdecl
// function on a 64-bit target: ?name@@YA <return> <params> @Z, or XZ for no
// parameters. Names and argument types longer than one character are
// remembered and later written as a back-reference digit.
func (h *header) msvc(f *cFunc) (string, error) {
	m := &msvcMangler{h: h}
	var b strings.Builder
	b.WriteString("?")
	b.WriteString(m.name(f.name))
	b.WriteString("@YA")
	ret, err := m.typ(f.ret, true)
	if err != nil {
		return "", fmt.Errorf("%s: return: %w", f.name, err)
	}
	b.WriteString(ret)
	if len(f.params) == 0 {
		b.WriteString("XZ")
		return b.String(), nil
	}
	for _, p := range f.params {
		// Argument back-references are by type, so the lookup key is the
		// encoding without name back-references; the return type is not
		// remembered (llama.dll: ?llama_get_ctx_other@@YAPEAUllama_context@@PEAU1@@Z).
		key, err := (&msvcMangler{h: h}).typ(p.typ, false)
		if err != nil {
			return "", fmt.Errorf("%s(%s): %w", f.name, p.name, err)
		}
		if i := indexOf(m.args, key); i >= 0 && len(key) > 1 {
			b.WriteString(strconv.Itoa(i))
			continue
		}
		s, err := m.typ(p.typ, false)
		if err != nil {
			return "", fmt.Errorf("%s(%s): %w", f.name, p.name, err)
		}
		if len(key) > 1 && len(m.args) < 10 {
			m.args = append(m.args, key)
		}
		b.WriteString(s)
	}
	b.WriteString("@Z")
	return b.String(), nil
}

type msvcMangler struct {
	h     *header
	names []string // name back-references
	args  []string // argument type back-references
}

// name writes an identifier followed by '@', or its back-reference digit.
func (m *msvcMangler) name(s string) string {
	if i := indexOf(m.names, s); i >= 0 {
		return strconv.Itoa(i)
	}
	if len(m.names) < 10 {
		m.names = append(m.names, s)
	}
	return s + "@"
}

func (m *msvcMangler) typ(t *cType, result bool) (string, error) {
	t, err := m.h.resolve(t)
	if err != nil {
		return "", err
	}
	switch t.kind {
	case kPrim:
		s, ok := msvcBuiltin[t.name]
		if !ok {
			return "", fmt.Errorf("type %s is not supported", t.name)
		}
		return s, nil
	case kNamed:
		switch t.tag {
		case "struct":
			return "U" + m.name(t.name) + "@", nil
		case "enum":
			return "W4" + m.name(t.name) + "@", nil
		}
	case kPointer:
		e, err := m.h.resolve(t.elem)
		if err != nil {
			return "", err
		}
		cv := "A"
		if e.konst {
			cv = "B"
		}
		inner, err := m.typ(e, false)
		if err != nil {
			return "", err
		}
		// P: pointer, E: __ptr64; a const pointer itself would be Q.
		p := "P"
		if t.konst {
			p = "Q"
		}
		return p + "E" + cv + inner, nil
	}
	_ = result
	return "", fmt.Errorf("type %s is not supported", t)
}

func indexOf(list []string, s string) int {
	for i, v := range list {
		if v == s {
			return i
		}
	}
	return -1
}
