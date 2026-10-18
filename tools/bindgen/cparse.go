package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode"
)

// The parser understands the subset of C that llama.h, ggml.h, ggml-backend.h
// and the staging llama-ext.h use for the declarations NanoLLM binds: enums,
// plain structs (nested structs and fixed arrays, no bit fields), typedefs,
// function pointer typedefs and function declarations marked with an export
// macro. Everything else — preprocessor logic, inline functions, C++ classes —
// is skipped, and a declaration the allowlist names but the parser could not
// read is an error rather than a guess.

type tokKind int

const (
	tIdent tokKind = iota
	tNumber
	tString
	tPunct
)

type token struct {
	kind tokKind
	text string
	line int
}

// tokenize strips comments and preprocessor lines and splits the rest into
// tokens. Preprocessor conditionals are not evaluated: both branches of an
// #ifdef reach the parser, which is harmless for these headers because the
// conditional parts are export macro definitions.
func tokenize(src string) []token {
	var toks []token
	line := 1
	atLineStart := true
	for i := 0; i < len(src); {
		c := src[i]
		switch {
		case c == '\n':
			line++
			atLineStart = true
			i++
			continue
		case c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v':
			i++
			continue
		case c == '#' && atLineStart:
			// A directive runs to the end of the line, continuations included.
			for i < len(src) && src[i] != '\n' {
				if src[i] == '\\' && i+1 < len(src) && src[i+1] == '\n' {
					line++
					i += 2
					continue
				}
				if src[i] == '/' && i+1 < len(src) && src[i+1] == '*' {
					end := strings.Index(src[i+2:], "*/")
					if end < 0 {
						i = len(src)
						break
					}
					line += strings.Count(src[i:i+2+end], "\n")
					i += 2 + end + 2
					continue
				}
				i++
			}
			continue
		case c == '/' && i+1 < len(src) && src[i+1] == '/':
			for i < len(src) && src[i] != '\n' {
				i++
			}
			continue
		case c == '/' && i+1 < len(src) && src[i+1] == '*':
			end := strings.Index(src[i+2:], "*/")
			if end < 0 {
				i = len(src)
				continue
			}
			line += strings.Count(src[i:i+2+end], "\n")
			i += 2 + end + 2
			continue
		}
		atLineStart = false
		switch {
		case c == '_' || unicode.IsLetter(rune(c)):
			j := i
			for j < len(src) && (src[j] == '_' || unicode.IsLetter(rune(src[j])) || unicode.IsDigit(rune(src[j]))) {
				j++
			}
			toks = append(toks, token{tIdent, src[i:j], line})
			i = j
		case unicode.IsDigit(rune(c)):
			j := i
			for j < len(src) && (unicode.IsLetter(rune(src[j])) || unicode.IsDigit(rune(src[j])) || src[j] == '.') {
				j++
			}
			toks = append(toks, token{tNumber, src[i:j], line})
			i = j
		case c == '"' || c == '\'':
			j := i + 1
			for j < len(src) && src[j] != c {
				if src[j] == '\\' {
					j++
				}
				j++
			}
			toks = append(toks, token{tString, src[i : j+1], line})
			i = j + 1
		default:
			// Two-character operators that matter for enum values.
			if i+1 < len(src) {
				two := src[i : i+2]
				if two == "<<" || two == ">>" || two == "::" {
					toks = append(toks, token{tPunct, two, line})
					i += 2
					continue
				}
			}
			toks = append(toks, token{tPunct, string(c), line})
			i++
		}
	}
	return toks
}

// typeKind classifies a C type.
type typeKind int

const (
	kPrim    typeKind = iota // int32_t, float, bool, char, size_t, void, …
	kNamed                   // struct, enum or typedef name, resolved later
	kPointer                 // elem is the pointee
	kArray                   // elem[n]
	kFuncPtr                 // any function pointer
)

type cType struct {
	kind  typeKind
	name  string // kPrim: the primitive; kNamed: the tag or typedef name
	tag   string // kNamed: "struct", "enum" or "" for a typedef name
	elem  *cType
	n     int
	konst bool
}

func (t *cType) String() string {
	switch t.kind {
	case kPointer:
		s := t.elem.String() + " *"
		if t.konst {
			s += " const"
		}
		return s
	case kArray:
		return fmt.Sprintf("%s[%d]", t.elem, t.n)
	case kFuncPtr:
		return "fnptr"
	}
	s := t.name
	if t.tag != "" {
		s = t.tag + " " + s
	}
	if t.konst {
		s = "const " + s
	}
	return s
}

type cField struct {
	name string
	typ  *cType
}

type cStruct struct {
	name   string
	fields []cField
	union  bool // contains an anonymous union: not bindable by value
	file   string
}

type cEnumConst struct {
	name  string
	value int64
}

type cEnum struct {
	name   string
	consts []cEnumConst
	file   string
}

type cParam struct {
	name string
	typ  *cType
}

type cFunc struct {
	name   string
	ret    *cType
	params []cParam
	file   string
	cxx    bool // declared outside extern "C": the symbol is decorated
}

type cTypedef struct {
	name string
	typ  *cType
}

// header is everything the parser read from a set of files.
type header struct {
	structs  map[string]*cStruct
	enums    map[string]*cEnum
	consts   map[string]int64 // every enum constant, for later enum values
	funcs    map[string]*cFunc
	typedefs map[string]*cTypedef
	// broken records declarations the subset could not read, by name. It is
	// an error only when the allowlist asks for one of them.
	broken map[string]error
}

func newHeader() *header {
	return &header{
		structs:  map[string]*cStruct{},
		enums:    map[string]*cEnum{},
		consts:   map[string]int64{},
		funcs:    map[string]*cFunc{},
		typedefs: map[string]*cTypedef{},
		broken:   map[string]error{},
	}
}

// exportMacros mark a function declaration.
var exportMacros = map[string]bool{"LLAMA_API": true, "GGML_API": true, "GGML_BACKEND_API": true}

// ignoredMacros expand to nothing that matters for the ABI.
var ignoredMacros = map[string]bool{
	"GGML_RESTRICT": true, "GGML_NORETURN": true, "restrict": true, "__restrict": true,
	"inline": true, "static": true, "extern": true,
}

type parser struct {
	toks []token
	pos  int
	h    *header
	file string
	cxx  bool // the file is C++: declarations are not extern "C"
}

func (h *header) parseFile(path string, cxx bool) error {
	src, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	p := &parser{toks: tokenize(string(src)), h: h, file: path, cxx: cxx}
	return p.parseTop()
}

func (p *parser) peek(off int) token {
	if p.pos+off < len(p.toks) {
		return p.toks[p.pos+off]
	}
	return token{kind: tPunct, text: ""}
}

func (p *parser) next() token {
	t := p.peek(0)
	p.pos++
	return t
}

func (p *parser) is(text string) bool { return p.peek(0).text == text }

func (p *parser) expect(text string) error {
	t := p.next()
	if t.text != text {
		return fmt.Errorf("%s:%d: expected %q, got %q", p.file, t.line, text, t.text)
	}
	return nil
}

// skipBalanced skips from an opening bracket to its match.
func (p *parser) skipBalanced(open, close string) {
	depth := 0
	for p.pos < len(p.toks) {
		t := p.next()
		switch t.text {
		case open:
			depth++
		case close:
			depth--
			if depth == 0 {
				return
			}
		}
	}
}

// skipStatement skips to the end of the current top-level statement.
func (p *parser) skipStatement() {
	for p.pos < len(p.toks) {
		switch p.peek(0).text {
		case ";":
			p.pos++
			return
		case "{":
			p.skipBalanced("{", "}")
			if p.is(";") {
				p.pos++
			}
			return
		case "(":
			p.skipBalanced("(", ")")
		default:
			p.pos++
		}
	}
}

func (p *parser) parseTop() error {
	for p.pos < len(p.toks) {
		t := p.peek(0)
		switch {
		case t.text == "extern" && p.peek(1).kind == tString:
			// extern "C" { — the block is transparent.
			p.pos += 2
			if p.is("{") {
				p.pos++
			}
		case t.text == "}" || t.text == ";":
			p.pos++
		case t.text == "enum" && (p.peek(1).kind == tIdent && p.peek(2).text == "{"):
			p.tolerate(p.peek(1).text, p.parseEnum)
		case t.text == "struct" && p.peek(1).kind == tIdent && p.peek(2).text == "{":
			p.tolerate(p.peek(1).text, func() error { return p.parseStruct("") })
		case t.text == "typedef":
			p.tolerate(p.peek(2).text, p.parseTypedef)
		case exportMacros[t.text]:
			p.pos++
			p.tolerate(p.funcName(), p.parseFunc)
		case t.text == "DEPRECATED" || t.text == "GGML_DEPRECATED":
			// Deprecated declarations are never bound.
			p.pos++
			p.skipBalanced("(", ")")
			if p.is(";") {
				p.pos++
			}
		default:
			p.skipStatement()
		}
	}
	return nil
}

// tolerate runs one top-level parse; on failure it records the error under
// name and resumes after the statement.
func (p *parser) tolerate(name string, parse func() error) {
	start := p.pos
	if err := parse(); err != nil {
		p.h.broken[name] = err
		p.pos = start
		p.skipStatement()
	}
}

// funcName guesses the name of the declaration at the cursor: the identifier
// before the first parenthesis.
func (p *parser) funcName() string {
	for i := p.pos; i < len(p.toks) && p.toks[i].text != ";"; i++ {
		if p.toks[i].text == "(" && i > 0 {
			return p.toks[i-1].text
		}
	}
	return ""
}

func (p *parser) parseEnum() error {
	p.pos++ // enum
	name := p.next().text
	if err := p.expect("{"); err != nil {
		return err
	}
	e := &cEnum{name: name, file: p.file}
	var next int64
	for !p.is("}") {
		cname := p.next()
		if cname.kind != tIdent {
			return fmt.Errorf("%s:%d: enum %s: expected a constant, got %q", p.file, cname.line, name, cname.text)
		}
		v := next
		if p.is("=") {
			p.pos++
			var expr []token
			for !p.is(",") && !p.is("}") {
				expr = append(expr, p.next())
			}
			val, err := p.evalConst(expr)
			if err != nil {
				return fmt.Errorf("%s:%d: enum %s.%s: %w", p.file, cname.line, name, cname.text, err)
			}
			v = val
		}
		e.consts = append(e.consts, cEnumConst{cname.text, v})
		p.h.consts[cname.text] = v
		next = v + 1
		if p.is(",") {
			p.pos++
		}
	}
	p.pos++ // }
	if p.is(";") {
		p.pos++
	}
	p.h.enums[name] = e
	return nil
}

// evalConst evaluates the integer expressions enum values use: literals,
// earlier constants, unary minus, parentheses, |, <<, + and -.
func (p *parser) evalConst(expr []token) (int64, error) {
	pos := 0
	var parseOr func() (int64, error)
	parseUnary := func() (int64, error) {
		if pos >= len(expr) {
			return 0, fmt.Errorf("truncated expression")
		}
		neg := false
		for pos < len(expr) && (expr[pos].text == "-" || expr[pos].text == "+") {
			if expr[pos].text == "-" {
				neg = !neg
			}
			pos++
		}
		if pos >= len(expr) {
			return 0, fmt.Errorf("truncated expression")
		}
		t := expr[pos]
		pos++
		var v int64
		switch {
		case t.text == "(":
			inner, err := parseOr()
			if err != nil {
				return 0, err
			}
			if pos >= len(expr) || expr[pos].text != ")" {
				return 0, fmt.Errorf("unbalanced parenthesis")
			}
			pos++
			v = inner
		case t.kind == tNumber:
			s := strings.TrimRight(t.text, "uUlL")
			n, err := strconv.ParseInt(s, 0, 64)
			if err != nil {
				u, uerr := strconv.ParseUint(s, 0, 64)
				if uerr != nil {
					return 0, err
				}
				n = int64(u)
			}
			v = n
		case t.kind == tIdent:
			c, ok := p.h.consts[t.text]
			if !ok {
				return 0, fmt.Errorf("unknown constant %s", t.text)
			}
			v = c
		default:
			return 0, fmt.Errorf("unexpected %q", t.text)
		}
		if neg {
			v = -v
		}
		return v, nil
	}
	parseShift := func() (int64, error) {
		v, err := parseUnary()
		if err != nil {
			return 0, err
		}
		for pos < len(expr) {
			switch expr[pos].text {
			case "<<":
				pos++
				r, err := parseUnary()
				if err != nil {
					return 0, err
				}
				v <<= uint(r)
			case "+":
				pos++
				r, err := parseUnary()
				if err != nil {
					return 0, err
				}
				v += r
			case "-":
				pos++
				r, err := parseUnary()
				if err != nil {
					return 0, err
				}
				v -= r
			default:
				return v, nil
			}
		}
		return v, nil
	}
	parseOr = func() (int64, error) {
		v, err := parseShift()
		if err != nil {
			return 0, err
		}
		for pos < len(expr) && expr[pos].text == "|" {
			pos++
			r, err := parseShift()
			if err != nil {
				return 0, err
			}
			v |= r
		}
		return v, nil
	}
	v, err := parseOr()
	if err != nil {
		return 0, err
	}
	if pos != len(expr) {
		return 0, fmt.Errorf("trailing %q", expr[pos].text)
	}
	return v, nil
}

// parseStruct reads "struct NAME { … }" (the caller consumed nothing) and
// registers it under NAME, and under alias when it is a typedef.
func (p *parser) parseStruct(alias string) error {
	p.pos++ // struct
	name := p.next().text
	if err := p.expect("{"); err != nil {
		return err
	}
	s := &cStruct{name: name, file: p.file}
members:
	for !p.is("}") {
		if p.pos >= len(p.toks) {
			return fmt.Errorf("%s: struct %s is not closed", p.file, name)
		}
		if p.is("union") || (p.is("struct") && p.peek(1).text == "{") {
			// An anonymous union or struct member: recorded, never bound.
			s.union = true
			p.pos++
			p.skipBalanced("{", "}")
			for !p.is(";") {
				p.pos++
			}
			p.pos++
			continue
		}
		base, err := p.parseBaseType()
		if err != nil {
			return fmt.Errorf("struct %s: %w", name, err)
		}
		for {
			typ, fname, err := p.parseDeclarator(base)
			if err != nil {
				return fmt.Errorf("struct %s: %w", name, err)
			}
			if p.is("(") {
				// A C++ member function: not a C struct, skip its body.
				s.union = true
				p.skipBalanced("(", ")")
				for p.is("const") || p.is("noexcept") {
					p.pos++
				}
				if p.is("{") {
					p.skipBalanced("{", "}")
				} else if p.is(";") {
					p.pos++
				}
				continue members
			}
			s.fields = append(s.fields, cField{fname, typ})
			if p.is(",") {
				p.pos++
				continue
			}
			break
		}
		if p.is("=") {
			// A C++ default member initializer (llama-ext.h): not a C struct.
			s.union = true
			for !p.is(";") {
				p.pos++
			}
		}
		if err := p.expect(";"); err != nil {
			return fmt.Errorf("struct %s: %w", name, err)
		}
	}
	p.pos++ // }
	p.h.structs[name] = s
	if alias != "" && alias != name {
		p.h.typedefs[alias] = &cTypedef{alias, &cType{kind: kNamed, tag: "struct", name: name}}
	}
	return nil
}

// parseBaseType reads qualifiers and a type name, e.g. "const struct llama_model".
func (p *parser) parseBaseType() (*cType, error) {
	t := &cType{kind: kPrim}
	sawName := false
	for {
		tok := p.peek(0)
		switch {
		case tok.text == "const" || tok.text == "volatile":
			t.konst = t.konst || tok.text == "const"
			p.pos++
			continue
		case ignoredMacros[tok.text]:
			p.pos++
			continue
		case tok.text == "struct" || tok.text == "enum" || tok.text == "union":
			if sawName {
				return t, nil
			}
			p.pos++
			t.kind = kNamed
			t.tag = tok.text
			t.name = p.next().text
			sawName = true
			continue
		case tok.text == "unsigned" || tok.text == "signed" || tok.text == "long" || tok.text == "short":
			// Multi-word primitives: "unsigned int", "long long".
			p.pos++
			if t.name == "" {
				t.name = tok.text
			} else {
				t.name += " " + tok.text
			}
			sawName = true
			continue
		case tok.kind == tIdent && !sawName:
			p.pos++
			t.name = tok.text
			sawName = true
			if !isPrim(tok.text) {
				t.kind = kNamed
			}
			continue
		case tok.kind == tIdent && (t.name == "unsigned" || t.name == "signed") && (tok.text == "int" || tok.text == "char"):
			p.pos++
			t.name += " " + tok.text
			continue
		}
		break
	}
	if !sawName {
		tok := p.peek(0)
		return nil, fmt.Errorf("%s:%d: expected a type, got %q", p.file, tok.line, tok.text)
	}
	return t, nil
}

// parseDeclarator reads "* const * name[4]" or "(*name)(args)" after a base
// type and returns the full type and the declared name (may be empty for an
// abstract parameter).
func (p *parser) parseDeclarator(base *cType) (*cType, string, error) {
	t := base
	for p.is("*") || p.is("&") {
		if p.is("&") {
			return nil, "", fmt.Errorf("%s:%d: references are not supported", p.file, p.peek(0).line)
		}
		p.pos++
		t = &cType{kind: kPointer, elem: t}
		for p.is("const") || ignoredMacros[p.peek(0).text] {
			if p.is("const") {
				t.konst = true
			}
			p.pos++
		}
	}
	if p.is("(") && p.peek(1).text == "*" {
		// Function pointer: (*name)(params)
		p.pos += 2
		name := ""
		if p.peek(0).kind == tIdent {
			name = p.next().text
		}
		if err := p.expect(")"); err != nil {
			return nil, "", err
		}
		p.skipBalanced("(", ")")
		return &cType{kind: kFuncPtr}, name, nil
	}
	name := ""
	if p.peek(0).kind == tIdent {
		name = p.next().text
	}
	for p.is("[") {
		p.pos++
		var expr []token
		for !p.is("]") {
			expr = append(expr, p.next())
		}
		p.pos++
		n, err := p.evalConst(expr)
		if err != nil {
			return nil, "", fmt.Errorf("%s: array %s: %w", p.file, name, err)
		}
		t = &cType{kind: kArray, elem: t, n: int(n)}
	}
	return t, name, nil
}

func (p *parser) parseTypedef() error {
	p.pos++ // typedef
	if p.is("struct") && p.peek(1).kind == tIdent && p.peek(2).text == "{" {
		// typedef struct NAME { … } ALIAS;
		save := p.pos
		p.pos++
		p.pos++
		p.skipBalanced("{", "}")
		alias := p.next().text
		end := p.pos
		p.pos = save
		if err := p.parseStruct(alias); err != nil {
			return err
		}
		p.pos = end
		if p.is(";") {
			p.pos++
		}
		return nil
	}
	if p.is("enum") && p.peek(1).kind == tIdent && p.peek(2).text == "{" {
		if err := p.parseEnum(); err != nil {
			return err
		}
		return nil
	}
	if p.is("struct") && p.peek(1).text == "{" || p.is("enum") && p.peek(1).text == "{" || p.is("union") {
		p.skipStatement()
		return nil
	}
	base, err := p.parseBaseType()
	if err != nil {
		p.skipStatement()
		return nil //nolint:nilerr // a typedef the subset does not cover is skipped
	}
	typ, name, err := p.parseDeclarator(base)
	if err != nil || name == "" {
		p.skipStatement()
		return nil //nolint:nilerr // as above
	}
	p.h.typedefs[name] = &cTypedef{name, typ}
	p.skipStatement()
	return nil
}

// parseFunc reads a declaration after its export macro. A declaration that is
// not a function (a variable, a C++ "using") is skipped.
func (p *parser) parseFunc() error {
	start := p.pos
	base, err := p.parseBaseType()
	if err != nil {
		p.pos = start
		p.skipStatement()
		return nil //nolint:nilerr // not a declaration the subset covers
	}
	typ, name, err := p.parseDeclarator(base)
	if err != nil || name == "" || !p.is("(") {
		p.pos = start
		p.skipStatement()
		return nil //nolint:nilerr // as above
	}
	f := &cFunc{name: name, ret: typ, file: p.file, cxx: p.cxx}
	p.pos++ // (
	if p.is("void") && p.peek(1).text == ")" {
		p.pos++
	}
	for !p.is(")") {
		pbase, err := p.parseBaseType()
		if err != nil {
			return fmt.Errorf("function %s: %w", name, err)
		}
		ptyp, pname, err := p.parseDeclarator(pbase)
		if err != nil {
			return fmt.Errorf("function %s: %w", name, err)
		}
		f.params = append(f.params, cParam{pname, ptyp})
		if p.is(",") {
			p.pos++
		}
	}
	p.pos++ // )
	p.skipStatement()
	if _, dup := p.h.funcs[name]; !dup {
		p.h.funcs[name] = f
	}
	return nil
}

var prims = map[string]bool{
	"void": true, "bool": true, "_Bool": true, "char": true, "float": true, "double": true,
	"int": true, "unsigned": true, "signed": true, "short": true, "long": true,
	"int8_t": true, "uint8_t": true, "int16_t": true, "uint16_t": true,
	"int32_t": true, "uint32_t": true, "int64_t": true, "uint64_t": true,
	"size_t": true, "intptr_t": true, "uintptr_t": true,
}

func isPrim(name string) bool { return prims[name] }
