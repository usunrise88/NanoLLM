package chat

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// value is a JSON value with the semantics the Gemma 4 template sees through
// Jinja: object keys keep their order (dictsort then sorts them), numbers keep
// their literal so ints and floats print as Python prints them, and a missing
// key is "undefined" rather than an error.
type value struct {
	kind kind
	str  string // string, or the number literal
	b    bool
	arr  []value
	keys []string
	obj  map[string]value
}

type kind int

const (
	kUndefined kind = iota
	kNull
	kBool
	kNumber
	kString
	kArray
	kObject
)

func parseValue(raw []byte) (value, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return value{}, err
	}
	if dec.More() {
		return value{}, fmt.Errorf("trailing data after JSON value")
	}
	if _, err := dec.Token(); err == nil {
		return value{}, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

func decodeValue(dec *json.Decoder) (value, error) {
	tok, err := dec.Token()
	if err != nil {
		return value{}, err
	}
	switch t := tok.(type) {
	case nil:
		return value{kind: kNull}, nil
	case bool:
		return value{kind: kBool, b: t}, nil
	case json.Number:
		return value{kind: kNumber, str: t.String()}, nil
	case string:
		return value{kind: kString, str: t}, nil
	case json.Delim:
		switch t {
		case '[':
			v := value{kind: kArray}
			for dec.More() {
				e, err := decodeValue(dec)
				if err != nil {
					return value{}, err
				}
				v.arr = append(v.arr, e)
			}
			_, err := dec.Token()
			return v, err
		case '{':
			v := value{kind: kObject, obj: map[string]value{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return value{}, err
				}
				k := kt.(string)
				e, err := decodeValue(dec)
				if err != nil {
					return value{}, err
				}
				if _, dup := v.obj[k]; !dup {
					v.keys = append(v.keys, k)
				}
				v.obj[k] = e
			}
			_, err := dec.Token()
			return v, err
		}
	}
	return value{}, fmt.Errorf("unexpected JSON token %v", tok)
}

func str(s string) value { return value{kind: kString, str: s} }

func (v value) isMapping() bool { return v.kind == kObject }
func (v value) isString() bool  { return v.kind == kString }

// get is v[key] in Jinja: undefined unless v is an object holding key.
func (v value) get(key string) value {
	if v.kind != kObject {
		return value{}
	}
	return v.obj[key]
}

// defined is Jinja's "is defined": a key that is present, even as null.
func (v value) defined() bool { return v.kind != kUndefined }

// truthy is Python truthiness.
func (v value) truthy() bool {
	switch v.kind {
	case kBool:
		return v.b
	case kNumber:
		f, err := strconv.ParseFloat(v.str, 64)
		return err != nil || f != 0
	case kString:
		return v.str != ""
	case kArray:
		return len(v.arr) > 0
	case kObject:
		return len(v.keys) > 0
	}
	return false
}

// sortedKeys is dictsort: case-insensitive, stable on ties.
func (v value) sortedKeys() []string {
	keys := append([]string(nil), v.keys...)
	sort.SliceStable(keys, func(i, j int) bool { return strings.ToLower(keys[i]) < strings.ToLower(keys[j]) })
	return keys
}

// pyStr is str(v) in Python, which Jinja's output and filters use. An
// undefined value prints as nothing.
func (v value) pyStr() string {
	switch v.kind {
	case kUndefined:
		return ""
	case kString:
		return v.str
	default:
		return v.pyRepr()
	}
}

// pyRepr is repr(v), which is how containers print their elements.
func (v value) pyRepr() string {
	switch v.kind {
	case kUndefined:
		return ""
	case kNull:
		return "None"
	case kBool:
		if v.b {
			return "True"
		}
		return "False"
	case kNumber:
		return pyNumber(v.str)
	case kString:
		return pyQuote(v.str)
	case kArray:
		parts := make([]string, len(v.arr))
		for i, e := range v.arr {
			parts[i] = e.pyRepr()
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case kObject:
		parts := make([]string, len(v.keys))
		for i, k := range v.keys {
			parts[i] = pyQuote(k) + ": " + v.obj[k].pyRepr()
		}
		return "{" + strings.Join(parts, ", ") + "}"
	}
	return ""
}

// upper is Jinja's upper filter: str(v).upper().
func (v value) upper() string { return strings.ToUpper(v.pyStr()) }

func pyQuote(s string) string {
	q := "'"
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		q = `"`
	}
	var sb strings.Builder
	sb.WriteString(q)
	for _, r := range s {
		switch {
		case r == '\\':
			sb.WriteString(`\\`)
		case string(r) == q:
			sb.WriteString(`\` + q)
		case r == '\n':
			sb.WriteString(`\n`)
		case r == '\r':
			sb.WriteString(`\r`)
		case r == '\t':
			sb.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&sb, `\x%02x`, r)
		default:
			sb.WriteRune(r)
		}
	}
	sb.WriteString(q)
	return sb.String()
}

// pyNumber prints a JSON number literal the way Python prints what json.loads
// made of it: integers exactly, anything with a fraction or exponent as a
// float repr.
func pyNumber(lit string) string {
	if !strings.ContainsAny(lit, ".eE") {
		if strings.TrimLeft(lit, "-0") == "" {
			return "0"
		}
		return lit
	}
	f, err := strconv.ParseFloat(lit, 64)
	if err != nil {
		return lit
	}
	return pyFloat(f)
}

// pyFloat is repr(float): the shortest round-trip digits, positional for
// exponents in [-4, 16), scientific with a two-digit exponent otherwise, and a
// trailing ".0" on integral values.
func pyFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case math.IsNaN(f):
		return "nan"
	}
	s := strconv.FormatFloat(f, 'e', -1, 64) // d.ddde±XX
	mant, expStr, _ := strings.Cut(s, "e")
	exp, _ := strconv.Atoi(expStr)
	sign := ""
	if strings.HasPrefix(mant, "-") {
		sign, mant = "-", mant[1:]
	}
	digits := strings.Replace(mant, ".", "", 1)
	if exp < -4 || exp >= 16 {
		m := digits[:1]
		if len(digits) > 1 {
			m += "." + digits[1:]
		}
		es := "+"
		if exp < 0 {
			es, exp = "-", -exp
		}
		return fmt.Sprintf("%s%se%s%02d", sign, m, es, exp)
	}
	var out string
	if exp >= 0 {
		if len(digits) <= exp+1 {
			out = digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
		} else {
			out = digits[:exp+1] + "." + digits[exp+1:]
		}
	} else {
		out = "0." + strings.Repeat("0", -exp-1) + digits
	}
	return sign + out
}

// pyStrip is Python's str.strip(), which Jinja's trim filter is: it also
// strips the ASCII separators U+001C–U+001F that unicode.IsSpace does not.
func pyStrip(s string) string {
	return strings.TrimFunc(s, func(r rune) bool {
		return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
	})
}
