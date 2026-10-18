package grammar

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// jv is a JSON value as llama.cpp's common_json holds it: object keys in
// insertion order, numbers either integers or doubles, and a dump that prints
// what nlohmann::json prints, because const and enum values and property
// names reach the grammar through that dump.
type jv struct {
	kind  jkind
	s     string // string value, or the number literal
	b     bool
	isInt bool
	i     int64
	f     float64
	arr   []jv
	keys  []string
	obj   map[string]jv
}

type jkind int

const (
	jNull jkind = iota
	jBool
	jNumber
	jString
	jArray
	jObject
)

func parseJSON(raw []byte) (jv, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := decodeJV(dec)
	if err != nil {
		return jv{}, err
	}
	if _, err := dec.Token(); err == nil {
		return jv{}, fmt.Errorf("trailing data after the schema")
	}
	return v, nil
}

func decodeJV(dec *json.Decoder) (jv, error) {
	tok, err := dec.Token()
	if err != nil {
		return jv{}, err
	}
	switch t := tok.(type) {
	case nil:
		return jv{kind: jNull}, nil
	case bool:
		return jv{kind: jBool, b: t}, nil
	case string:
		return jv{kind: jString, s: t}, nil
	case json.Number:
		v := jv{kind: jNumber, s: t.String()}
		if !strings.ContainsAny(v.s, ".eE") {
			if i, err := strconv.ParseInt(v.s, 10, 64); err == nil {
				v.isInt, v.i, v.f = true, i, float64(i)
				return v, nil
			}
		}
		v.f, err = strconv.ParseFloat(v.s, 64)
		return v, err
	case json.Delim:
		switch t {
		case '[':
			v := jv{kind: jArray}
			for dec.More() {
				e, err := decodeJV(dec)
				if err != nil {
					return jv{}, err
				}
				v.arr = append(v.arr, e)
			}
			_, err := dec.Token()
			return v, err
		case '{':
			v := jv{kind: jObject, obj: map[string]jv{}}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return jv{}, err
				}
				k := kt.(string)
				e, err := decodeJV(dec)
				if err != nil {
					return jv{}, err
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
	return jv{}, fmt.Errorf("unexpected JSON token %v", tok)
}

func (v jv) has(k string) bool {
	if v.kind != jObject {
		return false
	}
	_, ok := v.obj[k]
	return ok
}

func (v jv) at(k string) jv { return v.obj[k] }

// set returns a copy of an object with k set; the order of an existing key is
// kept, a new one goes last.
func (v jv) set(k string, val jv) jv {
	out := jv{kind: jObject, obj: map[string]jv{}, keys: append([]string(nil), v.keys...)}
	for kk, vv := range v.obj {
		out.obj[kk] = vv
	}
	if _, ok := out.obj[k]; !ok {
		out.keys = append(out.keys, k)
	}
	out.obj[k] = val
	return out
}

func (v jv) isTrue() bool { return v.kind == jBool && v.b }

// dump is nlohmann::json::dump(): compact, keys in order, non-ASCII as is.
func (v jv) dump() string {
	var sb strings.Builder
	v.dumpTo(&sb)
	return sb.String()
}

func (v jv) dumpTo(sb *strings.Builder) {
	switch v.kind {
	case jNull:
		sb.WriteString("null")
	case jBool:
		if v.b {
			sb.WriteString("true")
		} else {
			sb.WriteString("false")
		}
	case jNumber:
		if v.isInt {
			sb.WriteString(strconv.FormatInt(v.i, 10))
		} else {
			sb.WriteString(nlohmannFloat(v.f))
		}
	case jString:
		dumpString(sb, v.s)
	case jArray:
		sb.WriteByte('[')
		for i, e := range v.arr {
			if i > 0 {
				sb.WriteByte(',')
			}
			e.dumpTo(sb)
		}
		sb.WriteByte(']')
	case jObject:
		sb.WriteByte('{')
		for i, k := range v.keys {
			if i > 0 {
				sb.WriteByte(',')
			}
			dumpString(sb, k)
			sb.WriteByte(':')
			v.obj[k].dumpTo(sb)
		}
		sb.WriteByte('}')
	}
}

func dumpString(sb *strings.Builder, s string) {
	sb.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			sb.WriteString(`\"`)
		case '\\':
			sb.WriteString(`\\`)
		case '\b':
			sb.WriteString(`\b`)
		case '\f':
			sb.WriteString(`\f`)
		case '\n':
			sb.WriteString(`\n`)
		case '\r':
			sb.WriteString(`\r`)
		case '\t':
			sb.WriteString(`\t`)
		default:
			if c < 0x20 {
				fmt.Fprintf(sb, `\u%04x`, c)
			} else {
				sb.WriteByte(c)
			}
		}
	}
	sb.WriteByte('"')
}

// nlohmannFloat prints a double as nlohmann's dtoa does: the shortest digits
// that round-trip, positional for decimal exponents in (-4, 15], scientific
// with a signed two-digit exponent otherwise, and ".0" on integral values.
func nlohmannFloat(f float64) string {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return "null"
	}
	if f == 0 {
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	sign := ""
	if f < 0 {
		sign, f = "-", -f
	}
	e := strconv.FormatFloat(f, 'e', -1, 64) // d.ddde±XX
	mant, expStr, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(expStr)
	digits := strings.Replace(mant, ".", "", 1)
	k, n := len(digits), exp+1
	switch {
	case k <= n && n <= 15:
		return sign + digits + strings.Repeat("0", n-k) + ".0"
	case 0 < n && n <= 15:
		return sign + digits[:n] + "." + digits[n:]
	case -4 < n && n <= 0:
		return sign + "0." + strings.Repeat("0", -n) + digits
	}
	m := digits[:1]
	if k > 1 {
		m += "." + digits[1:]
	}
	x := n - 1
	es := "+"
	if x < 0 {
		es, x = "-", -x
	}
	return fmt.Sprintf("%s%se%s%02d", sign, m, es, x)
}
