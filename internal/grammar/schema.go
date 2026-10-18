package grammar

import (
	"math"
	"strconv"
	"strings"
)

// The schema tree, ported from llama.cpp's common/json-schema.cpp: JSON
// Schema is first reduced to these few kinds, and the converter only ever
// sees them. Keeping the two steps apart, as upstream does, keeps the port
// line for line comparable.

type skind int

const (
	kRef skind = iota
	kAnyOf
	kAllOf
	kConst
	kEnum
	kObject
	kTuple
	kArray
	kString
	kInteger
	kNumber
	kBoolean
	kNull
	kAny
)

type strFormat int

const (
	fmtNone strFormat = iota
	fmtDate
	fmtTime
	fmtDateTime
	fmtUUID
)

type snode struct {
	kind skind

	ref    string // kRef
	target *snode

	children []*snode // kAnyOf, kAllOf

	value  jv   // kConst
	values []jv // kEnum

	props      []sprop // kObject
	additional *snode  // nil: no additional properties

	items              []*snode // kTuple
	item               *snode   // kArray
	minItems, maxItems int      // maxItems -1 for unbounded

	pattern              string // kString
	format               strFormat
	minLength, maxLength int

	minimum, maximum int64 // kInteger
}

type sprop struct {
	name     string
	schema   *snode
	required bool
}

type schemaError struct{ msg string }

func (e schemaError) Error() string { return e.msg }

type schemaBuilder struct {
	root    jv
	refs    map[string]*snode
	pending []*snode
}

func (b *schemaBuilder) fail(path, msg string) {
	panic(schemaError{"JSON schema error at " + path + ": " + msg})
}

// parseSchema is common_chat_schema_from_json.
func parseSchema(root jv) (node *snode, err error) {
	defer func() {
		if r := recover(); r != nil {
			se, ok := r.(schemaError)
			if !ok {
				panic(r)
			}
			err = se
		}
	}()
	b := &schemaBuilder{root: root, refs: map[string]*snode{}}
	node = b.build(root, "#")
	for _, r := range b.pending {
		r.target = b.refs[r.ref]
	}
	return node, nil
}

func (b *schemaBuilder) count(s jv, key, path string, def int) int {
	if !s.has(key) {
		return def
	}
	v := s.at(key)
	if v.kind != jNumber || !v.isInt || v.i < 0 || v.i > math.MaxInt32 {
		b.fail(path, key+" must be a non-negative integer")
	}
	return int(v.i)
}

// bound reads a numeric bound; a fractional one is rounded inwards, towards
// the integers it still admits.
func (b *schemaBuilder) bound(s jv, key, path string, roundUp bool) int64 {
	v := s.at(key)
	if v.kind != jNumber {
		b.fail(path, key+" must be a number")
	}
	if v.isInt {
		return v.i
	}
	if roundUp {
		return int64(math.Ceil(v.f))
	}
	return int64(math.Floor(v.f))
}

func (b *schemaBuilder) format(s jv, path string) strFormat {
	if !s.has("format") {
		return fmtNone
	}
	v := s.at("format")
	if v.kind != jString {
		b.fail(path, "format must be a string")
	}
	switch f := v.s; {
	case f == "date":
		return fmtDate
	case f == "time":
		return fmtTime
	case f == "date-time":
		return fmtDateTime
	case f == "uuid", len(f) == 5 && strings.HasPrefix(f, "uuid") && f[4] >= '1' && f[4] <= '5':
		return fmtUUID
	}
	return fmtNone
}

func (b *schemaBuilder) resolve(ref, path string) jv {
	target := b.root
	tokens := strings.Split(ref[1:], "/")
	for _, sel := range tokens[1:] {
		switch {
		case target.kind == jObject && target.has(sel):
			target = target.at(sel)
		case target.kind == jArray:
			idx, err := strconv.ParseUint(sel, 10, 64)
			if err != nil || idx >= uint64(len(target.arr)) {
				b.fail(path, "cannot resolve $ref "+ref+", "+sel+" is out of range")
			}
			target = target.arr[idx]
		default:
			b.fail(path, "cannot resolve $ref "+ref+", "+sel+" not found")
		}
	}
	return target
}

func (b *schemaBuilder) buildRef(v jv, path string) *snode {
	if v.kind != jString {
		b.fail(path, "$ref must be a string")
	}
	ref := v.s
	if !strings.HasPrefix(ref, "#/") {
		b.fail(path, "unsupported $ref "+ref+", only references into the same document are supported")
	}
	if _, ok := b.refs[ref]; !ok {
		// Reserve the key first, so a cycle back to this $ref stops here.
		b.refs[ref] = nil
		b.refs[ref] = b.build(b.resolve(ref, path), ref)
	}
	n := &snode{kind: kRef, ref: ref}
	b.pending = append(b.pending, n)
	return n
}

func (b *schemaBuilder) alternatives(kind skind, alts jv, path string) *snode {
	if alts.kind != jArray {
		b.fail(path, "must be an array of schemas")
	}
	if len(alts.arr) == 0 {
		b.fail(path, "must not be empty")
	}
	n := &snode{kind: kind}
	for i, alt := range alts.arr {
		n.children = append(n.children, b.build(alt, path+"/"+strconv.Itoa(i)))
	}
	return n
}

func (b *schemaBuilder) object(s jv, path string) *snode {
	n := &snode{kind: kObject}
	required := map[string]bool{}
	if r := s.at("required"); s.has("required") && r.kind == jArray {
		for _, name := range r.arr {
			if name.kind == jString {
				required[name.s] = true
			}
		}
	}
	if s.has("properties") {
		props := s.at("properties")
		if props.kind != jObject {
			b.fail(path, "properties must be an object")
		}
		for _, name := range props.keys {
			n.props = append(n.props, sprop{name: name, schema: b.build(props.obj[name], path+"/properties/"+name), required: required[name]})
		}
	}
	if s.has("additionalProperties") {
		add := s.at("additionalProperties")
		switch add.kind {
		case jBool:
			if add.b {
				n.additional = &snode{kind: kAny}
			}
		case jObject:
			n.additional = b.build(add, path+"/additionalProperties")
		default:
			b.fail(path, "additionalProperties must be a boolean or a schema")
		}
	} else if !s.has("properties") {
		// {"type": "object"} on its own accepts any object.
		n.additional = &snode{kind: kAny}
	}
	return n
}

func (b *schemaBuilder) array(s jv, path string) *snode {
	n := &snode{kind: kArray}
	if s.has("items") || s.has("prefixItems") {
		// "items" wins when both are present; a schema rather than an array
		// is the item schema.
		key := "prefixItems"
		if s.has("items") {
			key = "items"
		}
		items := s.at(key)
		if items.kind == jArray {
			t := &snode{kind: kTuple}
			for i, it := range items.arr {
				t.items = append(t.items, b.build(it, path+"/"+key+"/"+strconv.Itoa(i)))
			}
			return t
		}
		n.item = b.build(items, path+"/"+key)
	} else {
		n.item = &snode{kind: kAny}
	}
	n.minItems = b.count(s, "minItems", path, 0)
	n.maxItems = b.count(s, "maxItems", path, -1)
	return n
}

func (b *schemaBuilder) str(s jv, path string) *snode {
	n := &snode{kind: kString}
	if s.has("pattern") {
		p := s.at("pattern")
		if p.kind != jString {
			b.fail(path, "pattern must be a string")
		}
		n.pattern = p.s
	}
	n.format = b.format(s, path)
	n.minLength = b.count(s, "minLength", path, 0)
	n.maxLength = b.count(s, "maxLength", path, -1)
	return n
}

func (b *schemaBuilder) integer(s jv, path string) *snode {
	n := &snode{kind: kInteger, minimum: math.MinInt64, maximum: math.MaxInt64}
	if s.has("minimum") {
		n.minimum = b.bound(s, "minimum", path, true)
	} else if s.has("exclusiveMinimum") {
		n.minimum = b.bound(s, "exclusiveMinimum", path, false) + 1
	}
	if s.has("maximum") {
		n.maximum = b.bound(s, "maximum", path, false)
	} else if s.has("exclusiveMaximum") {
		n.maximum = b.bound(s, "exclusiveMaximum", path, true) - 1
	}
	return n
}

func (b *schemaBuilder) build(s jv, path string) *snode {
	if s.kind != jObject {
		b.fail(path, "schema must be an object")
	}
	if s.has("$ref") {
		return b.buildRef(s.at("$ref"), path)
	}
	if s.has("oneOf") || s.has("anyOf") {
		key := "anyOf"
		if s.has("oneOf") {
			key = "oneOf"
		}
		return b.alternatives(kAnyOf, s.at(key), path+"/"+key)
	}
	var typ jv
	if s.has("type") {
		typ = s.at("type")
	}
	if typ.kind == jArray {
		// {"type": ["a", "b"], …} is {"anyOf": [{"type": "a", …}, {"type": "b", …}]}.
		if len(typ.arr) == 0 {
			b.fail(path, "type must not be empty")
		}
		n := &snode{kind: kAnyOf}
		for i, t := range typ.arr {
			n.children = append(n.children, b.build(s.set("type", t), path+"/type/"+strconv.Itoa(i)))
		}
		return n
	}
	if s.has("const") {
		return &snode{kind: kConst, value: s.at("const")}
	}
	if s.has("enum") {
		vals := s.at("enum")
		if vals.kind != jArray || len(vals.arr) == 0 {
			b.fail(path, "enum must be a non-empty array")
		}
		return &snode{kind: kEnum, values: vals.arr}
	}
	if s.has("type") && typ.kind != jString && typ.kind != jNull {
		b.fail(path, "type must be a string or an array of strings")
	}
	typeName := ""
	if typ.kind == jString {
		typeName = typ.s
	}
	hasProps := s.has("properties") || (s.has("additionalProperties") && !s.at("additionalProperties").isTrue())

	if typeName == "" {
		// Without a type the structural keywords decide, in the order the
		// converter checks them.
		switch {
		case hasProps:
			return b.object(s, path)
		case s.has("allOf"):
			return b.alternatives(kAllOf, s.at("allOf"), path+"/allOf")
		case s.has("items") || s.has("prefixItems"):
			return b.array(s, path)
		case s.has("pattern") || s.has("minLength") || s.has("maxLength") || b.format(s, path) != fmtNone:
			return b.str(s, path)
		}
		return &snode{kind: kAny}
	}
	switch typeName {
	case "object":
		if !hasProps && s.has("allOf") {
			return b.alternatives(kAllOf, s.at("allOf"), path+"/allOf")
		}
		return b.object(s, path)
	case "string":
		if s.has("allOf") {
			return b.alternatives(kAllOf, s.at("allOf"), path+"/allOf")
		}
		return b.str(s, path)
	case "array":
		return b.array(s, path)
	case "integer":
		return b.integer(s, path)
	case "number":
		return &snode{kind: kNumber}
	case "boolean":
		return &snode{kind: kBoolean}
	case "null":
		return &snode{kind: kNull}
	}
	b.fail(path, "unrecognized type "+typeName)
	return nil
}
