package benchfmt

import (
	_ "embed"
	"encoding/json"
	"reflect"
	"sort"
	"strings"
)

// SchemaJSON is the JSON Schema of v1, generated from the Go types (see
// GenerateSchema) and committed, so the contract is reviewable as a file and
// a change to it shows in a diff.
//
//go:embed schema.json
var SchemaJSON []byte

// GenerateSchema derives the JSON Schema from Result. Required properties are
// the fields without omitempty.
func GenerateSchema() ([]byte, error) {
	s := typeSchema(reflect.TypeFor[Result]())
	s["$schema"] = "https://json-schema.org/draft/2020-12/schema"
	s["$id"] = "https://nano.llmto.day/schema/" + strings.ReplaceAll(Schema, "/", "-") + ".json"
	s["title"] = Schema
	return json.MarshalIndent(s, "", "  ")
}

func typeSchema(t reflect.Type) map[string]any {
	switch t.Kind() {
	case reflect.Pointer:
		return typeSchema(t.Elem())
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int64:
		return map[string]any{"type": "integer"}
	case reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice:
		return map[string]any{"type": "array", "items": typeSchema(t.Elem())}
	case reflect.Map:
		return map[string]any{"type": "object", "additionalProperties": typeSchema(t.Elem())}
	case reflect.Struct:
		props := map[string]any{}
		var required []string
		for i := range t.NumField() {
			f := t.Field(i)
			name, opts, _ := strings.Cut(f.Tag.Get("json"), ",")
			if name == "" || name == "-" {
				continue
			}
			props[name] = typeSchema(f.Type)
			if !strings.Contains(opts, "omitempty") {
				required = append(required, name)
			}
		}
		sort.Strings(required)
		out := map[string]any{"type": "object", "properties": props}
		if len(required) > 0 {
			out["required"] = required
		}
		return out
	}
	return map[string]any{}
}
