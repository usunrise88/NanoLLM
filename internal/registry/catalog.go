package registry

import (
	"bytes"
	_ "embed"
	"fmt"

	"gopkg.in/yaml.v3"
)

//go:embed catalog.yaml
var builtinCatalog []byte

// Builtin returns the catalog compiled into the binary.
func Builtin() ([]Manifest, error) {
	return parseCatalog(builtinCatalog)
}

func parseCatalog(b []byte) ([]Manifest, error) {
	var out []Manifest
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	if err := dec.Decode(&out); err != nil {
		return nil, fmt.Errorf("parse catalog: %w", err)
	}
	seen := map[string]bool{}
	for i, m := range out {
		if m.ID == "" {
			return nil, fmt.Errorf("catalog entry %d has no id", i)
		}
		if seen[m.ID] {
			return nil, fmt.Errorf("catalog lists %q twice", m.ID)
		}
		seen[m.ID] = true
		if _, ok := m.File(RoleModel); !ok {
			return nil, fmt.Errorf("catalog entry %q has no model file", m.ID)
		}
		switch m.Format {
		case FormatGGUF, FormatORTGenAI:
		default:
			return nil, fmt.Errorf("catalog entry %q: unknown format %q", m.ID, m.Format)
		}
		switch m.Commercial {
		case CommercialPermitted, CommercialUnconfirmed:
		default:
			return nil, fmt.Errorf("catalog entry %q: commercial must be %s or %s, got %q",
				m.ID, CommercialPermitted, CommercialUnconfirmed, m.Commercial)
		}
	}
	return out, nil
}

// Find returns the catalog entry with this id.
func Find(catalog []Manifest, id string) (Manifest, bool) {
	for _, m := range catalog {
		if m.ID == id {
			return m, true
		}
	}
	return Manifest{}, false
}
