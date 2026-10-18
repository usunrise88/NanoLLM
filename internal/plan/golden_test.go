package plan

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/registry"
)

var update = flag.Bool("update", false, "rewrite the golden plans in testdata/plan")

const goldenDir = "../../testdata/plan"

// e4b is the catalog's E4B with its measured inventory.
func e4b(t testing.TB) Model {
	t.Helper()
	cat, err := registry.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := registry.Find(cat, "gemma-4-e4b-qat")
	if !ok {
		t.Fatal("no E4B in the catalog")
	}
	model, err := ModelFrom(m.ID, m.Inventory)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

// TestGoldenPlans plans E4B under every preset on its synthetic device and
// compares with testdata/plan/<preset>.json. A change in the planner shows up
// as a diff of these files; -update rewrites them.
func TestGoldenPlans(t *testing.T) {
	model := e4b(t)
	for _, name := range Presets {
		t.Run(name, func(t *testing.T) {
			preset, err := LookupPreset(name)
			if err != nil {
				t.Fatal(err)
			}
			p, err := Make(Request{Model: model, Preset: preset})
			if err != nil {
				t.Fatal(err)
			}
			got, err := json.MarshalIndent(p, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, '\n')
			path := filepath.Join(goldenDir, strings.ReplaceAll(name, "+", "_")+".json")
			if *update {
				if err := os.MkdirAll(goldenDir, 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, got, 0o644); err != nil {
					t.Fatal(err)
				}
				return
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("%v (run with -update to create it)", err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("plan for %s differs from %s; rerun with -update and review the diff", name, path)
			}
		})
	}
}
