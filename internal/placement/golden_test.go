package placement

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
)

var update = flag.Bool("update", false, "rewrite the golden plans in testdata/plan/tune")

const goldenDir = "../../testdata/plan/tune"

// The plans the three modes give on the machines of docs/OPTIMIZE.md, without a
// profile: the limits of the mode alone. A change in the planner or in a mode's
// limits shows up as a diff of these files; -update rewrites them.
func TestGoldenModePlans(t *testing.T) {
	iris := backend.Device{Backend: "llamacpp", ID: "vulkan1", Type: backend.GPU, Name: "Iris Xe", MemoryTotal: 16 * gib, MemoryFree: 15 * gib,
		Unified: true, Features: []string{backend.FeatureFlashAttnSmallBatch}}
	cap4 := 4*gib + 100<<20
	modes := map[string]Tune{
		"silent":   {Source: SourceMode, Preset: "ram-4g", MinNCtx: 4608, Threads: 2, ThreadsBatch: 2},
		"optimal":  {Source: SourceMode, MemCap: cap4, MinNCtx: 4608, Threads: 7, ThreadsBatch: 7},
		"powerful": {Source: SourceMode, MinNCtx: 4608},
	}
	cases := []struct {
		name   string
		mode   string
		device backend.Device
	}{
		{"silent-cpu", "silent", cpu},
		{"optimal-cpu", "optimal", cpu},
		{"optimal-cuda", "optimal", rtx},
		{"optimal-igpu", "optimal", iris},
		{"powerful-cpu", "powerful", cpu},
		{"powerful-cuda", "powerful", rtx},
		{"powerful-igpu", "powerful", iris},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := resolve(t, Input{Device: c.device, Tune: modes[c.mode], GOOS: "windows"})
			if c.mode == "powerful" {
				// The powerful mode takes the machine's default teams
				// (hwinfo.DefaultThreads): the machine that runs the test, not the one
				// the plan is about.
				r.Spec.Threads, r.Spec.ThreadsBatch = 0, 0
			}
			got, err := json.MarshalIndent(struct {
				Plan any `json:"plan"`
				Spec any `json:"spec"`
			}{r.Plan, r.Spec}, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, '\n')
			path := filepath.Join(goldenDir, c.name+".json")
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
				t.Errorf("plan for %s differs from %s; rerun with -update and review the diff", c.name, path)
			}
			// Whatever else changes, the optimal mode stays within its limit.
			if c.mode == "optimal" {
				if cb := r.Plan.Memory.Combined; cb == nil || cb.Used > cb.Budget {
					t.Errorf("combined %+v", cb)
				}
			}
			if c.mode != "powerful" && r.Plan.NCtx < 4608 {
				t.Errorf("context %d", r.Plan.NCtx)
			}
		})
	}
}
