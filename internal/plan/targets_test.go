package plan

import "testing"

func TestTargetForDerivedBudgets(t *testing.T) {
	for _, tc := range []struct {
		preset string
		want   Target
	}{
		{"ram-8g", Target{32768, 1}},        // defined
		{"ram-72g", Target{32768, 1}},       // above every row: the largest
		{"ram-5g", Target{8192, 1}},         // between rows: the one below
		{"ram-0.5g", Target{FloorNCtx, 1}},  // below every row: the floor
		{"vram-24g", Target{32768, 4}},      // another kind
		{"vram-5g+ram-2g", Target{8192, 1}}, // a combined budget
		{"uma-64g", Target{32768, 1}},       // unified
		{"nonsense", Target{FloorNCtx, 1}},  // not a preset
	} {
		if got := TargetFor(tc.preset); got != tc.want {
			t.Errorf("%s: %+v, want %+v", tc.preset, got, tc.want)
		}
	}
}
