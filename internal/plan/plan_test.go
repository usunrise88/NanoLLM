package plan

import "testing"

func TestEveryDefinedPresetParses(t *testing.T) {
	for _, name := range Presets {
		p, err := LookupPreset(name)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if p.Budget.Total() <= 0 {
			t.Errorf("%s: empty budget", name)
		}
	}
}

func TestParsePreset(t *testing.T) {
	cases := []struct {
		name string
		want Budget
	}{
		{"ram-1.5g", Budget{RAM: 3 * GiB / 2}},
		{"vram-6g+ram-1g", Budget{VRAM: 6 * GiB, RAM: GiB}},
		{"uma-8g", Budget{Unified: 8 * GiB}},
	}
	for _, c := range cases {
		p, err := ParsePreset(c.name)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if p.Budget != c.want {
			t.Errorf("%s: got %+v, want %+v", c.name, p.Budget, c.want)
		}
	}
}

func TestParsePresetRejects(t *testing.T) {
	for _, name := range []string{
		"", "ram", "ram-4", "ram-0g", "ram--1g", "disk-4g",
		"ram-1g+ram-2g", "uma-4g+ram-1g",
	} {
		if _, err := ParsePreset(name); err == nil {
			t.Errorf("%q: accepted", name)
		}
	}
}

func TestLookupPresetIsClosed(t *testing.T) {
	// Well-formed but not one of the measured budgets.
	if _, err := LookupPreset("ram-5g"); err == nil {
		t.Fatal("ram-5g accepted as a preset")
	}
}

func TestParseMode(t *testing.T) {
	for in, want := range map[string]Mode{
		"":           ModeAuto,
		"auto":       ModeAuto,
		"M1":         VRAMFull,
		"m9":         MMap,
		"M10":        Stream,
		"ram-full":   RAMFull,
		"vram-split": VRAMSplit,
	} {
		got, err := ParseMode(in)
		if err != nil || got != want {
			t.Errorf("ParseMode(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	if _, err := ParseMode("M11"); err == nil {
		t.Error("M11 accepted")
	}
}

func TestPlannerOrderSkipsResearch(t *testing.T) {
	for _, m := range PlannerOrder {
		info, ok := Info(m)
		if !ok {
			t.Fatalf("%s is not in the mode table", m)
		}
		if info.Stage == StageResearch {
			t.Errorf("planner would pick research mode %s", m)
		}
	}
}
