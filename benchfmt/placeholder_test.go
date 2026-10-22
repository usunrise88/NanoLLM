package benchfmt

import "testing"

func TestPlaceholder(t *testing.T) {
	for s, want := range map[string]bool{
		"": true, " ": true, "To Be Filled By O.E.M.": true, "to be filled by o.e.m": true, "Default string": true,
		"System Product Name": true, "None": true, "ASRockRack": false, "EPYCD8-2T": false, "ThinkPad X1": false,
	} {
		if got := Placeholder(s); got != want {
			t.Errorf("%q: %v", s, got)
		}
	}
}
