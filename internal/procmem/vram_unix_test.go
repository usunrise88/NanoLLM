//go:build !windows

package procmem

import "testing"

func TestParseComputeApps(t *testing.T) {
	const out = "4242, 3560\n4242, 12\n77, 900\n4243, [N/A]\n\n"
	v, err := parseComputeApps(out, 4242)
	if err != nil || v.Dedicated != (3560+12)<<20 || v.Shared != 0 || v.Source != SourceNvidiaSMI {
		t.Fatalf("%+v %v", v, err)
	}
	if v, _ := parseComputeApps(out, 4243); v.Dedicated != 0 {
		t.Fatalf("N/A counted: %+v", v)
	}
}
