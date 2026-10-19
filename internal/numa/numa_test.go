package numa

import (
	"reflect"
	"strings"
	"testing"
)

func cpus(s string) []int {
	var out []int
	for _, f := range strings.Split(s, ",") {
		if f == "" {
			return nil
		}
		n := 0
		for _, c := range f {
			if c < '0' || c > '9' {
				return nil
			}
			n = n*10 + int(c-'0')
		}
		out = append(out, n)
	}
	return out
}

func TestParse(t *testing.T) {
	for in, want := range map[string]Spec{
		"":                      {Mode: "off"},
		"off":                   {Mode: "off"},
		"interleave":            {Mode: "interleave"},
		"node:2":                {Mode: "node", Node: 2},
		"interleave;cpus=0,1,2": {Mode: "interleave", CPUs: []int{0, 1, 2}},
		"node:1;cpus=4":         {Mode: "node", Node: 1, CPUs: []int{4}},
	} {
		got, err := Parse(in, cpus)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Errorf("%q: %+v, %v; want %+v", in, got, err, want)
		}
	}
	for _, bad := range []string{"all", "node:", "node:x", "interleave;cpus=", "interleave;cpus=a"} {
		if _, err := Parse(bad, cpus); err == nil {
			t.Errorf("%q should be refused", bad)
		}
	}
}

func TestIsZero(t *testing.T) {
	if !(Spec{}).IsZero() || !(Spec{Mode: "off"}).IsZero() {
		t.Error("off asks for nothing")
	}
	if (Spec{Mode: "interleave"}).IsZero() || (Spec{CPUs: []int{0}}).IsZero() {
		t.Error("interleave and pinning ask for something")
	}
}
