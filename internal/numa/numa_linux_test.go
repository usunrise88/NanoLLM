//go:build linux

package numa

import (
	"slices"
	"testing"
)

func TestExecEnv(t *testing.T) {
	base := []string{"PATH=/bin", "HOME=/h"}
	got := execEnv(Spec{Mode: "interleave"}, base)
	if !slices.Contains(got, EnvApplied+"=1") || slices.Contains(got, EnvGGML+"=distribute") {
		t.Errorf("interleave: %v", got)
	}
	got = execEnv(Spec{Mode: "distribute", Threads: 16}, base)
	for _, want := range []string{EnvApplied + "=1", EnvGGML + "=distribute", EnvRepackThreads + "=16"} {
		if !slices.Contains(got, want) {
			t.Errorf("distribute: %q missing from %v", want, got)
		}
	}
	// What the person who started the process set stays.
	got = execEnv(Spec{Mode: "distribute", Threads: 16}, append(slices.Clone(base), EnvGGML+"=isolate", EnvRepackThreads+"=4"))
	if slices.Contains(got, EnvGGML+"=distribute") || slices.Contains(got, EnvRepackThreads+"=16") {
		t.Errorf("explicit settings overridden: %v", got)
	}
	if len(base) != 2 {
		t.Errorf("the input slice grew: %v", base)
	}
	if got = execEnv(Spec{Mode: "distribute"}, base); slices.ContainsFunc(got, func(e string) bool {
		return len(e) > len(EnvRepackThreads) && e[:len(EnvRepackThreads)] == EnvRepackThreads
	}) {
		t.Errorf("no thread count asked, but %v", got)
	}
}
