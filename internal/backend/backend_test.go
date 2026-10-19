package backend

import "testing"

func TestWithDefaultThreads(t *testing.T) {
	for _, tc := range []struct {
		name string
		spec LoadSpec
		want bool // threads filled in
	}{
		{"cpu, no offload", LoadSpec{Device: "llamacpp:cpu"}, true},
		{"cpu with the run default of -1 layers", LoadSpec{Device: "cpu", GPULayers: -1}, true},
		{"no device", LoadSpec{GPULayers: -1}, true},
		{"partial offload", LoadSpec{Device: "llamacpp:vulkan0", GPULayers: 20}, true},
		{"full offload", LoadSpec{Device: "llamacpp:vulkan0", GPULayers: -1}, false},
		{"explicit", LoadSpec{Device: "llamacpp:cpu", Threads: 3}, true},
	} {
		got := tc.spec.WithDefaultThreads()
		if (got.Threads > 0) != tc.want {
			t.Errorf("%s: threads %d, want filled=%v", tc.name, got.Threads, tc.want)
		}
		if tc.spec.Threads != 0 && got.Threads != tc.spec.Threads {
			t.Errorf("%s: explicit %d changed to %d", tc.name, tc.spec.Threads, got.Threads)
		}
	}
}
