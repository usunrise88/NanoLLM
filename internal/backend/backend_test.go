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

func TestWithDefaultThreadsBatch(t *testing.T) {
	saved := defaultBatchThreads
	defer func() { defaultBatchThreads = saved }()
	defaultBatchThreads = func() int { return 20 }
	cpu := LoadSpec{Device: "llamacpp:cpu"}
	if got := cpu.WithDefaultThreads(); got.ThreadsBatch != max(got.Threads, 20) {
		t.Errorf("neither set: batch %d with %d decode threads, want 20", got.ThreadsBatch, got.Threads)
	}
	// An explicit decode team keeps the batch team with it: the user limited the CPU.
	if got := (LoadSpec{Device: "llamacpp:cpu", Threads: 3}).WithDefaultThreads(); got.ThreadsBatch != 3 {
		t.Errorf("threads=3: batch %d, want 3", got.ThreadsBatch)
	}
	if got := (LoadSpec{Device: "llamacpp:cpu", ThreadsBatch: 8}).WithDefaultThreads(); got.ThreadsBatch != 8 {
		t.Errorf("batch=8: batch %d, want 8", got.ThreadsBatch)
	}
	// A model fully on the GPU runs no CPU matrix work: no team to size.
	if got := (LoadSpec{Device: "llamacpp:vulkan0", GPULayers: -1}).WithDefaultThreads(); got.Threads != 0 || got.ThreadsBatch != 0 {
		t.Errorf("full offload: %d/%d, want 0/0", got.Threads, got.ThreadsBatch)
	}
	defaultBatchThreads = func() int { return 0 }
	if got := cpu.WithDefaultThreads(); got.ThreadsBatch != got.Threads {
		t.Errorf("not hybrid: batch %d, decode %d", got.ThreadsBatch, got.Threads)
	}
}
