package tune

import (
	"os"
	"path/filepath"
	"testing"
)

// repoRoot finds the directory with go.mod, so a cell's source can be checked.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, _ := os.Getwd()
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("no go.mod above the test")
		}
		dir = parent
	}
}

// Every cell and baseline names the file that holds its measurement (rule 2 of
// docs/AUTOTUNE.md), and the file must exist.
func TestMatrixSourcesExist(t *testing.T) {
	root := repoRoot(t)
	for _, c := range Default.Cells {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(c.Source))); err != nil {
			t.Errorf("cell %s %v: source %s: %v", c.Lever, c.Class, c.Source, err)
		}
	}
	for _, b := range Default.Baselines {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(b.Source))); err != nil {
			t.Errorf("baseline %s %v: source %s: %v", b.Device, b.Class, b.Source, err)
		}
	}
}

var (
	// The three machines of docs/OPTIMIZE.md §3.8 that the matrix has no cell for.
	meteorLakeCPU = Class{AttrCPUArch: "x86_64", AttrCPUVariant: "alderlake", AttrHybrid: "true", AttrNUMA: "1", AttrOS: "windows"}
	zen4CPU       = Class{AttrCPUArch: "x86_64", AttrCPUVariant: "zen4", AttrHybrid: "false", AttrNUMA: "1", AttrOS: "linux"}
	radeon780M    = Class{AttrGPUBackend: "vulkan", AttrGPUVendor: "amd", AttrGPUMemory: "shared", AttrOS: "windows"}
	rtx4070       = Class{AttrGPUBackend: "cuda", AttrGPUVendor: "nvidia", AttrGPUMemory: "own", AttrGPUCC: "89", AttrOS: "windows"}
	irisXe        = Class{AttrGPUBackend: "vulkan", AttrGPUVendor: "intel", AttrGPUMemory: "shared", AttrOS: "windows"}
	rtx3070ti     = Class{AttrGPUBackend: "cuda", AttrGPUVendor: "nvidia", AttrGPUMemory: "own", AttrGPUCC: "86", AttrOS: "windows"}
)

func TestLookupFindsTheMeasuredClass(t *testing.T) {
	m, ok := Default.Lookup("mtp_gpu", rtx3070ti)
	if !ok || m.Source != SourceClass || m.Distance != 0 || m.Cell.Variant["draft_max"] != 3 {
		t.Fatalf("RTX 3070 Ti: %+v ok=%v", m, ok)
	}
	m, ok = Default.Lookup("mtp_gpu", irisXe)
	if !ok || m.Source != SourceClass || m.Cell.Variant["draft_max"] != 2 {
		t.Fatalf("Iris Xe: %+v", m)
	}
	m, ok = Default.Lookup("repack", Class{AttrCPUVariant: "alderlake"})
	if !ok || m.Source != SourceClass || m.Cell.Verdict != Win {
		t.Fatalf("repack on alderlake: %+v", m)
	}
}

// The principle of §3.8: a machine the matrix has no cell for gets the nearest.
func TestUnknownHardwareInherits(t *testing.T) {
	// Meteor Lake runs the alderlake variant of ggml: the same class.
	if m, ok := Default.Lookup("mtp_cpu", meteorLakeCPU); !ok || m.Source != SourceClass {
		t.Fatalf("Core Ultra 7 155H: %+v", m)
	}
	// A Zen 4 part has a superset of the haswell extensions (and not alderlake's
	// AVX-VNNI): the haswell cell is the nearer.
	m, ok := Default.Lookup("mtp_cpu", zen4CPU)
	if !ok || m.Source != SourceInherited || m.Cell.Class[AttrCPUVariant] != "haswell" {
		t.Fatalf("Zen 4 MTP: %+v ok=%v", m, ok)
	}
	// An Ada card inherits the Ampere cell, and the compute capability breaks
	// no tie across architectures.
	m, ok = Default.Lookup("mmvq", rtx4070)
	if !ok || m.Source != SourceInherited || m.Cell.Class[AttrGPUCC] != "86" {
		t.Fatalf("RTX 4070 MMVQ: %+v ok=%v", m, ok)
	}
	m, ok = Default.Lookup("mtp_gpu", rtx4070)
	if !ok || m.Cell.Class[AttrGPUBackend] != "cuda" {
		t.Fatalf("RTX 4070 MTP: %+v ok=%v", m, ok)
	}
	// A Radeon with shared memory takes the Iris cell for device sampling (same
	// backend and memory, another vendor), not the CUDA one.
	m, ok = Default.Lookup("device_sampling", radeon780M)
	if !ok || m.Cell.Class[AttrGPUBackend] != "vulkan" || m.Cell.Verdict != Loss {
		t.Fatalf("Radeon 780M sampling: %+v ok=%v", m, ok)
	}
	// ... and the Intel-only large mat-vec of patch 0006 is a different vendor:
	// the cell is within reach but marked inherited, for the caller to treat
	// as a candidate to probe.
	m, ok = Default.Lookup("vk_matvec", radeon780M)
	if !ok || m.Source != SourceInherited {
		t.Fatalf("Radeon 780M vk_matvec: %+v ok=%v", m, ok)
	}
}

func TestACellOfAnotherBackendIsTooFar(t *testing.T) {
	if m, ok := Default.Lookup("cuda_graph", irisXe); ok {
		t.Fatalf("a CUDA cell was lent to a Vulkan GPU: %+v", m)
	}
	if m, ok := Default.Lookup("stream", Class{AttrDisk: "nvme"}); !ok || m.Cell.Class[AttrDisk] != "hdd" {
		t.Fatalf("NVMe streaming: %+v ok=%v", m, ok)
	}
}

func TestLookupAllGivesTheNearestOfEachVariant(t *testing.T) {
	all := Default.LookupAll("mtp_cpu", Class{AttrCPUVariant: "haswell"})
	if len(all) < 2 {
		t.Fatalf("got %d variants", len(all))
	}
	if all[0].Distance != 0 {
		t.Fatalf("%+v", all[0])
	}
	seen := map[string]bool{}
	for _, a := range all {
		k := variantKey(a.Cell.Variant)
		if seen[k] {
			t.Fatalf("variant %s twice", k)
		}
		seen[k] = true
	}
}

// The blind test of docs/OPTIMIZE.md §7 hides what is known about the machine
// under test.
func TestWithoutHidesTheClassesOfTheMachine(t *testing.T) {
	blind := Default.Without(Class{AttrCPUVariant: "alderlake", AttrHybrid: "true"}, Class{AttrGPUBackend: "vulkan", AttrGPUVendor: "intel", AttrGPUMemory: "shared", AttrOS: "windows"}, Class{AttrGPUCC: "86", AttrGPUBackend: "cuda", AttrGPUMemory: "own", AttrGPUVendor: "nvidia"})
	for _, c := range blind.Cells {
		if c.Class[AttrCPUVariant] == "alderlake" || c.Class[AttrGPUCC] == "86" || c.Class[AttrGPUVendor] == "intel" {
			t.Errorf("cell left: %s %v", c.Lever, c.Class)
		}
	}
	if len(blind.Cells) == 0 || len(blind.Cells) >= len(Default.Cells) {
		t.Fatalf("%d cells of %d", len(blind.Cells), len(Default.Cells))
	}
	// Without its own cells the machine still finds a neighbour for MTP on the
	// CPU (haswell) and on the GPU (the CUDA cell is gone, so Vulkan is too far).
	if m, ok := blind.Lookup("mtp_cpu", meteorLakeCPU); !ok || m.Source != SourceInherited {
		t.Fatalf("blind alderlake MTP: %+v ok=%v", m, ok)
	}
	if b, _, ok := blind.BaselineFor("cpu", meteorLakeCPU); !ok || b.Class[AttrCPUVariant] != "haswell" {
		t.Fatalf("blind baseline: %+v ok=%v", b, ok)
	}
}

func TestBaselineFor(t *testing.T) {
	b, d, ok := Default.BaselineFor("cuda", rtx4070)
	if !ok || b.TG0 < 50 || d == 0 {
		t.Fatalf("%+v %d %v", b, d, ok)
	}
	if _, _, ok := Default.BaselineFor("metal", Class{AttrGPUBackend: "metal"}); ok {
		t.Fatal("a baseline for Metal out of nowhere")
	}
	// Decoding slows with the context; prefill of a long prompt may be faster
	// per token than the first 512 (the batches amortize the launches: 1.12 on
	// the RTX), but not by a factor.
	if b.TG4096() >= b.TG0 || b.DepthPP <= 0.5 || b.DepthPP >= 1.5 {
		t.Fatalf("depth factors out of range: %+v", b)
	}
}

// A speed does not pass from an iGPU on shared memory to a card with memory of
// its own (an order of magnitude apart), nor the other way round.
func TestBaselineDoesNotCrossMemoryKinds(t *testing.T) {
	nvidiaVulkan := Class{AttrGPUBackend: "vulkan", AttrGPUVendor: "nvidia", AttrGPUMemory: "own", AttrOS: "windows"}
	if b, d, ok := Default.BaselineFor("vulkan", nvidiaVulkan); ok {
		t.Errorf("an NVIDIA card on Vulkan got the baseline %v (distance %d)", b.Class, d)
	}
	iris := Class{AttrGPUBackend: "vulkan", AttrGPUVendor: "intel", AttrGPUMemory: "shared", AttrOS: "windows"}
	if _, _, ok := Default.BaselineFor("vulkan", iris); !ok {
		t.Error("the Iris Xe lost its own baseline")
	}
	// Levers still pass: MTP on a Vulkan card of any kind is worth trying.
	if m, ok := Default.Lookup("mtp_gpu", nvidiaVulkan); !ok || m.Source != SourceInherited {
		t.Errorf("MTP on an NVIDIA card through Vulkan: %+v ok=%v", m, ok)
	}
}

func TestIsaDistance(t *testing.T) {
	// A part with more extensions than the measured class is near; one with
	// fewer is far.
	if near, far := isaDistance("zen4", "haswell"), isaDistance("haswell", "zen4"); near >= far {
		t.Fatalf("zen4 vs haswell %d, haswell vs zen4 %d", near, far)
	}
	if isaDistance("alderlake", "alderlake") != 0 {
		t.Fatal("same variant, nonzero distance")
	}
}
