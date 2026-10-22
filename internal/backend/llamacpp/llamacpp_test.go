package llamacpp

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
)

func TestCheckCommit(t *testing.T) {
	const full = "ace20f679e30a39c09a7b17bec14199855469633"
	for _, ok := range []string{"ace20f679", "ACE20F6", full} {
		if err := checkCommit(ok, full); err != nil {
			t.Errorf("%q: %v", ok, err)
		}
	}
	for _, bad := range []string{"", "ace20f", "unknown", "0123456789", "bce20f679"} {
		err := checkCommit(bad, full)
		if err == nil {
			t.Errorf("%q accepted", bad)
			continue
		}
		if !strings.Contains(err.Error(), "refused") {
			t.Errorf("%q: the error does not say the library is refused: %v", bad, err)
		}
	}
}

// devSeries is a run against a series of patches whose bindings have not been
// regenerated (NANOLLM_DEV_SERIES names the commit the test binary is stamped
// for): its commit is, by design, not the binding's.
func devSeries() bool { return os.Getenv("NANOLLM_DEV_SERIES") == PinnedCommit }

func TestPinnedDefaultsToBinding(t *testing.T) {
	// An unstamped build must check against the binding it was generated
	// from, never against nothing.
	if PinnedCommit == "" || PinnedTag == "" {
		t.Fatal("PinnedCommit or PinnedTag is empty")
	}
	if checkCommit(PinnedCommit, bindingCommit) != nil && !devSeries() {
		t.Fatalf("PinnedCommit %q is not the binding commit %q", PinnedCommit, bindingCommit)
	}
}

func TestCPUVariantOf(t *testing.T) {
	for in, want := range map[string]string{
		"/opt/nanollm/lib/libggml-cpu-sapphirerapids.so": "sapphirerapids",
		`C:\NanoLLM\lib\ggml-cpu-haswell.dll`:            "haswell",
		"lib/libggml-cpu-armv8.2_1.so":                   "armv8.2_1",
		"lib/libggml-cpu.so":                             "",
	} {
		if runtime.GOOS != "windows" && strings.Contains(in, `\`) {
			in = strings.ReplaceAll(in, `\`, "/")
		}
		if got := cpuVariantOf(in); got != want {
			t.Errorf("%s: %q, want %q", in, got, want)
		}
	}
}

func TestBackendLibPattern(t *testing.T) {
	for name, want := range map[string]string{
		"libggml-vulkan.so": "vulkan", "ggml-cuda.dll": "cuda", "libggml-rpc.so": "rpc",
		"libggml-base.so": "base", "libggml-cpu-haswell.so": "", "libggml.so": "", "libggml-vulkan.so.0": "",
		"libggml-metal.dylib": "metal", "libggml-metal.0.dylib": "", "libggml-metal.0.26.0.dylib": "", "libggml.dylib": "",
	} {
		m := backendLib.FindStringSubmatch(name)
		got := ""
		if m != nil {
			got = m[1]
		}
		if got != want {
			t.Errorf("%s: %q, want %q", name, got, want)
		}
	}
}

func TestCheckPattern(t *testing.T) {
	for _, ok := range []string{`per_layer_token_embd\.weight`, `blk\.[0-9]+\.ffn_.*`, `^token_embd`, `(attn|ffn)_norm`} {
		if err := checkPattern(ok); err != nil {
			t.Errorf("%q: %v", ok, err)
		}
	}
	// Valid RE2 that std::regex rejects with an exception, which aborts.
	for _, bad := range []string{"", `(?i)blk`, `(?P<n>blk)`, `blk\z`, `\p{L}`, `[`, "a\x00b"} {
		if err := checkPattern(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestTopOfTopK(t *testing.T) {
	row := make([]float32, 1000)
	for i := range row {
		row[i] = -10
	}
	row[7], row[500] = 5, 5-float32(math.Log(3)) // p(7) = 3/4 among the top 2
	tok, p := topOfTopK(row, 2)
	if tok != 7 || math.Abs(p-0.75) > 1e-6 {
		t.Errorf("top-2: token %d p %f, want 7 0.75", tok, p)
	}
	// Ten equal candidates: 1/10 each, however many follow.
	for i := range row {
		row[i] = 0
	}
	if _, p := topOfTopK(row, 10); math.Abs(p-0.1) > 1e-6 {
		t.Errorf("flat: p %f, want 0.1", p)
	}
}

func TestRegexEscape(t *testing.T) {
	if got, want := regexEscape(`<tool_call>{"a": [1]}.`), `<tool_call>\{"a": \[1\]\}\.`; got != want {
		t.Errorf("%s, want %s", got, want)
	}
}

func TestMemoryFromLog(t *testing.T) {
	log := []string{
		"load_tensors:          AMX model buffer size =  2491.17 MiB",
		"load_tensors:   CPU_Mapped model buffer size =  1872.00 MiB",
		"load_tensors:      Vulkan0 model buffer size =   100.50 MiB",
		"llama_kv_cache:        CPU KV buffer size =    20.00 MiB",
		"llama_kv_cache:    Vulkan0 KV buffer size =     8.00 MiB",
		"sched_reserve:        CPU compute buffer size =   559.03 MiB",
		"sched_reserve: Vulkan_Host compute buffer size =     1.00 MiB",
	}
	r := memoryFromLog(log, nil)
	if len(r.Devices) != 2 {
		t.Fatalf("devices: %+v", r.Devices)
	}
	cpu, gpu := r.Devices[0], r.Devices[1]
	mib := func(v float64) int64 { return int64(v * (1 << 20)) }
	if cpu.Device != "cpu" || cpu.Weights != mib(2491.17)+mib(1872) || cpu.KV != mib(20) || cpu.Compute != mib(559.03)+mib(1) {
		t.Errorf("cpu: %+v", cpu)
	}
	if gpu.Device != "llamacpp:vulkan0" || gpu.Weights != mib(100.5) || gpu.KV != mib(8) {
		t.Errorf("gpu: %+v", gpu)
	}
}

func TestLogLinesAssembled(t *testing.T) {
	var got []string
	untap := addLogTap(func(_ GgmlLogLevel, line string) { got = append(got, line) })
	defer untap()
	onLog(GgmlLogLevelInfo, "load_tensors: loading model tensors\n")
	onLog(GgmlLogLevelCont, ".")
	onLog(GgmlLogLevelCont, "..\n")
	onLog(GgmlLogLevelInfo, "two lines\nin one call\n")
	onLog(GgmlLogLevelWarn, "a line ")
	onLog(GgmlLogLevelCont, "in pieces\n")
	want := []string{"load_tensors: loading model tensors", "two lines", "in one call", "a line in pieces"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("lines %q, want %q", got, want)
	}
}

// TestMachOLoaderPaths checks the macOS archive when fetch-libs has put it
// in dist/ (it reads the files on any host).
func TestMachOLoaderPaths(t *testing.T) {
	dir := filepath.Join("..", "..", "..", "dist", "darwin_arm64", "lib")
	files, _ := filepath.Glob(filepath.Join(dir, "lib*.dylib"))
	if len(files) == 0 {
		t.Skip("no macOS libraries in dist/darwin_arm64/lib; run scripts/fetch-libs.sh darwin_arm64")
	}
	for _, f := range files {
		if fi, err := os.Lstat(f); err != nil || fi.Mode()&os.ModeSymlink != 0 {
			continue
		}
		if err := checkMachO(f); err != nil {
			t.Error(err)
		}
	}
}

func TestCPUFeatures(t *testing.T) {
	const info = "CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | REPACK_Q4_0_ROWS = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |"
	if got := cpuFeatures(info); len(got) != 1 || got[0] != backend.FeatureMultiRowQ4_0 {
		t.Errorf("features of %q: %v", info, got)
	}
	if got := cpuFeatures(strings.Replace(info, "REPACK_Q4_0_ROWS = 1 |", "", 1)); len(got) != 0 {
		t.Errorf("a build without the marker has features: %v", got)
	}
	if got := cpuFeatures(""); len(got) != 0 {
		t.Errorf("no information, but features %v", got)
	}
}

func TestGPUFeatures(t *testing.T) {
	const own = "CPU : SSE3 = 1 | AVX2 = 1 | REPACK = 1 | CUDA : ARCHS = 860 | USE_GRAPHS = 1 | FA_QUANTS = 1 | NANOLLM = 8 | NANO_GRAPH_SHAPES = 1 |"
	upstream := strings.Replace(own, "NANOLLM = 8 | NANO_GRAPH_SHAPES = 1 |", "", 1)
	if got := gpuFeatures("cuda0", own); len(got) != 2 || got[0] != backend.FeatureBackendSampling || got[1] != backend.FeatureCudaNano {
		t.Errorf("our ggml-cuda: %v", got)
	}
	for _, info := range []string{upstream, "", "CUDA :"} {
		if got := gpuFeatures("cuda0", info); len(got) != 1 || got[0] != backend.FeatureBackendSampling {
			t.Errorf("upstream's ggml-cuda %q: %v", info, got)
		}
	}
	// another backend's entry does not make a CUDA device ours
	if got := gpuFeatures("cuda0", "Vulkan : NANOLLM = 9 |"); len(got) != 1 {
		t.Errorf("a NANOLLM outside the CUDA entry: %v", got)
	}
	if got := gpuFeatures("vulkan0", own); len(got) != 0 {
		t.Errorf("a Vulkan device: %v", got)
	}
	if n := cudaNanoLevel("CUDA : NANOLLM = 12 | NANO_X = 1 |"); n != 12 {
		t.Errorf("level %d, want 12", n)
	}
}

func TestIGPUFeatures(t *testing.T) {
	const iris = "Intel(R) Iris(R) Xe Graphics"
	if got := igpuFeatures("vulkan0", iris, "windows"); len(got) != 1 || got[0] != backend.FeatureFlashAttnSmallBatch {
		t.Errorf("Iris Xe on Windows: %v", got)
	}
	for _, c := range [][3]string{
		{"vulkan0", iris, "linux"},                        // Mesa: not measured
		{"vulkan0", "AMD Radeon(TM) Graphics", "windows"}, // another vendor
		{"sycl0", iris, "windows"},                        // another backend
	} {
		if got := igpuFeatures(c[0], c[1], c[2]); len(got) != 0 {
			t.Errorf("%v: %v", c, got)
		}
	}
}

func TestUnifiedGPU(t *testing.T) {
	for _, c := range []struct {
		id, goos, goarch string
		want             bool
	}{
		{"mtl0", "darwin", "arm64", true},     // Apple Silicon: Metal maps the host's memory
		{"mtl0", "darwin", "amd64", false},    // an Intel Mac may carry a discrete GPU
		{"vulkan0", "darwin", "arm64", false}, // another backend (MoltenVK) says its own
		{"cuda0", "linux", "amd64", false},
		{"mtl0", "linux", "arm64", false},
	} {
		if got := unifiedGPU(c.id, c.goos, c.goarch); got != c.want {
			t.Errorf("%+v: %v, want %v", c, got, c.want)
		}
	}
}

func TestProbeBackendsMetalRegistersAsMTL(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"libggml-metal.dylib", "libggml-metal.0.dylib", "libggml-base.dylib"} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("not a library"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if got := probeBackends(dir, map[string]bool{"cpu": true, "mtl": true}); len(got) != 0 {
		t.Errorf("Metal registered as MTL, yet warned: %v", got)
	}
	if got := probeBackends(dir, map[string]bool{"cpu": true}); len(got) != 1 || !strings.HasPrefix(got[0], "libggml-metal.dylib:") {
		t.Errorf("an unregistered Metal library should warn once: %v", got)
	}
}
