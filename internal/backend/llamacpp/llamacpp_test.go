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
