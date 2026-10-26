//go:build native

package llamacpp

import (
	"context"
	"math"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
)

// TestNativeCudaNanoAtRuntime answers the question of docs/OPTIMIZE.md, section 6
// (the kernel-shape candidates of the autotuner): does a switch of NanoLLM's
// ggml-cuda, set on a model that has already run (its CUDA graphs captured),
// reach the kernels, or does only a new load see it? It compares the logprobs
// of one position (bitwise, a different block shape sums in another order):
// the default, the default with MMVQ_NWARPS set at runtime, and a model loaded
// with it set. Run on a CUDA device with NANOLLM_TEST_DEVICE=cuda0.
func TestNativeCudaNanoAtRuntime(t *testing.T) {
	b := testBackend(t)
	devs, _ := b.Devices(context.Background())
	var cuda bool
	for _, d := range devs {
		if d.Has(backend.FeatureCudaNano) {
			cuda = true
		}
	}
	if !cuda || testOnCPU() {
		t.Skip("needs NanoLLM's ggml-cuda: NANOLLM_TEST_DEVICE=cuda0")
	}
	load := func(nano map[string]int) *model {
		spec := baseSpec(testModelPath(t))
		spec.CudaNano = nano
		m, err := b.Load(context.Background(), spec)
		if err != nil {
			t.Fatal(err)
		}
		return m.(*model)
	}
	logprobsOf := func(m *model) []float64 {
		prompt := tokenize(t, m, chat(m, "Name the planets of the solar system."))
		m.ClearSeq(0)
		if err := m.SetSampler(0, greedy, prompt); err != nil {
			t.Fatal(err)
		}
		g := generate(t, m, 0, prompt, 0, 8, 5)
		var out []float64
		for _, lp := range g.lps {
			out = append(out, lp.Logprob)
			for _, a := range lp.Top {
				out = append(out, a.Logprob)
			}
		}
		return out
	}
	same := func(a, b []float64) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
				return false
			}
		}
		return true
	}
	// -1 is the switch's default (the table), 0 would be upstream's block shape,
	// which for one token and not for 2-4 stops a draft's verification from
	// reproducing plain decoding: the other tests of the process must not see it.
	const name, value, nanoTable = "MMVQ_NWARPS", 12, -1

	m := load(nil)
	logprobsOf(m) // graphs captured
	base := logprobsOf(m)
	if !same(base, logprobsOf(m)) {
		t.Fatal("the default is not repeatable on this machine; the comparison means nothing")
	}
	if !setCudaNano(name, value) {
		t.Fatalf("no such switch %s", name)
	}
	runtime := logprobsOf(m)
	runtimeAgain := logprobsOf(m)
	setCudaNano(name, nanoTable)
	_ = m.Close()

	fresh := load(map[string]int{name: value})
	freshLP := logprobsOf(fresh)
	_ = fresh.Close()
	setCudaNano(name, nanoTable)

	t.Logf("default == runtime-switched: %v; fresh-switched == default: %v; runtime-switched == fresh-switched: %v (repeat: %v)",
		same(base, runtime), same(base, freshLP), same(runtime, freshLP), same(runtime, runtimeAgain))
	if same(base, freshLP) {
		t.Skipf("%s=%d does not change the numbers of this model on this card, so the test cannot tell", name, value)
	}
	// Measured 2026-10-23 on the RTX 3070 Ti Laptop with the n8 libraries: the
	// graphs a model has captured keep the kernels they were captured with, so a
	// switch set afterwards changes nothing until the model is loaded again.
	// The autotuner therefore treats a kernel-shape candidate as one more load
	// (docs/OPTIMIZE.md, section 6). If the DLL starts to drop its graphs on a
	// switch, this fails and the probe can try shapes on one load.
	if !same(runtime, base) {
		t.Errorf("a switch set at runtime now reaches the captured graphs: kernel-shape candidates no longer need a reload; update docs/OPTIMIZE.md and the probe")
	}
}
