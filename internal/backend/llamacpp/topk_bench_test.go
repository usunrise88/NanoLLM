package llamacpp

import (
	"math/rand"
	"testing"
)

// BenchmarkTopOfTopK is the drafter's per-token host work on a Gemma 4 row
// (262144 logits): what each MTP draft round costs the CPU.
func BenchmarkTopOfTopK(b *testing.B) {
	row := make([]float32, 262144)
	r := rand.New(rand.NewSource(1))
	for i := range row {
		row[i] = float32(r.NormFloat64() * 3)
	}
	b.SetBytes(int64(len(row) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		topOfTopK(row, draftTopK)
	}
}
