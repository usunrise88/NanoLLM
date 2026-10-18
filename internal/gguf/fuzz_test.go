package gguf

import (
	"bytes"
	"testing"
)

// FuzzParse feeds the parser corrupt headers. Whatever it accepts must be a
// layout range-lock can trust: tensors back to back at their padded sizes,
// inside a data area that starts aligned.
func FuzzParse(f *testing.F) {
	seeds := []*Writer{
		TinyGemma4(),
		NewWriter(),
		NewWriter().Set(KeyAlignment, uint32(64)).Set("a", []string{"x", "y"}).
			Tensor("t", TypeQ8_0, []int64{32, 2}, nil),
	}
	for _, w := range seeds {
		b, err := w.Bytes()
		if err != nil {
			f.Fatal(err)
		}
		// The header is all the parser reads; dropping the tensor data keeps
		// the seeds small, and minimizing a large seed stalls the fuzzer.
		file, err := Parse(bytes.NewReader(b))
		if err != nil {
			f.Fatal(err)
		}
		f.Add(b[:file.DataOffset])
		f.Add(b[:file.DataOffset/2])
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		file, err := Parse(bytes.NewReader(data))
		if err != nil {
			return
		}
		if file.Alignment <= 0 || file.Alignment&(file.Alignment-1) != 0 {
			t.Fatalf("alignment %d", file.Alignment)
		}
		if len(file.Tensors) > 0 && file.DataOffset%file.Alignment != 0 {
			t.Fatalf("data offset %d not aligned to %d", file.DataOffset, file.Alignment)
		}
		var next int64
		for _, ti := range file.Tensors {
			if int64(ti.Offset) != next || ti.Size < 0 || !ti.Type.Known() {
				t.Fatalf("tensor %q at %d size %d, expected offset %d", ti.Name, ti.Offset, ti.Size, next)
			}
			next += file.PaddedSize(ti)
		}
		if next != file.DataSize {
			t.Fatalf("data size %d, tensors sum to %d", file.DataSize, next)
		}
		// Whatever hyperparameters a parsed file yields must not panic.
		if h, err := file.HParams(); err == nil {
			_ = h.EmbdVGQAMax()
			_ = h.KVLayers()
		}
		_, _ = BuildInventory(file)
	})
}
