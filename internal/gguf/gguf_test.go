package gguf

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func parseBytes(t *testing.T, w *Writer) (*File, error) {
	t.Helper()
	b, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	return Parse(bytes.NewReader(b))
}

func TestRoundTrip(t *testing.T) {
	big := make([]string, MaxHeldArray+1)
	for i := range big {
		big[i] = "tok"
	}
	w := NewWriter().
		Set("general.architecture", "test").
		Set("u8", uint8(7)).Set("i8", int8(-7)).
		Set("u16", uint16(700)).Set("i16", int16(-700)).
		Set("u32", uint32(70000)).Set("i32", int32(-70000)).
		Set("u64", uint64(1)<<40).Set("i64", int64(-1)<<40).
		Set("f32", float32(0.5)).Set("f64", 0.25).
		Set("b", true).Set("s", "hello").
		Set("arr", []int32{1, 2, 3}).
		Set("pattern", []bool{true, false}).
		Set("tokens", big).
		Tensor("a.weight", TypeF32, []int64{3}, nil).
		Tensor("b.weight", TypeQ4_0, []int64{64, 2}, nil).
		Tensor("c.weight", TypeQ4_K, []int64{256}, nil)
	f, err := parseBytes(t, w)
	if err != nil {
		t.Fatal(err)
	}
	if f.Version != Version || f.Alignment != DefaultAlignment {
		t.Errorf("version %d alignment %d", f.Version, f.Alignment)
	}
	for key, want := range map[string]int64{
		"u8": 7, "i8": -7, "u16": 700, "i16": -700, "u32": 70000, "i32": -70000,
		"u64": 1 << 40, "i64": -1 << 40,
	} {
		if got, ok := f.Int(key); !ok || got != want {
			t.Errorf("%s = %d, %v; want %d", key, got, ok, want)
		}
	}
	if v, _ := f.Float("f32"); v != 0.5 {
		t.Errorf("f32 = %v", v)
	}
	if v, _ := f.Float("f64"); v != 0.25 {
		t.Errorf("f64 = %v", v)
	}
	if s, _ := f.String("s"); s != "hello" {
		t.Errorf("s = %q", s)
	}
	if xs, ok := f.Ints("arr"); !ok || len(xs) != 3 || xs[2] != 3 {
		t.Errorf("arr = %v", xs)
	}
	if bs, ok := f.Bools("pattern"); !ok || len(bs) != 2 || !bs[0] || bs[1] {
		t.Errorf("pattern = %v", bs)
	}
	// The tokenizer-sized array is skipped but its length survives.
	v, _ := f.Lookup("tokens")
	if !v.Skipped || v.Data != nil || v.Len != MaxHeldArray+1 {
		t.Errorf("tokens: %+v", v)
	}
	if n, _ := f.ArrayLen("tokens"); n != MaxHeldArray+1 {
		t.Errorf("tokens length %d", n)
	}

	wantSizes := []int64{12, 2 * 18 * 2, 144}
	var next uint64
	for i, ti := range f.Tensors {
		if ti.Size != wantSizes[i] || ti.Offset != next {
			t.Errorf("%s: size %d offset %d, want %d at %d", ti.Name, ti.Size, ti.Offset, wantSizes[i], next)
		}
		next += uint64(f.PaddedSize(ti))
	}
	if f.DataSize != int64(next) || f.DataOffset%DefaultAlignment != 0 {
		t.Errorf("data at %d size %d", f.DataOffset, f.DataSize)
	}
}

func TestCustomAlignment(t *testing.T) {
	w := NewWriter().Set(KeyAlignment, uint32(64)).
		Tensor("a.weight", TypeF32, []int64{3}, nil).
		Tensor("b.weight", TypeF32, []int64{3}, nil)
	f, err := parseBytes(t, w)
	if err != nil {
		t.Fatal(err)
	}
	if f.Alignment != 64 || f.Tensors[1].Offset != 64 || f.DataOffset%64 != 0 || f.DataSize != 128 {
		t.Errorf("alignment %d, second offset %d, data at %d size %d",
			f.Alignment, f.Tensors[1].Offset, f.DataOffset, f.DataSize)
	}
}

func TestOpenChecksDataArea(t *testing.T) {
	b, err := TinyGemma4().Bytes()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	write := func(name string, data []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, data, 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	f, err := Open(write("ok.gguf", b))
	if err != nil {
		t.Fatal(err)
	}
	if f.FileSize != int64(len(b)) || f.DataOffset+f.DataSize != f.FileSize {
		t.Errorf("file %d, data %d+%d", f.FileSize, f.DataOffset, f.DataSize)
	}
	if _, err := Open(write("short.gguf", b[:len(b)-1])); err == nil {
		t.Error("a truncated data area was accepted")
	}
	if _, err := Open(write("long.gguf", append(b, 0))); err == nil {
		t.Error("bytes after the last tensor were accepted")
	}
	if _, err := Open(write("header.gguf", b[:100])); !errors.Is(err, ErrFormat) {
		t.Errorf("truncated header: %v", err)
	}
}

func TestParseRejects(t *testing.T) {
	off := uint64(64)
	shifted := NewWriter().
		Tensor("a.weight", TypeF32, []int64{8}, nil).
		Tensor("b.weight", TypeF32, []int64{8}, nil)
	shifted.tensors[1].offset = &off

	cases := map[string]*Writer{
		"duplicate key":        NewWriter().Set("k", uint32(1)).Set("k", uint32(2)),
		"empty key":            NewWriter().Set("", uint32(1)),
		"alignment not u32":    NewWriter().Set(KeyAlignment, uint64(32)),
		"alignment not pow2":   NewWriter().Set(KeyAlignment, uint32(48)),
		"alignment zero":       NewWriter().Set(KeyAlignment, uint32(0)),
		"duplicate tensor":     NewWriter().Tensor("a", TypeF32, []int64{1}, nil).Tensor("a", TypeF32, []int64{1}, nil),
		"long name":            NewWriter().Tensor(strings.Repeat("x", MaxName), TypeF32, []int64{1}, nil),
		"five dims":            NewWriter().Tensor("a", TypeF32, []int64{1, 1, 1, 1, 1}, nil),
		"negative dim":         NewWriter().Tensor("a", TypeF32, []int64{-1}, nil),
		"removed type":         NewWriter().Tensor("a", Type(4), []int64{32}, nil),
		"unknown type":         NewWriter().Tensor("a", Type(99), []int64{32}, nil),
		"partial block":        NewWriter().Tensor("a", TypeQ4_0, []int64{48}, nil),
		"non-cumulative":       shifted,
		"overflowing elements": NewWriter().Tensor("a", TypeF32, []int64{1 << 40, 1 << 40}, nil),
	}
	for name, w := range cases {
		b, err := w.Bytes()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if _, err := Parse(bytes.NewReader(b)); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: got %v, want a format error", name, err)
		}
	}
}

func TestParseRejectsHeader(t *testing.T) {
	good, err := NewWriter().Set("k", uint32(1)).Bytes()
	if err != nil {
		t.Fatal(err)
	}
	patch := func(at int, b ...byte) []byte {
		c := bytes.Clone(good)
		copy(c[at:], b)
		return c
	}
	cases := map[string][]byte{
		"magic":      patch(0, 'G', 'G', 'U', 'X'),
		"version 1":  patch(4, 1, 0, 0, 0),
		"version 4":  patch(4, 4, 0, 0, 0),
		"big-endian": patch(4, 0, 0, 0, 3),
		"negative":   patch(8, 0, 0, 0, 0, 0, 0, 0, 0x80),
		"nested": append(bytes.Clone(good[:16]), // one KV: "k" = array of arrays
			1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 'k', 9, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
		"value type": append(bytes.Clone(good[:16]),
			1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 'k', 13, 0, 0, 0),
		"huge string": append(bytes.Clone(good[:16]),
			1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 'k', 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10),
	}
	// The nested and value-type cases claim one KV pair; fix the count.
	for _, k := range []string{"nested", "value type", "huge string"} {
		copy(cases[k][16:], []byte{1, 0, 0, 0, 0, 0, 0, 0})
	}
	for name, b := range cases {
		if _, err := Parse(bytes.NewReader(b)); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: got %v, want a format error", name, err)
		}
	}
}

func TestTypeTable(t *testing.T) {
	// The b11600 block structs, as ggml-common.h static_asserts them.
	for typ, want := range map[Type][2]int64{
		TypeF32: {1, 4}, TypeF16: {1, 2}, TypeBF16: {1, 2},
		TypeQ4_0: {32, 18}, TypeQ8_0: {32, 34}, TypeQ4_K: {256, 144}, TypeQ6_K: {256, 210},
		TypeMXFP4: {32, 17}, TypeNVFP4: {64, 36}, TypeQ1_0: {128, 18}, TypeQ2_0: {64, 18},
		TypeIQ4XS: {256, 136}, TypeTQ1_0: {256, 54},
	} {
		if typ.BlockSize() != want[0] || typ.TypeSize() != want[1] {
			t.Errorf("%s: block %d size %d, want %v", typ, typ.BlockSize(), typ.TypeSize(), want)
		}
		back, err := ParseType(typ.String())
		if err != nil || back != typ {
			t.Errorf("ParseType(%q) = %v, %v", typ, back, err)
		}
	}
	if got := TypeQ8_0.RowSize(1024); got != 1088 {
		t.Errorf("q8_0 row of 1024 = %d", got)
	}
	if Type(31).Known() {
		t.Error("a removed type is known")
	}
}

func TestHParams(t *testing.T) {
	f, err := parseBytes(t, TinyGemma4())
	if err != nil {
		t.Fatal(err)
	}
	h, err := f.HParams()
	if err != nil {
		t.Fatal(err)
	}
	if h.Arch != ArchGemma4 || h.BlockCount != 6 || h.KVLayers() != 4 || h.VocabSize != 256 || !h.TiedOutput {
		t.Errorf("%+v", h)
	}
	if !h.IsSWA(0) || h.IsSWA(5) || h.EmbdKGQA(0) != 32 || h.EmbdKGQA(5) != 64 || h.EmbdVGQAMax() != 64 {
		t.Errorf("per-layer: swa %v/%v k %d/%d vmax %d",
			h.IsSWA(0), h.IsSWA(5), h.EmbdKGQA(0), h.EmbdKGQA(5), h.EmbdVGQAMax())
	}
	if len(h.HeadCountKV) != 6 || h.FF(3) != 128 {
		t.Errorf("head_count_kv %v ff %d", h.HeadCountKV, h.FF(3))
	}

	bad := NewWriter().Set("general.architecture", "gemma4").
		Set("gemma4.block_count", uint32(2)).Set("gemma4.embedding_length", uint32(8)).
		Set("gemma4.attention.head_count", uint32(1)).
		Set("gemma4.attention.sliding_window_pattern", []bool{true})
	f, err = parseBytes(t, bad)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.HParams(); err == nil {
		t.Error("a pattern shorter than block_count was accepted")
	}
}
