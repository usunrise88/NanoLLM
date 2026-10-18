package gguf

import "fmt"

// Type is a ggml tensor type, numbered as ggml.h numbers it.
type Type uint32

// The ggml types of the pinned tag (b11600, ggml.h). Gaps are ids that were
// removed upstream; a file that uses one is rejected, as gguf.cpp does.
const (
	TypeF32    Type = 0
	TypeF16    Type = 1
	TypeQ4_0   Type = 2
	TypeQ4_1   Type = 3
	TypeQ5_0   Type = 6
	TypeQ5_1   Type = 7
	TypeQ8_0   Type = 8
	TypeQ8_1   Type = 9
	TypeQ2_K   Type = 10
	TypeQ3_K   Type = 11
	TypeQ4_K   Type = 12
	TypeQ5_K   Type = 13
	TypeQ6_K   Type = 14
	TypeQ8_K   Type = 15
	TypeIQ2XXS Type = 16
	TypeIQ2XS  Type = 17
	TypeIQ3XXS Type = 18
	TypeIQ1S   Type = 19
	TypeIQ4NL  Type = 20
	TypeIQ3S   Type = 21
	TypeIQ2S   Type = 22
	TypeIQ4XS  Type = 23
	TypeI8     Type = 24
	TypeI16    Type = 25
	TypeI32    Type = 26
	TypeI64    Type = 27
	TypeF64    Type = 28
	TypeIQ1M   Type = 29
	TypeBF16   Type = 30
	TypeTQ1_0  Type = 34
	TypeTQ2_0  Type = 35
	TypeMXFP4  Type = 39
	TypeNVFP4  Type = 40
	TypeQ1_0   Type = 41
	TypeQ2_0   Type = 42
)

// typeInfo is one row of ggml's type_traits: how many elements a block holds
// and how many bytes it takes.
type typeInfo struct {
	name      string
	blockSize int64
	typeSize  int64
}

// typeTable is ggml.c type_traits for b11600. The byte sizes are sizeof of
// the block structs in ggml-common.h, which static_asserts pin; a new tag
// that changes one changes the static_assert first.
var typeTable = map[Type]typeInfo{
	TypeF32:    {"f32", 1, 4},
	TypeF16:    {"f16", 1, 2},
	TypeQ4_0:   {"q4_0", 32, 18},
	TypeQ4_1:   {"q4_1", 32, 20},
	TypeQ5_0:   {"q5_0", 32, 22},
	TypeQ5_1:   {"q5_1", 32, 24},
	TypeQ8_0:   {"q8_0", 32, 34},
	TypeQ8_1:   {"q8_1", 32, 36},
	TypeQ2_K:   {"q2_K", 256, 84},
	TypeQ3_K:   {"q3_K", 256, 110},
	TypeQ4_K:   {"q4_K", 256, 144},
	TypeQ5_K:   {"q5_K", 256, 176},
	TypeQ6_K:   {"q6_K", 256, 210},
	TypeQ8_K:   {"q8_K", 256, 292},
	TypeIQ2XXS: {"iq2_xxs", 256, 66},
	TypeIQ2XS:  {"iq2_xs", 256, 74},
	TypeIQ3XXS: {"iq3_xxs", 256, 98},
	TypeIQ1S:   {"iq1_s", 256, 50},
	TypeIQ4NL:  {"iq4_nl", 32, 18},
	TypeIQ3S:   {"iq3_s", 256, 110},
	TypeIQ2S:   {"iq2_s", 256, 82},
	TypeIQ4XS:  {"iq4_xs", 256, 136},
	TypeI8:     {"i8", 1, 1},
	TypeI16:    {"i16", 1, 2},
	TypeI32:    {"i32", 1, 4},
	TypeI64:    {"i64", 1, 8},
	TypeF64:    {"f64", 1, 8},
	TypeIQ1M:   {"iq1_m", 256, 56},
	TypeBF16:   {"bf16", 1, 2},
	TypeTQ1_0:  {"tq1_0", 256, 54},
	TypeTQ2_0:  {"tq2_0", 256, 66},
	TypeMXFP4:  {"mxfp4", 32, 17},
	TypeNVFP4:  {"nvfp4", 64, 36},
	TypeQ1_0:   {"q1_0", 128, 18},
	TypeQ2_0:   {"q2_0", 64, 18},
}

// Known reports whether t is a type of the pinned tag.
func (t Type) Known() bool { _, ok := typeTable[t]; return ok }

// String is ggml's type name (ggml_type_name), which is also how KV cache
// types are spelled in the configuration ("f16", "q8_0").
func (t Type) String() string {
	if info, ok := typeTable[t]; ok {
		return info.name
	}
	return fmt.Sprintf("type(%d)", uint32(t))
}

// BlockSize is the number of elements one block holds; 0 for an unknown type.
func (t Type) BlockSize() int64 { return typeTable[t].blockSize }

// TypeSize is the bytes of one block; 0 for an unknown type.
func (t Type) TypeSize() int64 { return typeTable[t].typeSize }

// RowSize is the bytes of n elements, which must be a whole number of blocks
// (ggml_row_size). The KV cache is sized with it.
func (t Type) RowSize(n int64) int64 {
	info := typeTable[t]
	if info.blockSize == 0 {
		return 0
	}
	return n / info.blockSize * info.typeSize
}

// ParseType accepts a ggml type name ("q8_0", "f16").
func ParseType(name string) (Type, error) {
	for t, info := range typeTable {
		if info.name == name {
			return t, nil
		}
	}
	return 0, fmt.Errorf("unknown ggml type %q", name)
}
