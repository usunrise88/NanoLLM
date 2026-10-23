package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// Writer builds a GGUF file in memory. It exists for tests and fixtures: the
// parser's tests, synthetic models for the planner, and the corrupt files the
// fuzzer starts from. It writes what gguf.cpp writes — header, metadata,
// tensor table, padding to the alignment, then each tensor padded in turn — so
// a file it produces is one libllama could open.
type Writer struct {
	kv        []KV
	tensors   []writerTensor
	alignment int64
}

type writerTensor struct {
	name  string
	typ   Type
	shape []int64
	data  []byte
	// offset, when set, replaces the computed offset: tests use it to write
	// a tensor table that does not match the layout.
	offset *uint64
}

// NewWriter returns an empty file with the default alignment.
func NewWriter() *Writer { return &Writer{alignment: DefaultAlignment} }

// Set adds a metadata pair. v is a Go value of one of the GGUF types
// (uint8…float64, bool, string) or a slice of one; general.alignment also
// changes the padding the writer applies.
func (w *Writer) Set(key string, v any) *Writer {
	w.kv = append(w.kv, KV{Key: key, Value: Value{Data: v}})
	if a, ok := v.(uint32); ok && key == KeyAlignment && a > 0 {
		w.alignment = int64(a)
	}
	return w
}

// Tensor adds a tensor. A nil data is written as zeros of the tensor's size.
func (w *Writer) Tensor(name string, typ Type, shape []int64, data []byte) *Writer {
	w.tensors = append(w.tensors, writerTensor{name: name, typ: typ, shape: shape, data: data})
	return w
}

// FillPattern gives every tensor that has no data a deterministic pattern of
// its own, never all zeros, so a test can tell tensors apart after they have
// moved and tell a tensor from the padding.
func (w *Writer) FillPattern() *Writer {
	for i := range w.tensors {
		t := &w.tensors[i]
		if t.data != nil {
			continue
		}
		t.data = make([]byte, Size(t.typ, t.shape))
		x := uint32(2166136261)
		for _, c := range []byte(t.name) {
			x = (x ^ uint32(c)) * 16777619
		}
		for j := range t.data {
			x = x*1664525 + 1013904223
			t.data[j] = byte(x>>24) | 1
		}
	}
	return w
}

// Size is the byte size of a tensor of this type and shape; 0 for a shape or
// type the parser would reject, so the writer can still produce such a file.
func Size(typ Type, shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		if d < 0 || (d > 0 && n > math.MaxInt64/d) {
			return 0
		}
		n *= d
	}
	if typ.BlockSize() == 0 {
		return 0
	}
	return n / typ.BlockSize() * typ.TypeSize()
}

// Bytes encodes the file.
func (w *Writer) Bytes() ([]byte, error) {
	var b bytes.Buffer
	if err := w.Encode(&b); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// Encode writes the file to out.
func (w *Writer) Encode(out io.Writer) error {
	var b bytes.Buffer
	le := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { le(uint64(len(s))); b.WriteString(s) }

	b.WriteString(Magic)
	le(uint32(Version))
	le(int64(len(w.tensors)))
	le(int64(len(w.kv)))
	for _, kv := range w.kv {
		str(kv.Key)
		if err := encodeValue(&b, kv.Value.Data); err != nil {
			return fmt.Errorf("%s: %w", kv.Key, err)
		}
	}
	var offset int64
	for _, t := range w.tensors {
		str(t.name)
		le(uint32(len(t.shape)))
		for _, d := range t.shape {
			le(d)
		}
		le(uint32(t.typ))
		if t.offset != nil {
			le(*t.offset)
		} else {
			le(uint64(offset))
		}
		offset += pad(Size(t.typ, t.shape), w.alignment)
	}
	if len(w.tensors) > 0 {
		b.Write(make([]byte, pad(int64(b.Len()), w.alignment)-int64(b.Len())))
	}
	for _, t := range w.tensors {
		size := Size(t.typ, t.shape)
		data := t.data
		if data == nil {
			data = make([]byte, size)
		}
		if int64(len(data)) != size {
			return fmt.Errorf("tensor %s: %d bytes of data for a %d-byte tensor", t.name, len(data), size)
		}
		b.Write(data)
		b.Write(make([]byte, pad(size, w.alignment)-size))
	}
	_, err := out.Write(b.Bytes())
	return err
}

func encodeValue(b *bytes.Buffer, v any) error {
	le := func(v any) { _ = binary.Write(b, binary.LittleEndian, v) }
	str := func(s string) { le(uint64(len(s))); b.WriteString(s) }
	array := func(t ValueType, n int, each func(i int)) {
		le(uint32(ValArray))
		le(uint32(t))
		le(uint64(n))
		for i := range n {
			each(i)
		}
	}
	switch x := v.(type) {
	case uint8:
		le(uint32(ValUint8))
		le(x)
	case int8:
		le(uint32(ValInt8))
		le(x)
	case uint16:
		le(uint32(ValUint16))
		le(x)
	case int16:
		le(uint32(ValInt16))
		le(x)
	case uint32:
		le(uint32(ValUint32))
		le(x)
	case int32:
		le(uint32(ValInt32))
		le(x)
	case float32:
		le(uint32(ValFloat32))
		le(math.Float32bits(x))
	case bool:
		le(uint32(ValBool))
		le(x)
	case string:
		le(uint32(ValString))
		str(x)
	case uint64:
		le(uint32(ValUint64))
		le(x)
	case int64:
		le(uint32(ValInt64))
		le(x)
	case float64:
		le(uint32(ValFloat64))
		le(math.Float64bits(x))
	case []uint32:
		array(ValUint32, len(x), func(i int) { le(x[i]) })
	case []int32:
		array(ValInt32, len(x), func(i int) { le(x[i]) })
	case []float32:
		array(ValFloat32, len(x), func(i int) { le(math.Float32bits(x[i])) })
	case []bool:
		array(ValBool, len(x), func(i int) { le(x[i]) })
	case []string:
		array(ValString, len(x), func(i int) { str(x[i]) })
	default:
		return fmt.Errorf("unsupported metadata value %T", v)
	}
	return nil
}
