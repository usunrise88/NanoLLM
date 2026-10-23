// Package gguf reads the header of a GGUF file: its metadata, its tensor table
// and where the tensor data starts (SPEC §4, docs/STAGE1.md 1.3).
//
// Only the header is read. Everything the planner and range-lock need — tensor
// names, types, byte sizes and file offsets, and the hyperparameters that size
// the KV cache — is there, so a 4 GB model is inventoried by reading a few
// megabytes of it. The tokenizer arrays, which are most of those megabytes,
// are skipped as they are read rather than held: their length is kept, their
// contents are not.
//
// The parser checks what gguf.cpp of the pinned tag checks, so a file it
// accepts is one libllama would accept too, and a file it rejects never
// reaches the native loader, where a malformed header ends in abort().
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// Magic and the versions this parser reads. Version 2 differs from 3 only in
// allowing big-endian files, which are rejected by the endianness check.
const (
	Magic            = "GGUF"
	Version          = 3
	DefaultAlignment = 32
)

// Limits gguf.cpp of the pinned tag also enforces (GGML_MAX_DIMS,
// GGML_MAX_NAME), and two of ours. MaxHeldArray is the longest array whose
// elements are kept: per-layer hyperparameters are tens of elements, the
// tokenizer is hundreds of thousands. MaxString bounds any one string, so a
// corrupt length cannot make the parser allocate gigabytes.
const (
	MaxDims      = 4
	MaxName      = 64
	MaxHeldArray = 4096
	MaxString    = 16 << 20
)

// KeyAlignment is the metadata key that overrides DefaultAlignment.
const KeyAlignment = "general.alignment"

// ValueType is the type of a metadata value, numbered as gguf.h numbers it.
type ValueType uint32

const (
	ValUint8   ValueType = 0
	ValInt8    ValueType = 1
	ValUint16  ValueType = 2
	ValInt16   ValueType = 3
	ValUint32  ValueType = 4
	ValInt32   ValueType = 5
	ValFloat32 ValueType = 6
	ValBool    ValueType = 7
	ValString  ValueType = 8
	ValArray   ValueType = 9
	ValUint64  ValueType = 10
	ValInt64   ValueType = 11
	ValFloat64 ValueType = 12
)

var valueTypeNames = [...]string{
	"u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "string", "array", "u64", "i64", "f64",
}

func (t ValueType) String() string {
	if int(t) < len(valueTypeNames) {
		return valueTypeNames[t]
	}
	return fmt.Sprintf("valuetype(%d)", uint32(t))
}

// size is the encoded size of a fixed-size value, 0 for strings and arrays.
func (t ValueType) size() int64 {
	switch t {
	case ValUint8, ValInt8, ValBool:
		return 1
	case ValUint16, ValInt16:
		return 2
	case ValUint32, ValInt32, ValFloat32:
		return 4
	case ValUint64, ValInt64, ValFloat64:
		return 8
	}
	return 0
}

// Value is one metadata value. For an array, Type is the element type and Len
// the element count; Data holds the elements as []any unless the array is
// longer than MaxHeldArray, in which case Data is nil and Skipped is set.
// Scalars are held as the Go type of their encoding (uint32, float32, string…).
type Value struct {
	Type    ValueType `json:"type"`
	Array   bool      `json:"array,omitempty"`
	Len     uint64    `json:"len,omitempty"`
	Skipped bool      `json:"skipped,omitempty"`
	Data    any       `json:"value,omitempty"`
}

// KV is one metadata pair, in file order.
type KV struct {
	Key   string `json:"key"`
	Value Value  `json:"value"`
}

// TensorInfo is one entry of the tensor table.
type TensorInfo struct {
	Name string `json:"name"`
	// Shape is ggml's ne[], innermost dimension first.
	Shape []int64 `json:"shape"`
	Type  Type    `json:"type"`
	// Offset is relative to the start of the data area, as the file stores it.
	Offset uint64 `json:"offset"`
	// Size is the tensor's bytes without the alignment padding that follows.
	Size int64 `json:"size"`
}

// Elements is the product of the shape.
func (t TensorInfo) Elements() int64 {
	n := int64(1)
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// File is a parsed header.
type File struct {
	Version   uint32
	KV        []KV
	Tensors   []TensorInfo
	Alignment int64
	// DataOffset is where the data area starts in the file; tensor offsets
	// are relative to it.
	DataOffset int64
	// DataSize is the sum of the tensors' padded sizes: the size the data
	// area must have.
	DataSize int64
	// FileSize is the size of the file on disk; 0 when the header came from a
	// stream of unknown length.
	FileSize int64
	// KVEnd is where the metadata ends and the tensor table starts; TableEnd
	// where the table ends, before the padding up to DataOffset. Relayout
	// copies the metadata between the fixed 24-byte start and KVEnd as it is.
	KVEnd    int64
	TableEnd int64

	index    map[string]int
	alignOff int64 // offset of the u32 value of general.alignment; 0 when absent
}

// AlignmentOffset is where the u32 value of general.alignment sits in the
// file; false when the key is absent and the alignment is the default.
func (f *File) AlignmentOffset() (int64, bool) { return f.alignOff, f.alignOff > 0 }

// FileOffset is where t's bytes start in the file.
func (f *File) FileOffset(t TensorInfo) int64 { return f.DataOffset + int64(t.Offset) }

// PaddedSize is t's size rounded up to the alignment: the stride from one
// tensor to the next in the data area.
func (f *File) PaddedSize(t TensorInfo) int64 { return pad(t.Size, f.Alignment) }

// Lookup returns the value of key.
func (f *File) Lookup(key string) (Value, bool) {
	i, ok := f.index[key]
	if !ok {
		return Value{}, false
	}
	return f.KV[i].Value, true
}

// String returns a string value.
func (f *File) String(key string) (string, bool) {
	v, ok := f.Lookup(key)
	if !ok || v.Array {
		return "", false
	}
	s, ok := v.Data.(string)
	return s, ok
}

// Int returns an integer value of any width. A one-element array counts, since
// converters write some per-layer keys as arrays even when every layer agrees.
func (f *File) Int(key string) (int64, bool) {
	v, ok := f.Lookup(key)
	if !ok {
		return 0, false
	}
	if v.Array {
		xs, ok := v.Data.([]any)
		if !ok || len(xs) != 1 {
			return 0, false
		}
		return toInt(xs[0])
	}
	return toInt(v.Data)
}

// Ints returns an integer array, or a scalar as a one-element slice.
func (f *File) Ints(key string) ([]int64, bool) {
	v, ok := f.Lookup(key)
	if !ok {
		return nil, false
	}
	if !v.Array {
		n, ok := toInt(v.Data)
		return []int64{n}, ok
	}
	xs, ok := v.Data.([]any)
	if !ok {
		return nil, false
	}
	out := make([]int64, len(xs))
	for i, x := range xs {
		if out[i], ok = toInt(x); !ok {
			return nil, false
		}
	}
	return out, true
}

// Bools returns a boolean array, or a scalar as a one-element slice.
func (f *File) Bools(key string) ([]bool, bool) {
	v, ok := f.Lookup(key)
	if !ok {
		return nil, false
	}
	if !v.Array {
		b, ok := v.Data.(bool)
		return []bool{b}, ok
	}
	xs, ok := v.Data.([]any)
	if !ok {
		return nil, false
	}
	out := make([]bool, len(xs))
	for i, x := range xs {
		if out[i], ok = x.(bool); !ok {
			return nil, false
		}
	}
	return out, true
}

// Float returns a floating-point value.
func (f *File) Float(key string) (float64, bool) {
	v, ok := f.Lookup(key)
	if !ok || v.Array {
		return 0, false
	}
	switch x := v.Data.(type) {
	case float32:
		return float64(x), true
	case float64:
		return x, true
	}
	return 0, false
}

// ArrayLen is the element count of an array value, held or skipped: the
// vocabulary size is the length of tokenizer.ggml.tokens.
func (f *File) ArrayLen(key string) (uint64, bool) {
	v, ok := f.Lookup(key)
	if !ok || !v.Array {
		return 0, false
	}
	return v.Len, true
}

// Tensor returns the tensor named name.
func (f *File) Tensor(name string) (TensorInfo, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

func toInt(x any) (int64, bool) {
	switch n := x.(type) {
	case uint8:
		return int64(n), true
	case int8:
		return int64(n), true
	case uint16:
		return int64(n), true
	case int16:
		return int64(n), true
	case uint32:
		return int64(n), true
	case int32:
		return int64(n), true
	case uint64:
		if n > math.MaxInt64 {
			return 0, false
		}
		return int64(n), true
	case int64:
		return n, true
	}
	return 0, false
}

// Open parses the header of the file at path and checks that the data area
// the header describes is the rest of the file.
func Open(path string) (*File, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		return nil, err
	}
	f, err := Parse(bufio.NewReaderSize(fh, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	f.FileSize = st.Size()
	if err := f.CheckDataArea(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return f, nil
}

// CheckDataArea verifies that the file holds exactly the data the tensor table
// describes: every tensor padded to the alignment, the last one included, and
// nothing after it. A short file is truncated; a long one has bytes no tensor
// owns, which for a pinned file means the header is not the one measured.
func (f *File) CheckDataArea() error {
	if f.FileSize == 0 {
		return nil
	}
	area := f.FileSize - f.DataOffset
	if area != f.DataSize {
		return fmt.Errorf("data area is %d bytes but the tensor table describes %d (padded to %d)",
			area, f.DataSize, f.Alignment)
	}
	return nil
}

// ErrFormat wraps every complaint about the file's contents, as opposed to an
// I/O error reading it.
var ErrFormat = errors.New("malformed gguf")

func formatErr(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrFormat, fmt.Sprintf(format, args...))
}

// Parse reads a header from r, which is left positioned at or before the start
// of the data area.
func Parse(r io.Reader) (*File, error) {
	rd := &reader{r: bufio.NewReader(r)}
	if br, ok := r.(*bufio.Reader); ok {
		rd.r = br
	}

	var magic [4]byte
	if err := rd.full(magic[:]); err != nil {
		return nil, err
	}
	if string(magic[:]) != Magic {
		return nil, formatErr("magic %q is not %q", magic[:], Magic)
	}
	f := &File{index: map[string]int{}, Alignment: DefaultAlignment}
	var err error
	if f.Version, err = rd.u32(); err != nil {
		return nil, err
	}
	switch {
	case f.Version&0xFFFF == 0:
		return nil, formatErr("version %#x: the file is big-endian", f.Version)
	case f.Version < 2 || f.Version > Version:
		return nil, formatErr("version %d; only 2 and 3 are read", f.Version)
	}
	nTensors, err := rd.i64()
	if err != nil {
		return nil, err
	}
	nKV, err := rd.i64()
	if err != nil {
		return nil, err
	}
	if nTensors < 0 || nKV < 0 {
		return nil, formatErr("negative count: %d tensors, %d metadata pairs", nTensors, nKV)
	}

	// Counts are not trusted for allocation: a corrupt header claiming 2^60
	// tensors fails on the first missing byte instead.
	f.KV = make([]KV, 0, min(nKV, 1024))
	for i := int64(0); i < nKV; i++ {
		key, err := rd.str(MaxString)
		if err != nil {
			return nil, fmt.Errorf("metadata pair %d: %w", i, err)
		}
		if key == "" {
			return nil, formatErr("metadata pair %d has an empty key", i)
		}
		if _, dup := f.index[key]; dup {
			return nil, formatErr("duplicate metadata key %q", key)
		}
		v, err := rd.value()
		if err != nil {
			return nil, fmt.Errorf("metadata %q: %w", key, err)
		}
		if key == KeyAlignment {
			f.alignOff = rd.n - 4 // checked to be a u32 below
		}
		f.index[key] = len(f.KV)
		f.KV = append(f.KV, KV{Key: key, Value: v})
	}
	f.KVEnd = rd.n

	if v, ok := f.Lookup(KeyAlignment); ok {
		a, isU32 := v.Data.(uint32)
		if v.Array || !isU32 {
			return nil, formatErr("%s must be u32", KeyAlignment)
		}
		if a == 0 || a&(a-1) != 0 {
			return nil, formatErr("%s %d is not a power of two", KeyAlignment, a)
		}
		f.Alignment = int64(a)
	}

	f.Tensors = make([]TensorInfo, 0, min(nTensors, 4096))
	seen := map[string]bool{}
	for i := int64(0); i < nTensors; i++ {
		t, err := rd.tensorInfo()
		if err != nil {
			return nil, fmt.Errorf("tensor %d: %w", i, err)
		}
		if seen[t.Name] {
			return nil, formatErr("duplicate tensor %q", t.Name)
		}
		seen[t.Name] = true
		f.Tensors = append(f.Tensors, t)
	}

	f.TableEnd = rd.n
	f.DataOffset = rd.n
	if nTensors > 0 {
		f.DataOffset = pad(rd.n, f.Alignment)
	}

	// Tensors are laid out back to back in table order, each padded to the
	// alignment. gguf.cpp refuses any other layout, and range-lock relies on
	// it: the byte range of a tensor is [offset, offset+size) and nothing else
	// lives there.
	var total int64
	for _, t := range f.Tensors {
		if t.Offset != uint64(total) {
			return nil, formatErr("tensor %q has offset %d, expected %d", t.Name, t.Offset, total)
		}
		p := pad(t.Size, f.Alignment)
		if p < t.Size || total > math.MaxInt64-p {
			return nil, formatErr("tensor %q overflows the data area", t.Name)
		}
		total += p
	}
	f.DataSize = total
	if f.DataOffset > math.MaxInt64-total {
		return nil, formatErr("data area overflows")
	}
	return f, nil
}

func pad(n, align int64) int64 {
	if align <= 1 {
		return n
	}
	return (n + align - 1) &^ (align - 1)
}

// reader counts what it consumes, so the data offset is known without seeking.
type reader struct {
	r   *bufio.Reader
	n   int64
	buf [8]byte
}

func (r *reader) full(p []byte) error {
	k, err := io.ReadFull(r.r, p)
	r.n += int64(k)
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return formatErr("truncated header at byte %d", r.n)
	}
	return err
}

func (r *reader) skip(n uint64) error {
	for n > 0 {
		chunk := int(min(n, 1<<30))
		k, err := r.r.Discard(chunk)
		r.n += int64(k)
		n -= uint64(k)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return formatErr("truncated header at byte %d", r.n)
			}
			return err
		}
	}
	return nil
}

func (r *reader) u8() (uint8, error) {
	err := r.full(r.buf[:1])
	return r.buf[0], err
}

func (r *reader) u16() (uint16, error) {
	err := r.full(r.buf[:2])
	return binary.LittleEndian.Uint16(r.buf[:2]), err
}

func (r *reader) u32() (uint32, error) {
	err := r.full(r.buf[:4])
	return binary.LittleEndian.Uint32(r.buf[:4]), err
}

func (r *reader) u64() (uint64, error) {
	err := r.full(r.buf[:8])
	return binary.LittleEndian.Uint64(r.buf[:8]), err
}

func (r *reader) i64() (int64, error) {
	n, err := r.u64()
	return int64(n), err
}

func (r *reader) str(limit uint64) (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > limit {
		return "", formatErr("string of %d bytes exceeds %d", n, limit)
	}
	b := make([]byte, n)
	if err := r.full(b); err != nil {
		return "", err
	}
	return string(b), nil
}

func (r *reader) value() (Value, error) {
	t, err := r.u32()
	if err != nil {
		return Value{}, err
	}
	v := Value{Type: ValueType(t)}
	if v.Type != ValArray {
		v.Data, err = r.scalar(v.Type)
		return v, err
	}
	et, err := r.u32()
	if err != nil {
		return Value{}, err
	}
	v.Array, v.Type = true, ValueType(et)
	if v.Len, err = r.u64(); err != nil {
		return Value{}, err
	}
	if v.Type == ValArray {
		return Value{}, formatErr("nested arrays are not allowed")
	}
	if v.Type > ValFloat64 {
		return Value{}, formatErr("unknown value type %d", et)
	}
	if v.Len > MaxHeldArray {
		v.Skipped = true
		return v, r.skipArray(v.Type, v.Len)
	}
	xs := make([]any, v.Len)
	for i := range xs {
		if xs[i], err = r.scalar(v.Type); err != nil {
			return Value{}, err
		}
	}
	v.Data = xs
	return v, nil
}

func (r *reader) skipArray(t ValueType, n uint64) error {
	if sz := t.size(); sz > 0 {
		if n > math.MaxUint64/uint64(sz) {
			return formatErr("array of %d elements overflows", n)
		}
		return r.skip(n * uint64(sz))
	}
	// Strings: each carries its own length and is skipped without being held.
	for range n {
		l, err := r.u64()
		if err != nil {
			return err
		}
		if l > MaxString {
			return formatErr("string of %d bytes exceeds %d", l, MaxString)
		}
		if err := r.skip(l); err != nil {
			return err
		}
	}
	return nil
}

func (r *reader) scalar(t ValueType) (any, error) {
	switch t {
	case ValUint8:
		return r.u8()
	case ValInt8:
		n, err := r.u8()
		return int8(n), err
	case ValUint16:
		return r.u16()
	case ValInt16:
		n, err := r.u16()
		return int16(n), err
	case ValUint32:
		return r.u32()
	case ValInt32:
		n, err := r.u32()
		return int32(n), err
	case ValFloat32:
		n, err := r.u32()
		return math.Float32frombits(n), err
	case ValBool:
		n, err := r.u8()
		return n != 0, err
	case ValString:
		return r.str(MaxString)
	case ValUint64:
		return r.u64()
	case ValInt64:
		return r.i64()
	case ValFloat64:
		n, err := r.u64()
		return math.Float64frombits(n), err
	}
	return nil, formatErr("unknown value type %d", uint32(t))
}

func (r *reader) tensorInfo() (TensorInfo, error) {
	var t TensorInfo
	name, err := r.str(MaxString)
	if err != nil {
		return t, err
	}
	// GGML_MAX_NAME counts the terminating NUL.
	if len(name) >= MaxName {
		return t, formatErr("tensor name of %d bytes; the limit is %d", len(name), MaxName-1)
	}
	t.Name = name
	nDims, err := r.u32()
	if err != nil {
		return t, err
	}
	if nDims > MaxDims {
		return t, formatErr("tensor %q has %d dimensions; the limit is %d", name, nDims, MaxDims)
	}
	t.Shape = make([]int64, nDims)
	elems := int64(1)
	for j := range t.Shape {
		if t.Shape[j], err = r.i64(); err != nil {
			return t, err
		}
		d := t.Shape[j]
		if d < 0 {
			return t, formatErr("tensor %q has negative dimension %d", name, d)
		}
		if d != 0 && elems > math.MaxInt64/d {
			return t, formatErr("tensor %q has too many elements", name)
		}
		elems *= d
	}
	typ, err := r.u32()
	if err != nil {
		return t, err
	}
	t.Type = Type(typ)
	if !t.Type.Known() {
		return t, formatErr("tensor %q has unknown or removed ggml type %d", name, typ)
	}
	ne0 := int64(1)
	if nDims > 0 {
		ne0 = t.Shape[0]
	}
	bs := t.Type.BlockSize()
	if ne0%bs != 0 {
		return t, formatErr("tensor %q: row of %d elements is not a multiple of the %s block size %d",
			name, ne0, t.Type, bs)
	}
	blocks := elems / bs
	if blocks > math.MaxInt64/t.Type.TypeSize() {
		return t, formatErr("tensor %q is too large", name)
	}
	t.Size = blocks * t.Type.TypeSize()
	if t.Offset, err = r.u64(); err != nil {
		return t, err
	}
	return t, nil
}
