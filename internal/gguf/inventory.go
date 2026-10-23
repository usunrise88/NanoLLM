package gguf

import (
	"errors"
	"fmt"
)

// Tensor is one inventoried tensor: where it sits in the file and which
// placement group it belongs to.
type Tensor struct {
	Name     string `json:"name" yaml:"name"`
	Group    Group  `json:"group" yaml:"group"`
	Subgroup string `json:"subgroup,omitempty" yaml:"subgroup,omitempty"`
	// Layer is the repeating layer, -1 for none (see Class).
	Layer  int     `json:"layer" yaml:"layer"`
	Input  bool    `json:"input,omitempty" yaml:"input,omitempty"`
	Output bool    `json:"output,omitempty" yaml:"output,omitempty"`
	Type   string  `json:"type" yaml:"type"`
	Shape  []int64 `json:"shape" yaml:"shape"`
	// Bytes is the tensor's size without alignment padding.
	Bytes int64 `json:"bytes" yaml:"bytes"`
	// Offset is absolute in the file, so a range-lock can pin
	// [Offset, Offset+Bytes) without knowing where the data area starts.
	Offset int64 `json:"offset" yaml:"offset"`
}

// Inventory is the measured header of one GGUF file. It is what the planner
// works from, so it carries the file's identity: an inventory is only valid
// for the bytes whose SHA256 it names.
type Inventory struct {
	Role   string `json:"role"`
	SHA256 string `json:"sha256"`
	// Layout names the tensor order of the file: empty for the file as
	// published, LayoutNano for one written by Relayout. The planner merges
	// the stream ranges of a nano file, which lie back to back by design.
	Layout     string  `json:"layout,omitempty"`
	Size       int64   `json:"size"`
	Version    uint32  `json:"version"`
	Alignment  int64   `json:"alignment"`
	DataOffset int64   `json:"data_offset"`
	DataSize   int64   `json:"data_size"`
	HParams    HParams `json:"hparams"`
	// Groups is the sum of Bytes per group, for readers of the stored file;
	// Check recomputes it.
	Groups  map[Group]int64 `json:"groups"`
	Tensors []Tensor        `json:"tensors"`
}

// Sum is the bytes of one group.
func (inv Inventory) Sum(g Group) int64 {
	var n int64
	for _, t := range inv.Tensors {
		if t.Group == g {
			n += t.Bytes
		}
	}
	return n
}

// SumSub is the bytes of one subgroup of a group.
func (inv Inventory) SumSub(g Group, sub string) int64 {
	var n int64
	for _, t := range inv.Tensors {
		if t.Group == g && t.Subgroup == sub {
			n += t.Bytes
		}
	}
	return n
}

// Total is the bytes of every tensor, unpadded.
func (inv Inventory) Total() int64 {
	var n int64
	for _, t := range inv.Tensors {
		n += t.Bytes
	}
	return n
}

// Find returns the tensor named name.
func (inv Inventory) Find(name string) (Tensor, bool) {
	for _, t := range inv.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return Tensor{}, false
}

// BuildInventory classifies every tensor of f. Tensors the group table does
// not know are kept with an empty group and reported in the returned error
// (one per tensor, joined): `models inspect` shows them as a warning, the
// catalog build refuses them. Role and SHA256 are the caller's to fill.
func BuildInventory(f *File) (Inventory, error) {
	h, err := f.HParams()
	if err != nil {
		return Inventory{}, err
	}
	inv := Inventory{
		Size:       f.FileSize,
		Version:    f.Version,
		Alignment:  f.Alignment,
		DataOffset: f.DataOffset,
		DataSize:   f.DataSize,
		HParams:    h,
		Tensors:    make([]Tensor, 0, len(f.Tensors)),
	}
	var unknown []error
	for _, ti := range f.Tensors {
		c, err := Classify(h.Arch, ti.Name)
		if err != nil {
			unknown = append(unknown, err)
			c = Class{Layer: -1}
		}
		inv.Tensors = append(inv.Tensors, Tensor{
			Name: ti.Name, Group: c.Group, Subgroup: c.Subgroup, Layer: c.Layer,
			Input: c.Input, Output: c.Output,
			Type: ti.Type.String(), Shape: ti.Shape,
			Bytes: ti.Size, Offset: f.FileOffset(ti),
		})
	}
	inv.Groups = inv.groupSums()
	return inv, errors.Join(unknown...)
}

func (inv Inventory) groupSums() map[Group]int64 {
	m := map[Group]int64{}
	for _, t := range inv.Tensors {
		if t.Group != "" {
			m[t.Group] += t.Bytes
		}
	}
	return m
}

// Check verifies a stored inventory against itself and the group table: every
// tensor classified as the table classifies it today, the tensors laid out
// back to back with padding, the padded sum equal to the data area, and the
// recorded group sums right. A stored inventory that fails it was edited by
// hand or written by an older table, and plans made from it would be wrong.
func (inv Inventory) Check() error {
	if inv.Alignment <= 0 || inv.Alignment&(inv.Alignment-1) != 0 {
		return fmt.Errorf("alignment %d is not a power of two", inv.Alignment)
	}
	next := inv.DataOffset
	for _, t := range inv.Tensors {
		c, err := Classify(inv.HParams.Arch, t.Name)
		if err != nil {
			return err
		}
		if c.Group != t.Group || c.Subgroup != t.Subgroup || c.Layer != t.Layer ||
			c.Input != t.Input || c.Output != t.Output {
			return fmt.Errorf("tensor %q is recorded as %s/%s layer %d, the table says %s/%s layer %d",
				t.Name, t.Group, t.Subgroup, t.Layer, c.Group, c.Subgroup, c.Layer)
		}
		typ, err := ParseType(t.Type)
		if err != nil {
			return fmt.Errorf("tensor %q: %w", t.Name, err)
		}
		elems := int64(1)
		for _, d := range t.Shape {
			elems *= d
		}
		if bs := typ.BlockSize(); elems%bs != 0 || elems/bs*typ.TypeSize() != t.Bytes {
			return fmt.Errorf("tensor %q: %d bytes do not match %s %v", t.Name, t.Bytes, t.Type, t.Shape)
		}
		if t.Offset != next {
			return fmt.Errorf("tensor %q at offset %d, expected %d", t.Name, t.Offset, next)
		}
		next += pad(t.Bytes, inv.Alignment)
	}
	if got := next - inv.DataOffset; got != inv.DataSize {
		return fmt.Errorf("padded tensor sizes sum to %d, the data area is %d", got, inv.DataSize)
	}
	if inv.Size != 0 && inv.DataOffset+inv.DataSize != inv.Size {
		return fmt.Errorf("data area ends at %d, the file is %d bytes", inv.DataOffset+inv.DataSize, inv.Size)
	}
	sums := inv.groupSums()
	if len(sums) != len(inv.Groups) {
		return fmt.Errorf("recorded group sums %v, computed %v", inv.Groups, sums)
	}
	for g, n := range sums {
		if inv.Groups[g] != n {
			return fmt.Errorf("group %s: recorded %d bytes, computed %d", g, inv.Groups[g], n)
		}
	}
	return nil
}
