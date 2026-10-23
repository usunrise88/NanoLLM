package gguf

import (
	"errors"
	"fmt"
	"slices"
	"sort"
)

// OPT-7: a tensor order of our own (docs/AUTOTUNE.md, "OPT-7").
//
// A published GGUF lists tensors in the converter's order: the globals, then
// every block with its tensors in alphabetical order. A layer's FFN matrices
// sit a few kilobytes of norms apart, and the small tensors the planner pins
// (norms and scales, 171 ranges for 2 MiB on E4B) are spread over the whole
// file. Relayout writes the same tensors, byte for byte, in the order the
// graph reads them, with the groups the planner places apart kept apart:
//
//	P  every G4 tensor: the pinned set is one range
//	G  token_embd (and a separate LM head): the other tensor the hot modes lock
//	L  per_layer_model_proj, then blk.0 … blk.N-1; inside a block the matrices
//	   in execution order, so a layer's streamed bytes are one range
//	E  per_layer_token_embd, the PLE table: read by row, never locked
//
// The order within the sections is deterministic. Nothing but the order, the
// offsets and the alignment changes: the metadata block is copied as it is.
//
// The format leaves little room. gguf.cpp, and Parse after it, require the
// tensor table to list tensors in the order of their data and each tensor to
// start where the previous one ends, padded to the single general.alignment of
// the file. So the layout cannot put a section start on a 2 MiB boundary
// without padding every tensor to 2 MiB (E4B: +1.1 GiB, and the pinned section
// 640 MiB wide); the alignment is the knob, and its default is the 4 KiB of a
// disk block and a page, the unit O_DIRECT and mlock work in.

// LayoutVersion is the version of the order Relayout writes. A change of the
// order bumps it, and a sidecar of another version is stale.
const LayoutVersion = 1

// The two layouts an Inventory (and a configuration) can name.
const (
	LayoutOriginal = "original"
	LayoutNano     = "nano"
)

// Alignments Relayout accepts. The default is the page and disk block size.
const (
	DefaultRelayoutAlignment = 4096
	MaxRelayoutAlignment     = 1 << 26
)

// Section is a part of the relaid data area.
type Section string

// The sections, in file order.
const (
	SectionPinned Section = "P"
	SectionGlobal Section = "G"
	SectionLayers Section = "L"
	SectionPLE    Section = "E"
)

var sectionOrder = []Section{SectionPinned, SectionGlobal, SectionLayers, SectionPLE}

func (s Section) index() int { return slices.Index(sectionOrder, s) }

// Description says what a section holds, for `models relayout`.
func (s Section) Description() string {
	switch s {
	case SectionPinned:
		return "pinned: norms and scales (G4)"
	case SectionGlobal:
		return "token embedding and LM head (G0)"
	case SectionLayers:
		return "layers: per_layer_model_proj, then blk.0 … blk.N-1 in execution order"
	case SectionPLE:
		return "per-layer embedding table (G1)"
	}
	return string(s)
}

// Execution order of a block's tensors, by the name without .weight, from
// src/models/gemma4.cpp of the pinned tag: the attention projections, the
// output projection, the FFN (gate and up read the same input, then down),
// then the per-layer-embedding gate and projection. A name that is not listed
// goes after the listed ones, in file order.
var (
	matrixOrder = []string{"attn_qkv", "attn_q", "attn_k", "attn_v", "attn_output",
		"ffn_gate", "ffn_up", "ffn_down", "inp_gate", "proj"}
	smallOrder = []string{"attn_norm", "attn_q_norm", "attn_k_norm", "attn_post_norm", "post_attention_norm",
		"ffn_norm", "ffn_post_norm", "post_ffw_norm", "post_norm", "layer_output_scale", "rope_freqs"}
	globalOrder = []string{"token_embd", "output"}
)

func rankIn(order []string, base string) int {
	if i := slices.Index(order, base); i >= 0 {
		return i
	}
	return len(order)
}

// Placed is one tensor in the relaid file.
type Placed struct {
	Name    string  `json:"name"`
	Section Section `json:"section"`
	// Layer is the block number in the name, -1 for a global tensor.
	Layer int `json:"layer"`
	Type  Type
	Shape []int64
	Bytes int64 `json:"bytes"`
	// From and Offset are absolute file offsets of the tensor's first byte in
	// the source and in the result.
	From   int64 `json:"from"`
	Offset int64 `json:"offset"`
}

// Extent is the byte range one section takes in the result, from its first
// tensor's start to its last tensor's end (padding between tensors included).
type Extent struct {
	Section Section `json:"section"`
	Offset  int64   `json:"offset"`
	Length  int64   `json:"length"`
	Tensors int     `json:"tensors"`
	// Bytes is the tensors' own size, without padding.
	Bytes int64 `json:"bytes"`
}

// Layout is a plan for the relaid file: where every tensor goes. It is
// computed without writing anything.
type Layout struct {
	Version   int   `json:"version"`
	Alignment int64 `json:"alignment"`
	// Tensors are in the order of the result's table and data.
	Tensors []Placed `json:"tensors"`
	// DataOffset is where the data area starts; FileSize the result's size.
	DataOffset int64    `json:"data_offset"`
	FileSize   int64    `json:"file_size"`
	Sections   []Extent `json:"sections"`
	// Approximate is set when the plan was made from an inventory, which does
	// not know the header's exact size: offsets can be off by one alignment.
	Approximate bool `json:"approximate,omitempty"`

	source struct {
		version  uint32
		headEnd  int64 // unpadded end of the source's header
		addAlign bool  // the result needs a general.alignment pair the source lacks
	}
}

// alignKVSize is the encoded size of a general.alignment pair: key, type, u32.
const alignKVSize = 8 + int64(len(KeyAlignment)) + 4 + 4

// CheckAlignment says whether a is an alignment Relayout writes.
func CheckAlignment(a int64) error {
	switch {
	case a < DefaultAlignment || a > MaxRelayoutAlignment:
		return fmt.Errorf("alignment %d is outside %d…%d", a, DefaultAlignment, MaxRelayoutAlignment)
	case a&(a-1) != 0:
		return fmt.Errorf("alignment %d is not a power of two", a)
	}
	return nil
}

// PlanFile plans the layout of an opened file. The offsets are exact.
func PlanFile(f *File, alignment int64) (*Layout, error) {
	inv, err := BuildInventory(f)
	if err != nil {
		return nil, fmt.Errorf("relayout: %w", err)
	}
	_, has := f.AlignmentOffset()
	return planLayout(inv, f.Version, f.TableEnd, has, alignment, false)
}

// PlanInventory plans the layout of a file known only by its inventory (the
// catalog's, when the file is not at hand). The header's size is taken from
// the data offset, so the offsets are approximate.
func PlanInventory(inv Inventory, alignment int64) (*Layout, error) {
	return planLayout(inv, Version, inv.DataOffset, false, alignment, true)
}

func planLayout(inv Inventory, version uint32, headEnd int64, hasAlign bool, alignment int64, approx bool) (*Layout, error) {
	if alignment == 0 {
		alignment = DefaultRelayoutAlignment
	}
	if err := CheckAlignment(alignment); err != nil {
		return nil, fmt.Errorf("relayout: %w", err)
	}
	if inv.HParams.Arch != ArchGemma4 {
		return nil, fmt.Errorf("relayout: no tensor order for architecture %q (layout version %d knows %s)",
			inv.HParams.Arch, LayoutVersion, ArchGemma4)
	}
	if len(inv.Tensors) == 0 {
		return nil, errors.New("relayout: the file has no tensors")
	}
	type item struct {
		t       Tensor
		idx     int
		section Section
		layer   int
		rank    int
	}
	items := make([]item, 0, len(inv.Tensors))
	var unknown []error
	for i, t := range inv.Tensors {
		c, err := Classify(inv.HParams.Arch, t.Name)
		if err != nil {
			unknown = append(unknown, err)
			continue
		}
		layer, base, _ := splitName(t.Name)
		it := item{t: t, idx: i, layer: layer}
		switch c.Group {
		case G4:
			it.section, it.rank = SectionPinned, rankIn(smallOrder, base)
		case G0:
			it.section, it.rank = SectionGlobal, rankIn(globalOrder, base)
		case G1:
			it.section = SectionPLE
		default: // G2 and G3, the matrices
			it.section, it.rank = SectionLayers, rankIn(matrixOrder, base)
		}
		items = append(items, it)
	}
	if len(unknown) > 0 {
		return nil, fmt.Errorf("relayout: %w", errors.Join(unknown...))
	}
	sort.SliceStable(items, func(a, b int) bool {
		x, y := items[a], items[b]
		if x.section != y.section {
			return x.section.index() < y.section.index()
		}
		if x.layer != y.layer {
			return x.layer < y.layer
		}
		if x.rank != y.rank {
			return x.rank < y.rank
		}
		return x.idx < y.idx
	})

	l := &Layout{Version: LayoutVersion, Alignment: alignment, Approximate: approx}
	l.source.version, l.source.headEnd = version, headEnd
	l.source.addAlign = !hasAlign && alignment != DefaultAlignment
	end := headEnd
	if l.source.addAlign {
		end += alignKVSize
	}
	l.DataOffset = pad(end, alignment)
	off := l.DataOffset
	l.Tensors = make([]Placed, len(items))
	for i, it := range items {
		typ, err := ParseType(it.t.Type)
		if err != nil {
			return nil, fmt.Errorf("relayout: tensor %q: %w", it.t.Name, err)
		}
		l.Tensors[i] = Placed{
			Name: it.t.Name, Section: it.section, Layer: it.layer, Type: typ, Shape: it.t.Shape,
			Bytes: it.t.Bytes, From: it.t.Offset, Offset: off,
		}
		off += pad(it.t.Bytes, alignment)
	}
	l.FileSize = off
	for _, s := range sectionOrder {
		var e Extent
		for _, p := range l.Tensors {
			if p.Section != s {
				continue
			}
			if e.Tensors == 0 {
				e = Extent{Section: s, Offset: p.Offset}
			}
			e.Length = p.Offset + p.Bytes - e.Offset
			e.Tensors++
			e.Bytes += p.Bytes
		}
		if e.Tensors > 0 {
			l.Sections = append(l.Sections, e)
		}
	}
	return l, nil
}

// Inventory is the inventory of the file this layout writes: the source's
// tensors in the new order with the new offsets. SHA256 and Role are the
// caller's to fill; the planner needs neither.
func (l *Layout) Inventory(src Inventory) Inventory {
	inv := src
	inv.SHA256 = ""
	inv.Layout = LayoutNano
	inv.Alignment = l.Alignment
	inv.DataOffset = l.DataOffset
	inv.DataSize = l.FileSize - l.DataOffset
	inv.Size = l.FileSize
	by := make(map[string]Tensor, len(src.Tensors))
	for _, t := range src.Tensors {
		by[t.Name] = t
	}
	inv.Tensors = make([]Tensor, len(l.Tensors))
	for i, p := range l.Tensors {
		t := by[p.Name]
		t.Offset = p.Offset
		inv.Tensors[i] = t
	}
	return inv
}

// SameTensors says whether two inventories describe the same tensors: names,
// groups, types, shapes and sizes, in any order and at any offsets. The nano
// file of a catalog entry must pass it against the entry's inventory.
func (inv Inventory) SameTensors(o Inventory) error {
	if len(inv.Tensors) != len(o.Tensors) {
		return fmt.Errorf("%d tensors against %d", len(inv.Tensors), len(o.Tensors))
	}
	by := make(map[string]Tensor, len(o.Tensors))
	for _, t := range o.Tensors {
		by[t.Name] = t
	}
	for _, t := range inv.Tensors {
		u, ok := by[t.Name]
		if !ok {
			return fmt.Errorf("tensor %q is missing", t.Name)
		}
		if u.Group != t.Group || u.Subgroup != t.Subgroup || u.Layer != t.Layer || u.Type != t.Type ||
			u.Bytes != t.Bytes || !slices.Equal(u.Shape, t.Shape) {
			return fmt.Errorf("tensor %q differs: %s %v %d bytes against %s %v %d bytes",
				t.Name, t.Type, t.Shape, t.Bytes, u.Type, u.Shape, u.Bytes)
		}
	}
	return nil
}

// ErrExists is returned by Relayout when the result is there and Force is off.
var ErrExists = errors.New("the relaid file exists; replace it with force")
