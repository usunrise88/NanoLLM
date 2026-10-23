package plan

import (
	"slices"
	"testing"

	"github.com/usunrise88/nanollm/internal/gguf"
)

// nanoOf is E4B as `models relayout` would write it: the catalog's inventory
// laid out by the layout policy. The offsets are those of the planned file; the
// header's size is the catalog's data offset, so they are exact to within one
// alignment, which a plan does not care about.
func nanoOf(t *testing.T, model Model, alignment int64) Model {
	t.Helper()
	l, err := gguf.PlanInventory(model.Target, alignment)
	if err != nil {
		t.Fatal(err)
	}
	inv := l.Inventory(model.Target)
	if err := inv.Check(); err != nil {
		t.Fatalf("the planned inventory is invalid: %v", err)
	}
	model.Target = inv
	return model
}

func sumRanges(rs []Range) (n int64) {
	for _, r := range rs {
		n += r.Length
	}
	return n
}

func perLayer(rs []StreamRange) (counts map[int]int, bytes int64) {
	counts = map[int]int{}
	for _, r := range rs {
		counts[r.Layer]++
		bytes += r.Length
	}
	return counts, bytes
}

// OPT-7: the same tensors in the order of the graph make fewer ranges. In the
// file as published a layer's FFN is three stream ranges and the pinned set is
// 171 lock ranges; relaid, a layer is one range and the pinned set is one.
func TestNanoLayoutMakesFewerRanges(t *testing.T) {
	orig := e4b(t)
	nano := nanoOf(t, orig, gguf.DefaultRelayoutAlignment)
	if err := orig.Target.SameTensors(nano.Target); err != nil {
		t.Fatalf("not the same tensors: %v", err)
	}
	layers := orig.Target.HParams.BlockCount

	type c struct {
		name     string
		mode     Mode
		preset   string
		noRepack bool
	}
	for _, x := range []c{
		{"M7", RAMPLEDisk, "ram-4g", false},
		{"M7 without repack", RAMPLEDisk, "ram-4g", true},
		{"M8 ram-2g", RAMHybrid, "ram-2g", false},
		{"M8 ram-1.5g", RAMHybrid, "ram-1.5g", false},
		{"M9", MMap, "ram-1g", false},
	} {
		t.Run(x.name, func(t *testing.T) {
			req := func(m Model) Request {
				return Request{Model: m, Preset: preset(t, x.preset), Mode: x.mode, NoRepack: x.noRepack, NoMTP: true}
			}
			a, b := mustPlan(t, req(orig)), mustPlan(t, req(nano))
			if !a.OK() || !b.OK() {
				t.Fatalf("refused: %q, %q", a.Refused, b.Refused)
			}
			if a.Layout != "" || b.Layout != gguf.LayoutNano {
				t.Errorf("layout %q and %q", a.Layout, b.Layout)
			}

			// The same decisions: mode, groups, budget, context, overrides.
			if a.Mode != b.Mode || a.NCtx != b.NCtx || a.Slots != b.Slots || a.Repack != b.Repack ||
				a.Memory.RAM.Used != b.Memory.RAM.Used || a.Memory.FileBacked != b.Memory.FileBacked {
				t.Errorf("plans differ: %s %d×%d ram %d file %d, then %s %d×%d ram %d file %d",
					a.Mode, a.NCtx, a.Slots, a.Memory.RAM.Used, a.Memory.FileBacked,
					b.Mode, b.NCtx, b.Slots, b.Memory.RAM.Used, b.Memory.FileBacked)
			}
			for g, v := range a.Groups {
				if b.Groups[g] != v {
					t.Errorf("%s: %+v, then %+v", g, v, b.Groups[g])
				}
			}
			if !slices.Equal(a.Overrides, b.Overrides) {
				t.Errorf("overrides differ: %v, %v", a.Overrides, b.Overrides)
			}

			// The same bytes, in fewer ranges.
			la, lb := sumRanges(a.LockRanges), sumRanges(b.LockRanges)
			if len(b.LockRanges) > len(a.LockRanges) || len(b.LockRanges) != 1 {
				t.Errorf("lock ranges: %d, then %d (want 1)", len(a.LockRanges), len(b.LockRanges))
			}
			// The nano file pads each tensor to the alignment, and a range
			// covers the padding between tensors: no more than one alignment
			// per pinned tensor over what the plan accounts for.
			if lb < la || lb-la > int64(len(nano.Target.Tensors))*gguf.DefaultRelayoutAlignment {
				t.Errorf("locked bytes: %d, then %d", la, lb)
			}
			ca, sa := perLayer(a.StreamRanges)
			cb, sb := perLayer(b.StreamRanges)
			if sa != sb || len(ca) != len(cb) {
				t.Errorf("stream: %d bytes in %d layers, then %d in %d", sa, len(ca), sb, len(cb))
			}
			for il, n := range cb {
				if n != 1 {
					t.Errorf("layer %d has %d stream ranges", il, n)
				}
			}
			if len(a.StreamRanges) > 0 && (len(b.StreamRanges) != layers || len(b.StreamRanges) >= len(a.StreamRanges)) {
				t.Errorf("stream ranges: %d, then %d (want %d)", len(a.StreamRanges), len(b.StreamRanges), layers)
			}
			if len(b.RandomRanges) != len(a.RandomRanges) || sumRanges(b.RandomRanges) != sumRanges(a.RandomRanges) {
				t.Errorf("random ranges: %v, then %v", a.RandomRanges, b.RandomRanges)
			}
			t.Logf("lock %d → %d ranges, stream %d → %d", len(a.LockRanges), len(b.LockRanges), len(a.StreamRanges), len(b.StreamRanges))
		})
	}
}

// Every range of the nano plan lies where the tensor is in the relaid file,
// and covers only tensors of the kind the plan names.
func TestNanoRangesAreTheTensorsOfTheNanoFile(t *testing.T) {
	nano := nanoOf(t, e4b(t), gguf.DefaultRelayoutAlignment)
	p := mustPlan(t, Request{Model: nano, Preset: preset(t, "ram-1g"), Mode: MMap, NoMTP: true})
	if !p.OK() {
		t.Fatal(p.Refused)
	}
	inLayer := map[int][2]int64{}
	for _, tn := range nano.Target.Tensors {
		if tn.Layer < 0 || tn.Group == gguf.G1 || tn.Group == gguf.G4 {
			continue
		}
		e, ok := inLayer[tn.Layer]
		if !ok {
			e = [2]int64{tn.Offset, tn.Offset + tn.Bytes}
		}
		inLayer[tn.Layer] = [2]int64{min(e[0], tn.Offset), max(e[1], tn.Offset+tn.Bytes)}
	}
	for _, r := range p.StreamRanges {
		e := inLayer[r.Layer]
		if r.Offset != e[0] || r.Offset+r.Length != e[1] {
			t.Errorf("layer %d: range [%d, %d), tensors [%d, %d)", r.Layer, r.Offset, r.Offset+r.Length, e[0], e[1])
		}
	}
	// The pinned set is the G4 tensors, in the section that holds them.
	var g4lo, g4hi int64 = 1 << 62, 0
	for _, tn := range nano.Target.Tensors {
		if tn.Group == gguf.G4 {
			g4lo, g4hi = min(g4lo, tn.Offset), max(g4hi, tn.Offset+tn.Bytes)
		}
	}
	if len(p.LockRanges) != 1 || p.LockRanges[0].Offset != g4lo || p.LockRanges[0].Offset+p.LockRanges[0].Length != g4hi {
		t.Errorf("lock ranges %+v, G4 spans [%d, %d)", p.LockRanges, g4lo, g4hi)
	}
}

// With a coarse alignment every norm is padded to it: the ranges stay apart,
// so a lock never pins the zeros between tensors.
func TestCoarseAlignmentDoesNotLockPadding(t *testing.T) {
	orig := e4b(t)
	nano := nanoOf(t, orig, 2<<20)
	a := mustPlan(t, Request{Model: orig, Preset: preset(t, "ram-4g"), Mode: RAMPLEDisk, NoMTP: true})
	b := mustPlan(t, Request{Model: nano, Preset: preset(t, "ram-4g"), Mode: RAMPLEDisk, NoMTP: true})
	// Exactly the pinned tensors' bytes (the published file's ranges also span
	// the 32-byte padding between neighbours).
	if got, want := sumRanges(b.LockRanges), orig.Target.Groups[gguf.G4]; got != want {
		t.Errorf("locked %d bytes, the G4 tensors are %d (the published file locks %d)", got, want, sumRanges(a.LockRanges))
	}
	if len(b.LockRanges) < 100 {
		t.Errorf("%d lock ranges: tensors 2 MiB apart were joined", len(b.LockRanges))
	}
}

// The published file's plans are what they were: the layout is a property of
// the inventory, and the stream ranges of a file that has none stay one per
// tensor (the golden plans pin the rest).
func TestOriginalLayoutKeepsItsRanges(t *testing.T) {
	p := mustPlan(t, Request{Model: e4b(t), Preset: preset(t, "ram-2g"), Mode: RAMHybrid, NoMTP: true})
	counts, _ := perLayer(p.StreamRanges)
	if len(p.StreamRanges) != 3*len(counts) || p.Layout != "" {
		t.Errorf("%d stream ranges in %d layers, layout %q", len(p.StreamRanges), len(counts), p.Layout)
	}
}
