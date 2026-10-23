package gguf

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// writeModel writes w to a file of its own and returns the path.
func writeModel(t *testing.T, name string, w *Writer) string {
	t.Helper()
	b, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func fileSHA(t *testing.T, p string) string {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// tensorData reads the bytes of one tensor of an opened file.
func tensorData(t *testing.T, p string, f *File, name string) []byte {
	t.Helper()
	ti, ok := f.Tensor(name)
	if !ok {
		t.Fatalf("%s has no tensor %s", p, name)
	}
	fh, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	b := make([]byte, ti.Size)
	if _, err := fh.ReadAt(b, f.FileOffset(ti)); err != nil {
		t.Fatal(err)
	}
	return b
}

func relayout(t *testing.T, src string, o RelayoutOptions) (string, Sidecar) {
	t.Helper()
	dst := NanoPath(src)
	sc, err := Relayout(context.Background(), src, dst, o)
	if err != nil {
		t.Fatal(err)
	}
	return dst, sc
}

func TestRelayoutRoundTrip(t *testing.T) {
	src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
	dst, sc := relayout(t, src, RelayoutOptions{Alignment: 4096})

	a, err := Open(src)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Open(dst) // our own reader takes it: table in data order, padding right, no stray bytes
	if err != nil {
		t.Fatal(err)
	}
	if b.Alignment != 4096 || b.DataOffset%4096 != 0 || len(b.Tensors) != len(a.Tensors) {
		t.Fatalf("alignment %d, data at %d, %d tensors against %d", b.Alignment, b.DataOffset, len(b.Tensors), len(a.Tensors))
	}
	for _, ti := range b.Tensors {
		if b.FileOffset(ti)%4096 != 0 {
			t.Errorf("%s starts at %d, not on the alignment", ti.Name, b.FileOffset(ti))
		}
		old, ok := a.Tensor(ti.Name)
		if !ok {
			t.Errorf("%s is new", ti.Name)
			continue
		}
		if old.Type != ti.Type || !slices.Equal(old.Shape, ti.Shape) || old.Size != ti.Size {
			t.Errorf("%s: %s %v %d bytes became %s %v %d bytes", ti.Name, old.Type, old.Shape, old.Size, ti.Type, ti.Shape, ti.Size)
		}
		if !bytes.Equal(tensorData(t, src, a, ti.Name), tensorData(t, dst, b, ti.Name)) {
			t.Errorf("%s: bytes differ", ti.Name)
		}
	}

	// The metadata block is copied as it is; general.alignment is the one pair added.
	rawA, _ := os.ReadFile(src)
	rawB, _ := os.ReadFile(dst)
	if !bytes.Equal(rawA[fixedStart:a.KVEnd], rawB[fixedStart:a.KVEnd]) {
		t.Error("the metadata block changed")
	}
	if len(b.KV) != len(a.KV)+1 || b.KV[len(b.KV)-1].Key != KeyAlignment {
		t.Errorf("%d pairs against %d, last %q", len(b.KV), len(a.KV), b.KV[len(b.KV)-1].Key)
	}
	for i, kv := range a.KV {
		if b.KV[i].Key != kv.Key || fmt.Sprint(b.KV[i].Value.Data) != fmt.Sprint(kv.Value.Data) {
			t.Errorf("pair %d: %s=%v became %s=%v", i, kv.Key, kv.Value.Data, b.KV[i].Key, b.KV[i].Value.Data)
		}
	}
	if v, _ := b.Lookup(KeyAlignment); v.Data != uint32(4096) {
		t.Errorf("general.alignment = %v", v.Data)
	}

	// Padding is zero: everything that is not a tensor or the header.
	covered := make([]bool, len(rawB))
	for i := int64(0); i < b.DataOffset; i++ {
		covered[i] = true
	}
	for _, ti := range b.Tensors {
		for i := range ti.Size {
			covered[b.FileOffset(ti)+i] = true
		}
	}
	for i, c := range covered {
		if !c && rawB[i] != 0 {
			t.Fatalf("byte %d is padding and not zero", i)
		}
	}
	if int64(len(rawB)) != sc.Result.Size || sc.Result.SHA256 != fileSHA(t, dst) || sc.Source.SHA256 != fileSHA(t, src) {
		t.Errorf("sidecar %+v", sc)
	}

	rep, err := VerifyRelayout(context.Background(), src, dst, nil)
	if err != nil {
		t.Fatal(err)
	}
	var total int64
	for _, ti := range a.Tensors {
		total += ti.Size
	}
	if rep.Tensors != len(a.Tensors) || rep.Bytes != total || rep.FileSize != sc.Result.Size || rep.SHA256 != sc.Result.SHA256 {
		t.Errorf("verified %+v, want %d tensors, %d bytes", rep, len(a.Tensors), total)
	}
}

func TestRelayoutDeterministic(t *testing.T) {
	src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
	first, sc1 := relayout(t, src, RelayoutOptions{Alignment: 4096})
	b1, _ := os.ReadFile(first)
	if err := os.Remove(first); err != nil {
		t.Fatal(err)
	}
	second, sc2 := relayout(t, src, RelayoutOptions{Alignment: 4096})
	b2, _ := os.ReadFile(second)
	if !bytes.Equal(b1, b2) || sc1.Result.SHA256 != sc2.Result.SHA256 {
		t.Error("the same source and options gave different bytes")
	}
	// The default alignment is the page size.
	d, _ := relayout(t, writeModel(t, "tiny.gguf", TinyGemma4().FillPattern()), RelayoutOptions{})
	if f, err := Open(d); err != nil || f.Alignment != DefaultRelayoutAlignment {
		t.Errorf("default alignment: %v, %v", f, err)
	}
}

func TestRelayoutIdempotent(t *testing.T) {
	for name, w := range map[string]*Writer{
		"alignment absent":  TinyGemma4().FillPattern(),
		"alignment present": TinyGemma4().Set(KeyAlignment, uint32(64)).FillPattern(),
	} {
		t.Run(name, func(t *testing.T) {
			src := writeModel(t, "tiny.gguf", w)
			once, _ := relayout(t, src, RelayoutOptions{Alignment: 4096})
			b1, _ := os.ReadFile(once)
			twice := filepath.Join(filepath.Dir(once), "again.gguf")
			sc, err := Relayout(context.Background(), once, twice, RelayoutOptions{Alignment: 4096})
			if err != nil {
				t.Fatal(err)
			}
			b2, _ := os.ReadFile(twice)
			if !bytes.Equal(b1, b2) {
				t.Error("relaying out a nano file changed it")
			}
			if sc.Source.SHA256 != fileSHA(t, once) {
				t.Error("the second sidecar does not name the first result as its source")
			}
		})
	}
}

func TestRelayoutAlignmentKey(t *testing.T) {
	t.Run("added when absent", func(t *testing.T) {
		src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
		a, _ := Open(src)
		dst, _ := relayout(t, src, RelayoutOptions{Alignment: 1 << 16})
		b, err := Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		if _, had := a.Lookup(KeyAlignment); had || len(b.KV) != len(a.KV)+1 || b.Alignment != 1<<16 {
			t.Errorf("had %v, %d pairs against %d, alignment %d", had, len(b.KV), len(a.KV), b.Alignment)
		}
	})
	t.Run("replaced when present", func(t *testing.T) {
		src := writeModel(t, "tiny.gguf", TinyGemma4().Set(KeyAlignment, uint32(64)).FillPattern())
		a, _ := Open(src)
		dst, _ := relayout(t, src, RelayoutOptions{Alignment: 8192})
		b, err := Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		if a.Alignment != 64 || b.Alignment != 8192 || len(b.KV) != len(a.KV) {
			t.Errorf("alignment %d → %d, %d pairs against %d", a.Alignment, b.Alignment, len(b.KV), len(a.KV))
		}
		ra, _ := os.ReadFile(src)
		rb, _ := os.ReadFile(dst)
		off, _ := a.AlignmentOffset()
		if !bytes.Equal(ra[fixedStart:off], rb[fixedStart:off]) || !bytes.Equal(ra[off+4:a.KVEnd], rb[off+4:a.KVEnd]) {
			t.Error("more of the metadata changed than the alignment value")
		}
	})
	t.Run("not added at the default", func(t *testing.T) {
		src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
		a, _ := Open(src)
		dst, _ := relayout(t, src, RelayoutOptions{Alignment: DefaultAlignment})
		b, err := Open(dst)
		if err != nil || len(b.KV) != len(a.KV) || b.Alignment != DefaultAlignment {
			t.Errorf("%d pairs against %d, alignment %d, %v", len(b.KV), len(a.KV), b.Alignment, err)
		}
	})
}

// layerOf is the block number of a tensor name, -1 for none.
func layerOf(name string) int {
	l, _, _ := splitName(name)
	return l
}

func TestLayoutOrder(t *testing.T) {
	src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
	f, err := Open(src)
	if err != nil {
		t.Fatal(err)
	}
	l, err := PlanFile(f, 4096)
	if err != nil {
		t.Fatal(err)
	}
	inv, _ := BuildInventory(f)
	group := map[string]Group{}
	for _, tn := range inv.Tensors {
		group[tn.Name] = tn.Group
	}

	// Sections come in the order P, G, L, E, and G4 is all of P.
	var seen []Section
	for _, p := range l.Tensors {
		if len(seen) == 0 || seen[len(seen)-1] != p.Section {
			seen = append(seen, p.Section)
		}
		if (group[p.Name] == G4) != (p.Section == SectionPinned) {
			t.Errorf("%s is %s and lies in section %s", p.Name, group[p.Name], p.Section)
		}
	}
	if !slices.Equal(seen, sectionOrder) {
		t.Fatalf("sections in the order %v", seen)
	}

	// L: per_layer_model_proj, then blocks in order, each block's matrices in
	// the order of the graph.
	var layers []string
	for _, p := range l.Tensors {
		if p.Section == SectionLayers {
			layers = append(layers, p.Name)
		}
	}
	if layers[0] != "per_layer_model_proj.weight" {
		t.Errorf("L starts with %s", layers[0])
	}
	want := []string{"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down", "inp_gate", "proj"}
	for il := range 6 {
		var got []string
		for _, n := range layers {
			if layerOf(n) == il {
				_, base, _ := splitName(n)
				got = append(got, base)
			}
		}
		var exp []string
		for _, b := range want {
			if _, ok := f.Tensor(fmt.Sprintf("blk.%d.%s.weight", il, b)); ok {
				exp = append(exp, b)
			}
		}
		if !slices.Equal(got, exp) {
			t.Errorf("block %d: %v, want %v", il, got, exp)
		}
	}
	if !sortedBy(layers, layerOf) {
		t.Errorf("blocks are not in order: %v", layers)
	}

	// A layer's matrices are one range: each starts where the last ends.
	byLayer := map[int][]Placed{}
	for _, p := range l.Tensors {
		if p.Section == SectionLayers {
			byLayer[p.Layer] = append(byLayer[p.Layer], p)
		}
	}
	for il, ps := range byLayer {
		for i := 1; i < len(ps); i++ {
			if want := ps[i-1].Offset + pad(ps[i-1].Bytes, 4096); ps[i].Offset != want {
				t.Errorf("layer %d: %s at %d, the end of %s is %d", il, ps[i].Name, ps[i].Offset, ps[i-1].Name, want)
			}
		}
	}
	// and so is the pinned set, and layer -1 (per_layer_model_proj) joins block 0.
	for i := 1; i < len(l.Tensors); i++ {
		a, b := l.Tensors[i-1], l.Tensors[i]
		if a.Section == b.Section && b.Offset != a.Offset+pad(a.Bytes, 4096) {
			t.Errorf("%s does not follow %s", b.Name, a.Name)
		}
	}
	if byLayer[-1][0].Offset+pad(byLayer[-1][0].Bytes, 4096) != byLayer[0][0].Offset {
		t.Error("per_layer_model_proj does not join block 0")
	}
}

func sortedBy(names []string, key func(string) int) bool {
	last := -2
	for _, n := range names {
		k := key(n)
		if k < last {
			return false
		}
		last = k
	}
	return true
}

// The plan made from the header, the plan made from its inventory and the
// header of what was written agree.
func TestPlannedInventoryIsWhatIsWritten(t *testing.T) {
	src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
	f, _ := Open(src)
	inv, _ := BuildInventory(f)
	l, err := PlanFile(f, 4096)
	if err != nil {
		t.Fatal(err)
	}
	planned := l.Inventory(inv)
	if err := planned.Check(); err != nil {
		t.Fatalf("the planned inventory is not a valid one: %v", err)
	}
	dst, _ := relayout(t, src, RelayoutOptions{Alignment: 4096})
	g, _ := Open(dst)
	got, err := BuildInventory(g)
	if err != nil {
		t.Fatal(err)
	}
	got.Layout = LayoutNano
	if got.Size != planned.Size || got.DataOffset != planned.DataOffset || got.Alignment != planned.Alignment ||
		got.DataSize != planned.DataSize || len(got.Tensors) != len(planned.Tensors) {
		t.Fatalf("written %d/%d/%d, planned %d/%d/%d", got.Size, got.DataOffset, got.DataSize, planned.Size, planned.DataOffset, planned.DataSize)
	}
	for i := range got.Tensors {
		if fmt.Sprint(got.Tensors[i]) != fmt.Sprint(planned.Tensors[i]) {
			t.Errorf("tensor %d: written %+v, planned %+v", i, got.Tensors[i], planned.Tensors[i])
		}
	}
	if err := got.SameTensors(inv); err != nil {
		t.Error(err)
	}
	// From the inventory alone the header's size is a bound: the offsets are
	// right to within one alignment.
	est, err := PlanInventory(inv, 4096)
	if err != nil || !est.Approximate || est.FileSize < l.FileSize-4096 || est.FileSize > l.FileSize+4096 {
		t.Errorf("from the inventory: %v, %+v", err, est)
	}
	if _, err := est.encodeHeader(f, nil); err == nil {
		t.Error("an approximate plan wrote a header")
	}
}

func TestRelayoutRefusesToOverwrite(t *testing.T) {
	src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
	dst, _ := relayout(t, src, RelayoutOptions{Alignment: 4096})
	before := fileSHA(t, dst)
	_, err := Relayout(context.Background(), src, dst, RelayoutOptions{Alignment: 4096})
	if !errors.Is(err, ErrExists) {
		t.Fatalf("overwriting without force: %v", err)
	}
	if fileSHA(t, dst) != before {
		t.Error("the refusal changed the file")
	}
	if _, err := Relayout(context.Background(), src, src, RelayoutOptions{Force: true}); err == nil {
		t.Error("the source was relaid onto itself")
	}
	// Force replaces it, with another alignment, and the sidecar follows.
	sc, err := Relayout(context.Background(), src, dst, RelayoutOptions{Alignment: 8192, Force: true})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := ReadSidecar(SidecarPath(dst)); err != nil || got.Alignment != 8192 || got.Result.SHA256 != sc.Result.SHA256 {
		t.Errorf("sidecar after force: %+v, %v", got, err)
	}
	if _, err := os.Stat(dst + ".part"); err == nil {
		t.Error("a temporary file is left")
	}
}

func TestRelayoutCancelledLeavesNothing(t *testing.T) {
	src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
	dst := NanoPath(src)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Relayout(ctx, src, dst, RelayoutOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("a cancelled relayout: %v", err)
	}
	for _, p := range []string{dst, dst + ".part", SidecarPath(dst)} {
		if _, err := os.Stat(p); err == nil {
			t.Errorf("%s is left", p)
		}
	}
}

func TestRelayoutRejects(t *testing.T) {
	for name, w := range map[string]*Writer{
		"another architecture": NewWriter().Set("general.architecture", "qwen35").
			Tensor("token_embd.weight", TypeQ4_0, []int64{32, 4}, nil),
		"no tensors": NewWriter().Set("general.architecture", ArchGemma4),
	} {
		src := writeModel(t, "m.gguf", w)
		if _, err := Relayout(context.Background(), src, NanoPath(src), RelayoutOptions{}); err == nil {
			t.Errorf("%s: relaid", name)
		}
	}
	// A tensor the group table does not know is not placed by guess.
	w := TinyGemma4().Tensor("blk.0.mystery.weight", TypeF32, []int64{4}, nil)
	src := writeModel(t, "m.gguf", w)
	_, err := Relayout(context.Background(), src, NanoPath(src), RelayoutOptions{})
	if err == nil || !strings.Contains(err.Error(), "mystery") {
		t.Errorf("an unknown tensor: %v", err)
	}
	src = writeModel(t, "ok.gguf", TinyGemma4())
	for _, a := range []int64{16, 48, 4097, 1 << 40} {
		if _, err := Relayout(context.Background(), src, NanoPath(src), RelayoutOptions{Alignment: a}); err == nil {
			t.Errorf("alignment %d accepted", a)
		}
	}
	if err := CheckAlignment(2 << 20); err != nil {
		t.Errorf("2 MiB: %v", err)
	}
}

func TestSidecar(t *testing.T) {
	src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
	dst, sc := relayout(t, src, RelayoutOptions{Alignment: 4096, SourceSHA256: fileSHA(t, src), Tool: "nanollm test"})
	if filepath.Base(SidecarPath(dst)) != "tiny.nano.json" || filepath.Base(dst) != "tiny.nano.gguf" {
		t.Errorf("names %s, %s", dst, SidecarPath(dst))
	}
	got, err := ReadSidecar(SidecarPath(dst))
	if err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(dst)
	sst, _ := os.Stat(src)
	f, _ := Open(dst)
	if got.Schema != SidecarSchema || got.Layout != LayoutNano || got.LayoutVersion != LayoutVersion || got.Alignment != 4096 ||
		got.Tensors != len(f.Tensors) || got.Tool != "nanollm test" ||
		got.Source.File != "tiny.gguf" || got.Source.Size != sst.Size() || got.Source.SHA256 != fileSHA(t, src) ||
		got.Result.File != "tiny.nano.gguf" || got.Result.Size != st.Size() || got.Result.SHA256 != fileSHA(t, dst) ||
		!got.Result.MTime.Equal(st.ModTime()) || len(got.Sections) != 4 || got.Result.SHA256 != sc.Result.SHA256 {
		t.Errorf("sidecar %+v", got)
	}
	if err := got.Verify(dst); err != nil {
		t.Errorf("a fresh copy: %v", err)
	}

	// Cheap checks: a changed size, a touched file, another layout version.
	touched := time.Now().Add(-time.Hour)
	if err := os.Chtimes(dst, touched, touched); err != nil {
		t.Fatal(err)
	}
	if err := got.Verify(dst); err == nil || !strings.Contains(err.Error(), "modified") {
		t.Errorf("a touched copy: %v", err)
	}
	if err := Restamp(dst); err != nil {
		t.Fatal(err)
	}
	got, _ = ReadSidecar(SidecarPath(dst))
	if err := got.Verify(dst); err != nil {
		t.Errorf("after restamp: %v", err)
	}
	old := got
	old.LayoutVersion = LayoutVersion + 1
	if err := old.Verify(dst); err == nil || !strings.Contains(err.Error(), "version") {
		t.Errorf("another layout version: %v", err)
	}
	short := got
	short.Result.Size--
	if err := short.Verify(dst); err == nil {
		t.Error("a size that is not the file's")
	}
	if err := got.Verify(dst + ".missing"); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a missing copy: %v", err)
	}
	if _, err := ReadSidecar(filepath.Join(t.TempDir(), "none.json")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("a missing sidecar: %v", err)
	}
}

func TestVerifyRelayoutFindsDamage(t *testing.T) {
	src := writeModel(t, "tiny.gguf", TinyGemma4().FillPattern())
	dst, _ := relayout(t, src, RelayoutOptions{Alignment: 4096})
	g, _ := Open(dst)
	clean, _ := os.ReadFile(dst)

	flip := func(at int64) {
		b := bytes.Clone(clean)
		b[at] ^= 0x40
		if err := os.WriteFile(dst, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	ti, _ := g.Tensor("blk.3.ffn_down.weight")
	pin, _ := g.Tensor("blk.2.ffn_norm.weight")
	cases := map[string]struct {
		at   int64
		want string
	}{
		"a tensor":      {g.FileOffset(ti) + 5, "blk.3.ffn_down.weight"},
		"a pinned norm": {g.FileOffset(pin), "blk.2.ffn_norm.weight"},
		"the padding":   {g.FileOffset(ti) + ti.Size + 1, "blk.3.ffn_down.weight"},
		"the header":    {30, "header"},
	}
	for name, c := range cases {
		flip(c.at)
		_, err := VerifyRelayout(context.Background(), src, dst, nil)
		if err == nil {
			t.Errorf("%s: damage not found", name)
			continue
		}
		// Padding after a tensor is found while the next one is being compared.
		if name != "the padding" && !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v", name, err)
		}
	}
	if err := os.WriteFile(dst, clean[:len(clean)-1], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyRelayout(context.Background(), src, dst, nil); err == nil {
		t.Error("a truncated copy verified")
	}
	if err := os.WriteFile(dst, clean, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyRelayout(context.Background(), src, dst, nil); err != nil {
		t.Errorf("the restored copy: %v", err)
	}
	// A source other than the one the sidecar was made from.
	other := writeModel(t, "other.gguf", TinyGemma4())
	if _, err := VerifyRelayout(context.Background(), other, dst, nil); err == nil {
		t.Error("a copy verified against another file")
	}
}

func TestSameTensors(t *testing.T) {
	f, _ := Open(writeModel(t, "tiny.gguf", TinyGemma4()))
	inv, _ := BuildInventory(f)
	rev := inv
	rev.Tensors = slices.Clone(inv.Tensors)
	slices.Reverse(rev.Tensors)
	for i := range rev.Tensors {
		rev.Tensors[i].Offset += 7
	}
	if err := inv.SameTensors(rev); err != nil {
		t.Errorf("order and offsets do not matter: %v", err)
	}
	short := inv
	short.Tensors = inv.Tensors[1:]
	if inv.SameTensors(short) == nil {
		t.Error("a missing tensor")
	}
	bad := inv
	bad.Tensors = slices.Clone(inv.Tensors)
	bad.Tensors[3].Bytes++
	if err := inv.SameTensors(bad); err == nil || !strings.Contains(err.Error(), bad.Tensors[3].Name) {
		t.Errorf("a changed size: %v", err)
	}
}

func TestParseRecordsHeaderSpans(t *testing.T) {
	b, err := NewWriter().Set("a", uint32(1)).Set(KeyAlignment, uint32(64)).Set("b", "xy").
		Tensor("t.weight", TypeF32, []int64{4}, nil).Bytes()
	if err != nil {
		t.Fatal(err)
	}
	f, err := Parse(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	off, ok := f.AlignmentOffset()
	if !ok || b[off] != 64 || f.KVEnd <= off || f.TableEnd <= f.KVEnd || f.DataOffset != pad(f.TableEnd, 64) {
		t.Errorf("alignment value at %d (%v), metadata ends %d, table ends %d, data %d", off, ok, f.KVEnd, f.TableEnd, f.DataOffset)
	}
	f, _ = parseBytes(t, NewWriter().Tensor("t.weight", TypeF32, []int64{4}, nil))
	if _, ok := f.AlignmentOffset(); ok {
		t.Error("an alignment offset without the key")
	}
}
