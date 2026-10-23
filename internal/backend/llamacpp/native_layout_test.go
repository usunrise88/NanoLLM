//go:build native

package llamacpp

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/memlock"
	"github.com/usunrise88/nanollm/internal/plan"
)

// OPT-7: the model file with its tensors in another order, byte for byte the
// same tensors, loads and computes exactly what the published file does. The
// test relays the test model out once (a copy as large as the model is
// written, to NANOLLM_TEST_NANO_DIR or the test's temporary directory; or
// NANOLLM_TEST_NANO_FILE names a copy `nanollm models relayout` made), checks
// the copy against the original, and decodes the same prompt greedily from
// both in the file-mapped, repacked and locked modes and under M8's plan,
// with its lock and stream ranges taken from each file's own inventory.
// Slow; NANOLLM_TEST_NANO=1 runs it.
func TestNativeNanoLayout(t *testing.T) {
	if os.Getenv("NANOLLM_TEST_NANO") == "" {
		t.Skip("set NANOLLM_TEST_NANO=1: this writes a relaid copy of the test model (4 GB for E4B) and loads both")
	}
	b := testBackend(t)
	src := testModelPath(t)
	dir := os.Getenv("NANOLLM_TEST_NANO_DIR")
	if dir == "" {
		dir = t.TempDir()
	}
	dst := os.Getenv("NANOLLM_TEST_NANO_FILE")
	ctx := context.Background()
	if dst == "" {
		dst = filepath.Join(dir, filepath.Base(gguf.NanoPath(src)))
		if _, err := gguf.Relayout(ctx, src, dst, gguf.RelayoutOptions{Force: true, Tool: "native test"}); err != nil {
			t.Fatal(err)
		}
		defer os.Remove(dst)
		defer os.Remove(gguf.SidecarPath(dst))
	}
	// Either way the copy is compared with the original before it is trusted.
	sc, err := gguf.ReadSidecar(gguf.SidecarPath(dst))
	if err != nil {
		t.Fatal(err)
	}
	rep, err := gguf.VerifyRelayout(ctx, src, dst, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("relaid %s: %d bytes (the original %d), %d tensors / %d bytes identical, sha256 %s",
		dst, sc.Result.Size, sc.Source.Size, rep.Tensors, rep.Bytes, rep.SHA256)

	// One inventory per file, from its own header.
	inventory := func(path string, layout string) gguf.Inventory {
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		inv, err := gguf.BuildInventory(f)
		if err != nil {
			t.Fatal(err)
		}
		inv.Role, inv.Layout = "model", layout
		return inv
	}
	orig, nano := inventory(src, ""), inventory(dst, gguf.LayoutNano)
	if err := nano.SameTensors(orig); err != nil {
		t.Fatalf("the copy holds other tensors: %v", err)
	}

	type run struct {
		toks   []backend.Token
		stream memlock.StreamStats
		spec   backend.LoadSpec
	}
	load := func(spec backend.LoadSpec, pins ...memlock.Range) run {
		if len(pins) > 0 {
			pin, err := memlock.PinFile(spec.Path, pins)
			if err != nil {
				t.Fatalf("range-lock: %v", err)
			}
			defer pin.Close()
		}
		mm, err := b.Load(ctx, spec)
		if err != nil {
			t.Fatal(err)
		}
		m := mm.(*model)
		defer m.Close()
		prompt := tokenize(t, m, "The three primary colours are")
		toks := decode(t, m, 0, prompt, 24, greedy)
		return run{toks: toks, stream: m.StreamStats(), spec: spec}
	}

	files := func(path string) backend.LoadSpec {
		s := baseSpec(path)
		s.Slots = 1
		return s
	}
	for name, mutate := range map[string]func(*backend.LoadSpec){
		"mmap":         func(s *backend.LoadSpec) { s.Repack, s.LoadMode = false, backend.LoadMMap },
		"repack":       func(s *backend.LoadSpec) { s.Repack, s.LoadMode = true, backend.LoadMMap },
		"mmap+mlock":   func(s *backend.LoadSpec) { s.Repack, s.LoadMode = false, backend.LoadMMapMLock },
		"mlock+repack": func(s *backend.LoadSpec) { s.Repack, s.LoadMode = true, backend.LoadMLock },
	} {
		t.Run(name, func(t *testing.T) {
			so, sn := files(src), files(dst)
			mutate(&so)
			mutate(&sn)
			want, got := load(so), load(sn)
			if !slices.Equal(got.toks, want.toks) {
				t.Errorf("the relaid file changed the output:\n got %v\nwant %v", got.toks, want.toks)
			}
			t.Logf("%d tokens identical: %v", len(got.toks), got.toks)
		})
	}

	// M8's plan, each file with its own ranges: the pinned set locked as a
	// server does, the cold FFN layers streamed.
	t.Run("M8 plan", func(t *testing.T) {
		if runtime.GOOS != "linux" {
			t.Skip("range-lock and layer streaming are exercised on Linux")
		}
		pre, err := plan.ParsePreset("ram-2g")
		if err != nil {
			t.Fatal(err)
		}
		planned := func(inv gguf.Inventory, path string) (backend.LoadSpec, []memlock.Range, *plan.Plan) {
			model, err := plan.ModelFrom("native", []gguf.Inventory{inv})
			if err != nil {
				t.Fatal(err)
			}
			p, err := plan.Make(plan.Request{Model: model, Preset: pre, Mode: plan.RAMHybrid, NoMTP: true, NCtx: 1024, Slots: 1})
			if err != nil || !p.OK() {
				t.Skipf("the test model does not plan as M8 under ram-2g: %v %s", err, p.Refused)
			}
			spec, err := p.LoadSpec(path, "")
			if err != nil {
				t.Fatal(err)
			}
			spec.Threads, spec.ThreadsBatch = 4, 4
			if spec.Stream != nil {
				spec.Stream.Ahead, spec.Stream.Evict, spec.Stream.Keep = 2, true, 2
			}
			var locks []memlock.Range
			for _, r := range p.LockRanges {
				locks = append(locks, memlock.Range{Off: r.Offset, Len: r.Length})
			}
			return spec, locks, p
		}
		so, lo, po := planned(orig, src)
		sn, ln, pn := planned(nano, dst)
		t.Logf("original: %d lock ranges, %d stream ranges; nano: %d lock ranges, %d stream ranges",
			len(po.LockRanges), len(po.StreamRanges), len(pn.LockRanges), len(pn.StreamRanges))
		if len(pn.LockRanges) >= len(po.LockRanges) || len(pn.StreamRanges) >= len(po.StreamRanges) {
			t.Errorf("the relaid plan does not have fewer ranges")
		}
		want, got := load(so, lo...), load(sn, ln...)
		if !slices.Equal(got.toks, want.toks) {
			t.Errorf("M8 on the relaid file changed the output:\n got %v\nwant %v", got.toks, want.toks)
		}
		if len(pn.StreamRanges) > 0 {
			s := got.stream
			t.Logf("streamer on the relaid file: %+v", s)
			if s.Calls == 0 || s.WillNeed == 0 || s.PageOut == 0 || s.Errors != 0 {
				t.Errorf("the streamer was not fed from the relaid ranges: %+v", s)
			}
		}
	})
}
