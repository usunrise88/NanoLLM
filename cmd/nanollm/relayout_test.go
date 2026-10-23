package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/plan"
)

func TestParseAlignment(t *testing.T) {
	for in, want := range map[string]int64{"4096": 4096, "4k": 4096, "64K": 65536, "2m": 2 << 20, "32": 32} {
		if got, err := parseAlignment(in); err != nil || got != want {
			t.Errorf("%s = %d, %v", in, got, err)
		}
	}
	for _, in := range []string{"", "x", "48", "16", "3k", "-4k", "1g", "4.5k"} {
		if _, err := parseAlignment(in); err == nil {
			t.Errorf("%q accepted", in)
		}
	}
}

func TestRelayoutDryRunWritesNothing(t *testing.T) {
	src := writeTiny(t, gguf.TinyGemma4().FillPattern())
	dst := gguf.NanoPath(src)
	var out bytes.Buffer
	err := relayoutFile(context.Background(), &out, relayoutJob{ID: "tiny", Src: src, Dst: dst, Align: 4096, Dry: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"layout nano version 1, alignment 4096", "tiny.nano.gguf", "tensors", "pinned", "layers", "ranges the planner makes", "M8 ram-2g"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
	for _, p := range []string{dst, gguf.SidecarPath(dst)} {
		if _, err := os.Stat(p); err == nil {
			t.Errorf("--dry-run wrote %s", p)
		}
	}
}

func TestRelayoutCommandFlow(t *testing.T) {
	src := writeTiny(t, gguf.TinyGemma4().FillPattern())
	dst := gguf.NanoPath(src)
	ctx := context.Background()
	job := relayoutJob{ID: "tiny", Src: src, Dst: dst, Align: 4096}

	var out bytes.Buffer
	job.Verify = true
	if err := relayoutFile(ctx, &out, job); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"wrote " + dst, "sha256", "sidecar", "verified: "} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
	sc, err := gguf.ReadSidecar(gguf.SidecarPath(dst))
	if err != nil || sc.Alignment != 4096 || sc.Tool != "nanollm "+version {
		t.Fatalf("sidecar %+v, %v", sc, err)
	}
	// Run again: refused without --force, with the way out in the message.
	job.Verify = false
	err = relayoutFile(ctx, &bytes.Buffer{}, job)
	if err == nil || !strings.Contains(err.Error(), "--force") || !strings.Contains(err.Error(), "--verify") {
		t.Errorf("a second run: %v", err)
	}
	// --verify alone checks the copy that is there, and accepts a copy whose mtime moved.
	out.Reset()
	job.Verify = true
	if err := relayoutFile(ctx, &out, job); err != nil || !strings.Contains(out.String(), "verified: ") || strings.Contains(out.String(), "wrote ") {
		t.Fatalf("verify only: %v\n%s", err, out.String())
	}
	// A damaged copy fails --verify, with the tensor in the message.
	b, _ := os.ReadFile(dst)
	f, _ := gguf.Open(dst)
	ti, _ := f.Tensor("blk.1.ffn_up.weight")
	b[f.FileOffset(ti)] ^= 1
	if err := os.WriteFile(dst, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := relayoutFile(ctx, &bytes.Buffer{}, job); err == nil || !strings.Contains(err.Error(), "blk.1.ffn_up.weight") {
		t.Errorf("a damaged copy: %v", err)
	}
	// --force rewrites it.
	job.Force = true
	if err := relayoutFile(ctx, &bytes.Buffer{}, job); err != nil {
		t.Fatal(err)
	}
}

// The plan command reads a relaid file's sidecar and plans from its offsets.
func TestPlanCommandOnRelaidFile(t *testing.T) {
	src := writeTiny(t, gguf.TinyGemma4().FillPattern())
	dst := gguf.NanoPath(src)
	if _, err := gguf.Relayout(context.Background(), src, dst, gguf.RelayoutOptions{}); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := planCommand([]string{dst, "--preset", "ram-1g", "--json", "--layout", "nano"}, &out); err != nil {
		t.Fatal(err)
	}
	var p plan.Plan
	if err := json.Unmarshal(out.Bytes(), &p); err != nil {
		t.Fatal(err)
	}
	if p.Layout != "nano" {
		t.Errorf("plan layout %q", p.Layout)
	}
	// The published file is not a nano one, and the flag says so.
	out.Reset()
	if err := planCommand([]string{src, "--preset", "ram-1g", "--layout", "nano"}, &out); err == nil {
		t.Error("--layout nano on a file that is not a relaid copy")
	}
	if err := planCommand([]string{src, "--preset", "ram-1g", "--layout", "sorted"}, &out); err == nil {
		t.Error("an unknown layout")
	}
}
