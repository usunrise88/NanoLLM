package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/registry"
)

func writeTiny(t *testing.T, w *gguf.Writer) string {
	t.Helper()
	b, err := w.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestInspectFile(t *testing.T) {
	p := writeTiny(t, gguf.TinyGemma4())
	var out bytes.Buffer
	if err := inspectCommand([]string{p, "--tensors"}, &out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"not a catalog file", "G1 ", "ple_proj", "KV layers    4 (4 sliding-window, 0 full)",
		"per_layer_token_embd.weight", "tied output true",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}

	// --json is the inventory format the catalog embeds.
	out.Reset()
	if err := inspectCommand([]string{"--json", "--tensors", p}, &out); err != nil {
		t.Fatal(err)
	}
	inv, err := registry.ParseInventory(out.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if inv.ID != "tiny" || len(inv.Files) != 1 || len(inv.Files[0].Tensors) == 0 {
		t.Errorf("inventory: %+v", inv)
	}
}

func TestInspectWarnsAboutUnknownTensors(t *testing.T) {
	p := writeTiny(t, gguf.TinyGemma4().Tensor("mystery.weight", gguf.TypeF32, []int64{4}, nil))
	var out bytes.Buffer
	if err := inspectCommand([]string{p}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `warning: tensor "mystery.weight"`) {
		t.Errorf("no warning:\n%s", out.String())
	}
}

func TestInspectBuiltin(t *testing.T) {
	// Point the data directory somewhere empty so the built-in inventory is
	// what gets shown.
	t.Setenv("NANOLLM_SERVER_DATA_DIR", t.TempDir())
	var out bytes.Buffer
	if err := inspectCommand([]string{"gemma-4-e4b-qat"}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "built-in inventory") || !strings.Contains(out.String(), "1585446912 B") {
		t.Errorf("output:\n%s", out.String())
	}
	if err := inspectCommand([]string{"no-such-model"}, &out); err == nil {
		t.Error("an unknown id was inspected")
	}
}
