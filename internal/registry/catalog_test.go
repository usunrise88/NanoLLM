package registry

import (
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/config"
)

func TestBuiltinCatalog(t *testing.T) {
	cat, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	def, ok := Find(cat, config.DefaultModel)
	if !ok {
		t.Fatalf("the default model %q is not in the catalog", config.DefaultModel)
	}
	// The primary configuration is the target plus its MTP drafter (SPEC §1).
	if _, ok := def.File(RoleDrafter); !ok || def.Drafter == nil {
		t.Errorf("%s ships without its drafter", def.ID)
	}
	if fb, ok := Find(cat, "qwen3.5-9b-q4km"); !ok || fb.Drafter != nil {
		t.Errorf("fallback missing or carries a drafter: %+v", fb)
	}
}

func TestUnpinnedFilesCannotBeDownloaded(t *testing.T) {
	f := File{Repo: "unsloth/x", Path: "x.gguf"}
	if _, err := f.URL(); err == nil {
		t.Fatal("an unpinned file produced a download URL")
	}
	f.Revision = strings.Repeat("a", 40)
	f.SHA256 = strings.Repeat("b", 64)
	f.Size = 1
	u, err := f.URL()
	if err != nil {
		t.Fatal(err)
	}
	if want := "https://huggingface.co/unsloth/x/resolve/" + f.Revision + "/x.gguf"; u != want {
		t.Errorf("got %s, want %s", u, want)
	}
}

func TestParseCatalogRejects(t *testing.T) {
	for name, doc := range map[string]string{
		"duplicate id": "- {id: a, format: gguf, commercial: permitted, files: [{role: model}]}\n" +
			"- {id: a, format: gguf, commercial: permitted, files: [{role: model}]}\n",
		"no model file":  "- {id: a, format: gguf, commercial: permitted, files: []}\n",
		"unknown field":  "- {id: a, formt: gguf}\n",
		"unknown format": "- {id: a, format: onnx, commercial: permitted, files: [{role: model}]}\n",
	} {
		if _, err := parseCatalog([]byte(doc)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}
