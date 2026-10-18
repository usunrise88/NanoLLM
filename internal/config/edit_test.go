package config

import (
	"strings"
	"testing"
)

func TestDocumentKeepsCommentsAndEditsKeys(t *testing.T) {
	in := "# header\nserver:\n  addr: 127.0.0.1:8080 # inline\nauth:\n  mode: apikey\n"
	d, err := ParseDocument([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	if err := d.AddKey(APIKey{Name: "a", Secret: "sk-nanollm-1", Admin: true, RPS: 2.5}); err != nil {
		t.Fatal(err)
	}
	if err := d.AddKey(APIKey{Name: "b", Secret: HashSecret("x"), Interactive: true, RPS: 5}); err != nil {
		t.Fatal(err)
	}
	if err := d.AddKey(APIKey{Name: "a", Secret: "other"}); err == nil {
		t.Fatal("duplicate name accepted")
	}
	out, err := d.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	for _, want := range []string{"# header", "# inline", "rps: 2.5", "rps: 5\n", "interactive: true", "secret: sha256:"} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %q in:\n%s", want, s)
		}
	}
	if !d.RemoveKey("a") || d.RemoveKey("a") || len(d.Keys()) != 1 {
		t.Fatalf("remove: %+v", d.Keys())
	}
	// The edited file loads.
	if out, err = d.Bytes(); err != nil {
		t.Fatal(err)
	}
	cfg := Default()
	if err := decode(out, &cfg); err != nil {
		t.Fatal(err)
	}
	if len(cfg.Auth.Keys) != 1 || cfg.Auth.Keys[0].RPS != 5 {
		t.Fatalf("%+v", cfg.Auth.Keys)
	}
}
