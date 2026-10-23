package registry

import (
	"strings"
	"testing"
)

func TestShortlists(t *testing.T) {
	cat, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := Find(cat, "gemma-4-e4b-qat")
	if !ok {
		t.Fatal("the model is not in the catalog")
	}
	f, _ := m.File(RoleModel)
	big, err := LoadShortlist(m.ID, HeadShortlist, f.SHA256)
	if err != nil {
		t.Fatal(err)
	}
	small, err := LoadShortlist(m.ID, HeadShortlistSmall, f.SHA256)
	if err != nil {
		t.Fatal(err)
	}
	if big.Vocab != 262144 || len(big.IDs) != 170024 || len(small.IDs) != 96248 || len(small.IDs) >= len(big.IDs) {
		t.Fatalf("rows %d and %d of %d", len(big.IDs), len(small.IDs), big.Vocab)
	}
	for _, sl := range []*Shortlist{big, small} {
		for i, id := range sl.IDs {
			if id < 0 || int(id) >= sl.Vocab || i > 0 && id <= sl.IDs[i-1] {
				t.Fatalf("id %d at %d is out of order or out of the vocabulary", id, i)
			}
		}
	}
	// The small list is a part of the big one (the corpus-seen tokens are in both).
	in := map[int32]bool{}
	for _, id := range big.IDs {
		in[id] = true
	}
	missing := 0
	for _, id := range small.IDs {
		if !in[id] {
			missing++
		}
	}
	if missing > 8 { // the top-up of the last group of 8 may differ
		t.Errorf("%d ids of the small list are not in the big one", missing)
	}
	// Control tokens are listed: the end of a turn must be producible.
	if !in[1] && !in[2] {
		t.Error("neither BOS nor EOS is in the list")
	}
}

func TestShortlistRefusals(t *testing.T) {
	if _, err := LoadShortlist("gemma-4-e4b-qat", HeadShortlist, strings.Repeat("0", 64)); err == nil || !strings.Contains(err.Error(), "was made for the file") {
		t.Errorf("another file: %v", err)
	}
	if _, err := LoadShortlist("no-such-model", HeadShortlist, ""); err == nil {
		t.Error("an unknown model has no list")
	}
	if _, err := LoadShortlist("gemma-4-e4b-qat", "tiny", ""); err == nil {
		t.Error("an unknown kind")
	}
	if _, err := parseShortlist([]byte("NLSL")); err == nil {
		t.Error("a truncated list")
	}
}
