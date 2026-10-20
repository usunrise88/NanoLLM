package bench

import (
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
)

func TestVariantStreamKeys(t *testing.T) {
	vs, err := ParseSets([]string{"stream_ahead=0,2", "stream_evict=on"})
	if err != nil || len(vs) != 2 {
		t.Fatalf("%v %v", vs, err)
	}
	spec := backend.LoadSpec{Stream: &backend.StreamSpec{Ranges: []backend.StreamRange{{Layer: 0, Offset: 1, Length: 2}}, Keep: 3}}
	vs[1].Load(&spec)
	if st := spec.Stream; st.Ahead != 2 || !st.Evict || st.Keep != 3 || len(st.Ranges) != 1 {
		t.Errorf("stream %+v", st)
	}
	// A plan with no file-backed layers has no spec to change.
	none := backend.LoadSpec{}
	vs[1].Load(&none)
	if none.Stream != nil {
		t.Errorf("a stream spec out of nothing: %+v", none.Stream)
	}
	if _, err := ParseSets([]string{"stream_evict=maybe"}); err == nil {
		t.Error("stream_evict=maybe accepted")
	}
}

func TestVariantSetsSpeculation(t *testing.T) {
	if (Variant{}).setsSpeculation() || (Variant{"threads": "8", "kv": "q8_0"}).setsSpeculation() {
		t.Error("a variant with no speculation key must get the CPU default")
	}
	for _, k := range []string{"draft_max", "draft_p_min", "mtp", "ngram"} {
		if !(Variant{k: "1"}).setsSpeculation() {
			t.Errorf("%s names a speculation setting", k)
		}
	}
}
