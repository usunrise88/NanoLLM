//go:build native

package llamacpp

import (
	"context"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
)

// TestNativeBackendSampling checks that sampling in the graph
// (LoadSpec.BackendSampling) returns what the host chain returns: the
// target's greedy pick and the drafter's top-10, so MTP decoding gives the
// same tokens and the same drafts. On the CPU device the graph's argmax and
// top-k run on the CPU backend; the mechanics are those of a GPU. A request
// that reads logprobs stays on the host, and a greedy chain with the same
// settings stays attached between requests.
func TestNativeBackendSampling(t *testing.T) {
	deviceGreedy := backend.SamplerSpec{Temperature: 0}
	b := testBackend(t)
	path, drafter := testModelPath(t), testDrafterPath(t)
	const n = 48
	type run struct {
		toks              []int
		drafted, accepted int
	}
	collect := func(m *model, spec backend.SamplerSpec) []run {
		var out []run
		for _, p := range equivalencePrompts {
			prompt := tokenize(t, m, chat(m, p))
			m.ClearSeq(0)
			if err := m.SetSampler(0, spec, prompt); err != nil {
				t.Fatal(err)
			}
			g := generate(t, m, 0, prompt, 0, n, -1)
			r := run{drafted: g.drafted, accepted: g.accepted}
			for _, tk := range g.toks {
				r.toks = append(r.toks, int(tk))
			}
			out = append(out, r)
		}
		return out
	}

	hostGreedy := deviceGreedy
	hostGreedy.RawLogits = true // the reference samples on the host even when the test device could do it in its graph
	want := collect(sharedMTPModel(t), hostGreedy)
	releaseSharedMTP()

	s := mtpSpec(path, drafter)
	s.BackendSampling = true
	lm, err := b.Load(context.Background(), s)
	if err != nil {
		t.Fatal(err)
	}
	m := lm.(*model)
	t.Cleanup(func() { _ = m.Close() })
	if m.dft == nil || len(m.dft.chains) != m.caps.Slots {
		t.Fatalf("drafter top-10 chains attached: %d of %d slots", len(m.dft.chains), m.caps.Slots)
	}
	got := collect(m, deviceGreedy)
	if !m.slots[0].onDevice {
		t.Fatal("a greedy chain did not go to the device")
	}

	var drafted, accepted int
	for i, p := range equivalencePrompts {
		w, g := want[i], got[i]
		drafted += g.drafted
		accepted += g.accepted
		if w.drafted != g.drafted || w.accepted != g.accepted {
			t.Errorf("%q: drafted/accepted %d/%d on the host, %d/%d on the device", p, w.drafted, w.accepted, g.drafted, g.accepted)
		}
		for j := range max(len(w.toks), len(g.toks)) {
			if j >= len(w.toks) || j >= len(g.toks) || w.toks[j] != g.toks[j] {
				t.Errorf("%q: first difference at token %d of %d/%d", p, j, len(w.toks), len(g.toks))
				break
			}
		}
	}
	t.Logf("device sampling: drafted %d, accepted %d", drafted, accepted)
	if accepted == 0 && trained(m) {
		t.Error("no draft token was accepted")
	}

	// the same greedy settings keep the attached chain
	chain := m.slots[0].chain
	if err := m.SetSampler(0, deviceGreedy, nil); err != nil {
		t.Fatal(err)
	}
	if m.slots[0].chain != chain || !m.slots[0].onDevice {
		t.Error("a second greedy request replaced the attached chain")
	}

	// logprobs need the logits rows: that chain samples on the host
	raw := deviceGreedy
	raw.RawLogits = true
	prompt := tokenize(t, m, chat(m, equivalencePrompts[0]))
	m.ClearSeq(0)
	if err := m.SetSampler(0, raw, prompt); err != nil {
		t.Fatal(err)
	}
	if m.slots[0].onDevice {
		t.Fatal("a chain for a request with logprobs went to the device")
	}
	g := generate(t, m, 0, prompt, 0, 8, 2)
	if len(g.lps) == 0 || len(g.lps[0].Top) == 0 {
		t.Fatal("no logprobs from a host chain on a device-sampling model")
	}
	for j := range min(len(g.toks), len(want[0].toks)) {
		if int(g.toks[j]) != want[0].toks[j] {
			t.Errorf("host chain beside device chains: first difference at token %d", j)
			break
		}
	}
}
