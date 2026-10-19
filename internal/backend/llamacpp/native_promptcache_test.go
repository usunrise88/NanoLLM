//go:build native

package llamacpp

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/promptcache"
)

// TestNativePromptCacheRoundTrip is OPT-13 on a real sequence: a prefilled
// prefix goes through SaveSeq into a promptcache.Store on disk, a second
// store opened on the same directory (the process restarted) finds it by
// token ids, and RestoreSeq puts it into another slot and into the emptied
// first one. Greedy decoding from the restored state must give the tokens
// that the slot which computed the prefix gives, to the token.
//
// The second case has a prefix longer than the sliding window of the
// model's local-attention layers. The context is loaded with swa_full off
// (baseSpec), so the saved state holds only the last window of those layers:
// what a restore needs is exactly what is in it.
func TestNativePromptCacheRoundTrip(t *testing.T) {
	sizes := []int{96}
	if w := sharedModel(t).caps.SWAWindow; w > 0 {
		sizes = append(sizes, w+128)
	}
	for _, prefix := range sizes {
		t.Run(fmt.Sprintf("prefix of %d tokens", prefix), func(t *testing.T) {
			m := sharedModel(t) // empties every slot
			sentence := "The quick brown fox jumps over the lazy dog. "
			full := tokenize(t, m, chat(m, strings.Repeat(sentence, (prefix+80)/9+1)))
			if len(full) < prefix+8 {
				t.Fatalf("setup: the prompt has %d tokens, the prefix %d", len(full), prefix)
			}
			if len(full)+16 > m.caps.NCtx {
				t.Skipf("a prompt of %d tokens does not fit a context of %d", len(full), m.caps.NCtx)
			}
			head, rest := full[:prefix], full[prefix:]

			// Slot 0 computes the prefix and is saved; then it goes on from there.
			if err := m.SetSampler(0, greedy, full); err != nil {
				t.Fatal(err)
			}
			for p := 0; p < len(head); p += m.caps.NBatch {
				end := min(p+m.caps.NBatch, len(head))
				if _, err := m.Step(context.Background(), []backend.SeqStep{{Seq: 0, Tokens: head[p:end], Pos: p, Logprobs: -1}}); err != nil {
					t.Fatal(err)
				}
			}
			state, err := m.SaveSeq(0)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("state of %d tokens: %d bytes, %d per token", prefix, len(state), len(state)/prefix)

			fp := promptcache.NewFingerprint("native-test", m.meta["general.name"], fmt.Sprint(m.caps.NCtx, m.caps.Slots))
			dir := t.TempDir()
			store, err := promptcache.Open(promptcache.Options{Dir: dir, Fingerprint: fp})
			if err != nil {
				t.Fatal(err)
			}
			if err := store.Save(head, state); err != nil {
				t.Fatal(err)
			}
			if err := store.Close(); err != nil { // writes it
				t.Fatal(err)
			}
			want := continueFrom(t, m, 0, rest, len(head), 8)
			if len(want) == 0 {
				t.Fatal("setup: the reference generated nothing")
			}

			// The restart: a store that knows the directory and nothing else.
			store, err = promptcache.Open(promptcache.Options{Dir: dir, Fingerprint: fp})
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			n, ok := store.Lookup(full)
			if !ok || n != len(head) {
				t.Fatalf("lookup found %d tokens (%v), want %d", n, ok, len(head))
			}
			loaded, err := store.Load(full, n)
			if err != nil {
				t.Fatal(err)
			}

			// Into the other slot, and into the first one after it was emptied.
			for _, seq := range []int{1, 0} {
				m.ClearSeq(seq)
				if err := m.RestoreSeq(seq, loaded); err != nil {
					t.Fatalf("restore into slot %d: %v", seq, err)
				}
				if err := m.SetSampler(seq, greedy, full); err != nil {
					t.Fatal(err)
				}
				got := continueFrom(t, m, seq, rest, n, 8)
				if fmt.Sprint(got) != fmt.Sprint(want) {
					t.Errorf("slot %d after a restore from disk: %v, want %v", seq, got, want)
				}
			}
		})
	}
}
