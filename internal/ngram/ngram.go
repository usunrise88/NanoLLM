// Package ngram drafts tokens by lookup in the sequence itself ("prompt
// lookup decoding"): the last few tokens are searched for earlier in the
// prompt and the answer so far, and the tokens that followed that earlier
// occurrence are proposed as the next ones. A draft costs no model pass, so
// a miss costs only the slot in the verification batch; on a task that
// repeats its input (a summary that quotes, a JSON built from given fields,
// code that is edited, a retrieved passage that is answered from) a run of
// them is accepted at once. The target model still samples every position,
// so the output is exactly what it would have been without drafting
// (docs/STAGE2.md, OPT-5).
package ngram

// Window is how far back from the end Propose looks: the scan is linear, and
// a match in the far past of a long context is no likelier than a recent one.
const Window = 16384

// DefaultN is the match length used when none is configured.
const DefaultN = 3

// MinN is the shortest suffix Propose falls back to.
const MinN = 2

// Propose returns up to max tokens that followed the most recent earlier
// occurrence of the sequence's last n tokens, where the sequence is hist
// followed by next. It tries suffixes from n down to MinN and returns the
// continuation of the longest one that has a match with at least one token
// after it. nil means no proposal.
func Propose[T ~int32](hist []T, next T, n, max int) []T {
	if max <= 0 || n < MinN {
		return nil
	}
	total := len(hist) + 1
	at := func(i int) T {
		if i == len(hist) {
			return next
		}
		return hist[i]
	}
	lo := max0(total - Window)
	for nn := n; nn >= MinN; nn-- {
		if total < nn+1 {
			continue
		}
		start := total - nn // the suffix is [start, total)
		first := at(start)
		// Scan backwards for an earlier copy; the copy has to leave room for
		// at least one continuation token before the suffix itself begins.
		for i := start - 1; i >= lo; i-- {
			if at(i) != first {
				continue
			}
			ok := true
			for k := 1; k < nn; k++ {
				if at(i+k) != at(start+k) {
					ok = false
					break
				}
			}
			if !ok {
				continue
			}
			c := i + nn
			end := min(c+max, total)
			if end <= c {
				continue
			}
			out := make([]T, 0, end-c)
			for j := c; j < end; j++ {
				out = append(out, at(j))
			}
			return out
		}
	}
	return nil
}

func max0(a int) int {
	if a < 0 {
		return 0
	}
	return a
}

// Backoff throttles a slot whose drafts keep missing: after a draft that
// was entirely rejected, the next few steps draft nothing, longer after each
// further miss, so free text (where a lookup rarely helps) pays almost
// nothing for the attempts. An accepted token resets it.
type Backoff struct {
	misses int
	skip   int
}

// MaxSkip is the longest pause, in steps.
const MaxSkip = 32

// Allow reports whether this step may draft; a step that may not uses up one
// step of the pause.
func (b *Backoff) Allow() bool {
	if b.skip > 0 {
		b.skip--
		return false
	}
	return true
}

// Result records what a draft of drafted tokens got: accepted of them were
// kept.
func (b *Backoff) Result(drafted, accepted int) {
	if drafted == 0 {
		return
	}
	if accepted > 0 {
		b.misses, b.skip = 0, 0
		return
	}
	b.misses++
	b.skip = min(1<<min(b.misses, 6), MaxSkip) - 1
}
