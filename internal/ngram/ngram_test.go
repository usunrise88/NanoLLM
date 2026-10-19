package ngram

import (
	"slices"
	"testing"
)

type tok int32

func TestPropose(t *testing.T) {
	// "a b c d e a b c" + next "d": the suffix "b c d" occurred before and
	// "e" followed it... the sequence is hist+next.
	hist := []tok{1, 2, 3, 4, 5, 1, 2, 3}
	for _, tt := range []struct {
		name string
		hist []tok
		next tok
		n, k int
		want []tok
	}{
		{"copy after a match", hist, 4, 3, 4, []tok{5, 1, 2, 3}},
		{"limited by max", hist, 4, 3, 2, []tok{5, 1}},
		{"continuation may run into the suffix", []tok{7, 8, 7}, 8, 2, 4, []tok{7, 8}},
		{"no earlier copy", []tok{1, 2, 3, 4}, 9, 3, 4, nil},
		{"falls back to a shorter suffix", []tok{1, 2, 3, 9, 2}, 3, 3, 3, []tok{9, 2, 3}},
		{"the longest suffix wins over a more recent short one", []tok{5, 1, 2, 3, 6, 9, 2, 3, 7, 8, 1, 2}, 3, 3, 1, []tok{6}},
		{"too short a history", []tok{1}, 1, 3, 4, nil},
		{"empty history", nil, 1, 3, 4, nil},
		{"zero max", hist, 4, 3, 0, nil},
		{"n below the minimum", hist, 4, 1, 4, nil},
		{"most recent match", []tok{1, 2, 3, 10, 1, 2, 3, 20, 1, 2}, 3, 3, 1, []tok{20}},
	} {
		got := Propose(tt.hist, tt.next, tt.n, tt.k)
		if !slices.Equal(got, tt.want) {
			t.Errorf("%s: Propose = %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestProposeLeavesHistoryAlone(t *testing.T) {
	hist := []tok{1, 2, 3, 1, 2}
	cp := slices.Clone(hist)
	_ = Propose(hist, 3, 3, 4)
	if !slices.Equal(hist, cp) {
		t.Errorf("history changed: %v", hist)
	}
}

func TestProposeWindow(t *testing.T) {
	// A match older than the window is not found.
	hist := make([]tok, Window+100)
	for i := range hist {
		hist[i] = tok(1000 + i) // all distinct
	}
	copy(hist[:3], []tok{7, 8, 9})
	copy(hist[len(hist)-2:], []tok{7, 8})
	if got := Propose(hist, 9, 3, 4); got != nil {
		t.Errorf("found a match %v beyond the window", got)
	}
	// ...and one inside it is.
	copy(hist[len(hist)-Window/2:], []tok{7, 8, 9, 55})
	if got := Propose(hist, 9, 3, 1); !slices.Equal(got, []tok{55}) {
		t.Errorf("match inside the window: %v", got)
	}
}

func TestBackoff(t *testing.T) {
	var b Backoff
	if !b.Allow() {
		t.Fatal("a fresh slot may not draft")
	}
	b.Result(3, 0) // first miss: pause 1 step
	if b.Allow() {
		t.Error("drafted right after a miss")
	}
	if !b.Allow() {
		t.Error("the pause outlasted one step")
	}
	b.Result(3, 0) // second miss: 3 steps
	paused := 0
	for !b.Allow() {
		paused++
	}
	if paused != 3 {
		t.Errorf("paused %d steps after the second miss, want 3", paused)
	}
	b.Result(2, 1)
	if !b.Allow() || b.misses != 0 {
		t.Error("an accepted token did not reset the backoff")
	}
	for range 20 {
		b.Result(1, 0)
	}
	n := 0
	for !b.Allow() {
		n++
	}
	if n > MaxSkip-1 {
		t.Errorf("pause %d exceeds the cap %d", n, MaxSkip)
	}
	b.Result(0, 0) // nothing drafted: no change
}
