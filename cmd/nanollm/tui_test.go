package main

import (
	"bufio"
	"bytes"
	"strings"
	"testing"
)

func keysOf(t *testing.T, input string) []key {
	t.Helper()
	r := bufio.NewReader(strings.NewReader(input))
	var out []key
	for {
		k, err := readKeys(r)
		if err != nil {
			return out
		}
		out = append(out, k)
	}
}

func TestReadKeys(t *testing.T) {
	for _, c := range []struct {
		name  string
		input string
		want  []key
	}{
		{"arrows", "\x1b[A\x1b[B\x1b[A", []key{keyUp, keyDown, keyUp}},
		{"application mode arrows", "\x1bOA\x1bOB", []key{keyUp, keyDown}},
		{"vi keys", "kjk", []key{keyUp, keyDown, keyUp}},
		{"enter", "\r", []key{keyEnter}},
		{"line feed", "\n", []key{keyEnter}},
		{"digits", "13", []key{keyDigit1, keyDigit1 + 2}},
		{"quit", "q\x03", []key{keyQuit, keyQuit}},
		{"a lone escape", "\x1b", []key{keyQuit}},
		{"yes and no", "yN", []key{keyYes, keyNo}},
		{"an unknown sequence is skipped", "\x1b[3~\x1b[B", []key{keyDown}},
		{"noise is ignored", "xyz!", []key{keyYes}},
	} {
		if got := keysOf(t, c.input); !equalKeys(got, c.want) {
			t.Errorf("%s: %q gave %v, want %v", c.name, c.input, got, c.want)
		}
	}
}

func equalKeys(a, b []key) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestRenderMenu(t *testing.T) {
	var b bytes.Buffer
	renderMenu(&b, "Choose a mode:", []menuItem{{"silent", "quiet"}, {"optimal", "balanced"}, {"powerful", "fast"}}, 1)
	got := b.String()
	if !strings.Contains(got, "> 2. optimal") || !strings.Contains(got, "  1. silent") || strings.Count(got, "\r\n") != 4 {
		t.Fatalf("%q", got)
	}
}

func TestWrap(t *testing.T) {
	got := wrap("one two three four five", 9)
	if len(got) != 3 || got[0] != "one two" || got[1] != "three" || got[2] != "four five" {
		t.Fatalf("%q", got)
	}
}
