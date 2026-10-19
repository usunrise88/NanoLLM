package main

import (
	"strings"
	"testing"
)

func TestProseOf(t *testing.T) {
	in := `<div class="mw-parser-output"><p>Волга — река в Европейской части России.<sup class="reference">[1]</sup> Её длина &mdash; 3530&nbsp;км.</p>
<table><tr><td>Таблица</td></tr></table>
<h2><span class="mw-headline">География</span></h2>
<p>The river <a href="/wiki/X">flows</a> south [citation needed].</p>
<ul><li>Меню</li><li>Этот пункт списка достаточно длинный, чтобы считаться частью связного текста статьи.</li></ul>
<style>.x{color:red}</style><p></p></div>`
	got := ProseOf(in)
	for _, want := range []string{"Волга — река в Европейской части России. Её длина — 3530 км.", "География", "The river flows south .", "Этот пункт списка достаточно длинный"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in %q", want, got)
		}
	}
	for _, bad := range []string{"Таблица", "Меню", "color:red", "[1]", "citation"} {
		if strings.Contains(got, bad) {
			t.Errorf("%q should have been dropped from %q", bad, got)
		}
	}
}

func TestTruncate(t *testing.T) {
	text := "первый абзац\nвторой абзац\nтретий абзац\n"
	if got := Truncate(text, 0); got != text {
		t.Errorf("no limit: %q", got)
	}
	if got := Truncate(text, 40); got != "первый абзац\n" {
		t.Errorf("cut at a paragraph: %q", got)
	}
	// A first paragraph longer than the limit is still kept whole.
	if got := Truncate("один очень длинный абзац без перевода строки", 5); !strings.HasPrefix(got, "один очень") {
		t.Errorf("first paragraph: %q", got)
	}
}
