package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/term"
)

// A small terminal menu on golang.org/x/term, with no framework (the binary
// has four direct dependencies and keeps them): the arrow keys or j and k move,
// a digit jumps, Enter chooses, q or Esc or Ctrl-C leaves.

// key is a key press as the menu understands it.
type key int

const (
	keyNone key = iota
	keyUp
	keyDown
	keyEnter
	keyQuit
	keyYes
	keyNo
	keyDigit1 // keyDigit1+n is the digit n+1
)

// readKeys turns a byte stream from a raw terminal into key presses, one per
// call; it returns io.EOF when the stream ends. An escape followed by [ or O
// and a letter is an arrow; an escape alone is a quit.
func readKeys(r *bufio.Reader) (key, error) {
	for {
		b, err := r.ReadByte()
		if err != nil {
			return keyNone, err
		}
		switch {
		case b == 0x03, b == 'q', b == 'Q':
			return keyQuit, nil
		case b == '\r', b == '\n':
			return keyEnter, nil
		case b == 'k', b == 'K':
			return keyUp, nil
		case b == 'j', b == 'J':
			return keyDown, nil
		case b == 'y', b == 'Y':
			return keyYes, nil
		case b == 'n', b == 'N':
			return keyNo, nil
		case b >= '1' && b <= '9':
			return keyDigit1 + key(b-'1'), nil
		case b == 0x1b:
			// Sequences arrive in one read; an escape with nothing behind it is
			// the Escape key.
			if r.Buffered() == 0 {
				return keyQuit, nil
			}
			next, _ := r.ReadByte()
			if next != '[' && next != 'O' {
				return keyQuit, nil
			}
			c, _ := r.ReadByte()
			switch c {
			case 'A':
				return keyUp, nil
			case 'B':
				return keyDown, nil
			}
			// Another sequence (Home, Delete, …): skip its parameters.
			for c >= '0' && c <= '?' && r.Buffered() > 0 {
				c, _ = r.ReadByte()
			}
		}
	}
}

// menuItem is one line of a menu.
type menuItem struct {
	Label  string
	Detail string
}

// renderMenu draws the menu; the selected line is marked with ">". Each line
// ends so that redrawing over it leaves nothing behind.
func renderMenu(out io.Writer, title string, items []menuItem, sel int) {
	fmt.Fprintf(out, "%s\r\n", title)
	for i, it := range items {
		mark := "  "
		if i == sel {
			mark = "> "
		}
		fmt.Fprintf(out, "%s%d. %-9s %s\x1b[K\r\n", mark, i+1, it.Label, it.Detail)
	}
}

// chooseFromMenu shows a menu on the terminal in and out, and returns the index
// chosen, or -1 when the person left. The terminal is put back as it was on the
// way out, whatever the way.
func chooseFromMenu(in *os.File, out io.Writer, title string, items []menuItem, sel int) (int, error) {
	st, err := term.MakeRaw(int(in.Fd()))
	if err != nil {
		return -1, err
	}
	defer func() { _ = term.Restore(int(in.Fd()), st) }()
	enableVT(out)
	r := bufio.NewReader(in)
	renderMenu(out, title, items, sel)
	for {
		k, err := readKeys(r)
		if err != nil {
			return -1, nil
		}
		switch {
		case k == keyEnter:
			return sel, nil
		case k == keyQuit:
			return -1, nil
		case k == keyUp:
			sel = (sel + len(items) - 1) % len(items)
		case k == keyDown:
			sel = (sel + 1) % len(items)
		case k >= keyDigit1 && int(k-keyDigit1) < len(items):
			return int(k - keyDigit1), nil
		default:
			continue
		}
		// Back over the menu and draw it again.
		fmt.Fprintf(out, "\x1b[%dA", len(items)+1)
		renderMenu(out, title, items, sel)
	}
}

// askKey asks a yes or no question and waits for one key; Enter is the default.
func askKey(in *os.File, out io.Writer, prompt string, def bool) bool {
	st, err := term.MakeRaw(int(in.Fd()))
	if err != nil {
		return def
	}
	defer func() { _ = term.Restore(int(in.Fd()), st) }()
	fmt.Fprint(out, prompt)
	r := bufio.NewReader(in)
	for {
		k, err := readKeys(r)
		if err != nil {
			return def
		}
		switch k {
		case keyYes:
			fmt.Fprint(out, "y\r\n")
			return true
		case keyNo, keyQuit:
			fmt.Fprint(out, "n\r\n")
			return false
		case keyEnter:
			fmt.Fprint(out, "\r\n")
			return def
		}
	}
}

// wrap breaks text at word boundaries into lines of at most width characters.
func wrap(text string, width int) []string {
	var lines []string
	line := ""
	for _, w := range strings.Fields(text) {
		if line != "" && len(line)+1+len(w) > width {
			lines = append(lines, line)
			line = ""
		}
		if line != "" {
			line += " "
		}
		line += w
	}
	if line != "" {
		lines = append(lines, line)
	}
	return lines
}
