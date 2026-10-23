//go:build windows

package main

import (
	"io"
	"os"

	"golang.org/x/sys/windows"
)

// enableVT turns on escape-sequence processing for the console behind out, so
// that the menu can redraw itself in a console that has it off (the legacy
// console host); Windows Terminal has it on already.
func enableVT(out io.Writer) {
	f, ok := out.(*os.File)
	if !ok {
		return
	}
	h := windows.Handle(f.Fd())
	var mode uint32
	if windows.GetConsoleMode(h, &mode) != nil {
		return
	}
	_ = windows.SetConsoleMode(h, mode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING)
}
