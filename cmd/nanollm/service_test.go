package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestServiceFlagAcceptsTheWindowsSpelling(t *testing.T) {
	cases := map[string]string{
		"--install":   "install",
		"-install":    "install",
		"--uninstall": "uninstall",
		"--remove":    "uninstall",
		"--start":     "start",
		"--stop":      "stop",
		"--restart":   "restart",
		"--status":    "status",
	}
	for arg, want := range cases {
		got, ok := serviceFlag(arg)
		if !ok || got != want {
			t.Errorf("serviceFlag(%q) = %q, %v; want %q, true", arg, got, ok, want)
		}
	}
	// A bare verb is the `nanollm service install` form and must not be
	// mistaken for a top-level flag, and an unknown flag has to reach the
	// usage message rather than be read as a service verb.
	for _, arg := range []string{"install", "serve", "--help", "-h", "--addr", "--", "-"} {
		if got, ok := serviceFlag(arg); ok {
			t.Errorf("serviceFlag(%q) = %q, true; want false", arg, got)
		}
	}
}

func TestSplitVerbDefaultsToStatus(t *testing.T) {
	if v, rest := splitVerb(nil); v != "status" || len(rest) != 0 {
		t.Errorf("splitVerb(nil) = %q, %q", v, rest)
	}
	if v, rest := splitVerb([]string{"-name", "x"}); v != "status" || len(rest) != 2 {
		t.Errorf("splitVerb(-name x) = %q, %q", v, rest)
	}
	if v, _ := splitVerb([]string{"remove"}); v != "uninstall" {
		t.Errorf("splitVerb(remove) = %q", v)
	}
}

func TestListVerbsReadsAsASentence(t *testing.T) {
	got := listVerbs()
	if !strings.HasPrefix(got, "install, ") || !strings.Contains(got, " or status") {
		t.Errorf("listVerbs() = %q", got)
	}
}

func TestSystemdUnitLocksMemoryAndQuotesPaths(t *testing.T) {
	text, err := renderSystemd(unitParams{
		Exe: "/opt/nano llm/nanollm", Config: "/var/lib/nanollm/50%.yaml", DataDir: "/var/lib/nanollm", User: "svc",
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"LimitMEMLOCK=infinity\n",
		`ExecStart="/opt/nano llm/nanollm" serve -config /var/lib/nanollm/50%%.yaml` + "\n",
		"User=svc\nGroup=svc\n",
		"ReadWritePaths=/var/lib/nanollm\n",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("unit lacks %q:\n%s", want, text)
		}
	}
	// A GPU needs its device nodes and its driver needs executable memory.
	for _, banned := range []string{"PrivateDevices", "MemoryDenyWriteExecute"} {
		if strings.Contains(text, banned) {
			t.Errorf("unit sets %s, which takes the GPU away", banned)
		}
	}
}

func TestLaunchdPlistEscapesAndLogs(t *testing.T) {
	text, err := renderLaunchd(unitParams{
		Exe: "/usr/local/nanollm/nanollm", Config: "/Users/a&b/Library/Application Support/NanoLLM/nanollm.yaml",
		DataDir: "/Users/a&b/Library/Application Support/NanoLLM", LogFile: "/Users/a&b/Library/Logs/NanoLLM/nanollm.log",
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"<string>" + serviceLabel + "</string>",
		"<string>/Users/a&amp;b/Library/Application Support/NanoLLM/nanollm.yaml</string>",
		"<string>-log-file</string>",
		"<string>/Users/a&amp;b/Library/Logs/NanoLLM/nanollm.stderr.log</string>",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("plist lacks %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "UserName") {
		t.Error("an agent names a user; it runs as whoever loaded it")
	}
}

// The unit files in the archive are the ones `nanollm service install` would
// write with the installers' default paths. NANOLLM_UPDATE_UNITS=1 rewrites
// them after the templates change.
func TestShippedUnitsAreCurrent(t *testing.T) {
	for goos, file := range map[string]string{"linux": "nanollm.service", "darwin": "nanollm.plist"} {
		want, err := unitFor(goos, defaultUnitParams(goos))
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join("..", "..", "deploy", file)
		if os.Getenv("NANOLLM_UPDATE_UNITS") == "1" {
			if err := os.WriteFile(path, []byte(want), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if strings.ReplaceAll(string(got), "\r\n", "\n") != want {
			t.Errorf("deploy/%s is not what the template renders; run NANOLLM_UPDATE_UNITS=1 go test ./cmd/nanollm", file)
		}
	}
}

func TestOpenLogWithoutAPathIsStderr(t *testing.T) {
	w, closeLog, err := openLog("")
	if err != nil {
		t.Fatal(err)
	}
	defer closeLog()
	if w != os.Stderr {
		t.Errorf("openLog(\"\") = %v, want os.Stderr", w)
	}
}

// The directory is created because the service writes to a path nothing else
// has made yet: an install chooses logs/nanollm.log beside the configuration.
func TestOpenLogCreatesTheDirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "logs", "nanollm.log")
	w, closeLog, err := openLog(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("line\n")); err != nil {
		t.Fatal(err)
	}
	closeLog()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != "line\n" {
		t.Errorf("log contains %q", b)
	}
}

func TestRollingFileKeepsOneGeneration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nanollm.log")
	f, err := newRollingFile(path, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	line := []byte("0123456789\n") // 11 bytes
	for range 5 {
		if _, err := f.Write(line); err != nil {
			t.Fatal(err)
		}
	}
	// Five 11-byte lines against a 32-byte limit roll over on the third and
	// again on the fifth: a line is never split across generations.
	prev, err := os.ReadFile(path + ".1")
	if err != nil {
		t.Fatalf("no rotated file: %v", err)
	}
	if len(prev) != 2*len(line) {
		t.Errorf("rotated file is %d bytes, want %d", len(prev), 2*len(line))
	}
	cur, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(cur) != len(line) {
		t.Errorf("current file is %d bytes, want %d", len(cur), len(line))
	}
}

func TestResolveServiceConfigRefusesAMissingFile(t *testing.T) {
	_, _, err := resolveServiceConfig(filepath.Join(t.TempDir(), "nanollm.yaml"), t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "nanollm init") {
		t.Errorf("err = %v; want one that says to run nanollm init", err)
	}
}
