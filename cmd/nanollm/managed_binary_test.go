package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestManagedBinary runs a built binary the way an application does: a child
// process with pipes on its standard streams and, on Windows, no console
// window. CI points it at the archive it has just built on each platform. The
// scripted backend stands in for a model, so the measurement at the start goes
// through the real worker processes in a minute or two.
func TestManagedBinary(t *testing.T) {
	bin := os.Getenv("NANOLLM_MANAGED_BINARY")
	if bin == "" {
		t.Skip("NANOLLM_MANAGED_BINARY names no binary to run")
	}
	dir := t.TempDir()
	cmd := exec.Command(bin, "managed", "-data-dir", dir, "-parent-pid", strconv.Itoa(os.Getpid()), "-test-backend", "fake")
	asApplication(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cmd.Process.Kill() }()

	lines := make(chan map[string]any, 16)
	go func() {
		defer close(lines)
		sc := bufio.NewScanner(stdout)
		for sc.Scan() {
			var ev map[string]any
			if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
				t.Errorf("not a JSON line on stdout: %q", sc.Text())
				continue
			}
			lines <- ev
		}
	}()
	inst := &instance{dir: dir}
	next := func(name string, within time.Duration) map[string]any {
		t.Helper()
		select {
		case ev, ok := <-lines:
			if !ok {
				t.Fatalf("stdout closed; the log ends:\n%s", inst.logTail())
			}
			if ev["event"] != name {
				t.Fatalf("event %v, want %s; the log ends:\n%s", ev, name, inst.logTail())
			}
			return ev
		case <-time.After(within):
			t.Fatalf("no %s within %s; the log ends:\n%s", name, within, inst.logTail())
		}
		return nil
	}

	inst.ready = next("ready", 60*time.Second)
	started := next("tune_started", 30*time.Second)
	if started["reason"] != "missing" || started["mode"] != "silent" {
		t.Fatalf("tune_started: %v", started)
	}
	begin := time.Now()
	done := next("tune_done", 10*time.Minute)
	if !strings.HasPrefix(fmt.Sprint(done["profile"]), "silent@") {
		t.Fatalf("tune_done: %v", done)
	}
	t.Logf("measured in %v: %v", time.Since(begin).Round(time.Second), done)

	st := inst.waitAutotune(t, "ready")
	t.Logf("autotune: %v", autotune(st))
	if code, _ := inst.chat(t); code != http.StatusOK {
		t.Fatalf("a chat request: %d", code)
	}

	stop := time.Now()
	_ = stdin.Close()
	if ev := next("stopping", 30*time.Second); ev["reason"] != "stdin closed" {
		t.Fatalf("after closing stdin: %v", ev)
	}
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	select {
	case err := <-exited:
		if err != nil {
			t.Fatalf("exit: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("still running 30s after stdin closed")
	}
	t.Logf("stopped %v after stdin closed", time.Since(stop).Round(time.Millisecond))
}
