// Package e2e starts a real benchd on a temporary database and talks to it
// the way NanoLLM's client does (internal/bench/submit.go): sign the
// nanollm.bench/v1 document with the install key, POST it, read {id, url};
// DELETE with a signed withdrawal. The root module cannot import the
// service, nor the service the root module, so the client's few lines are
// mirrored here against the shared benchfmt; CI additionally runs the real
// `nanollm bench submit` against benchd (.github/workflows/bench-service.yml).
package e2e

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
)

type receipt struct {
	ID  string `json:"id"`
	URL string `json:"url"`
}

// submit is internal/bench.Client.Submit: sign, validate, POST.
func submit(base string, res *benchfmt.Result, key ed25519.PrivateKey) (*receipt, int, string, error) {
	if err := res.Sign(key); err != nil {
		return nil, 0, "", err
	}
	if err := res.Validate(); err != nil {
		return nil, 0, "", err
	}
	body, _ := json.Marshal(res)
	req, _ := http.NewRequest(http.MethodPost, strings.TrimSuffix(base, "/")+"/api/v1/submissions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, "", err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode/100 != 2 {
		return nil, resp.StatusCode, string(b), nil
	}
	var r receipt
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, resp.StatusCode, string(b), err
	}
	return &r, resp.StatusCode, string(b), nil
}

// withdraw is internal/bench.Client.Withdraw.
func withdraw(base, id string, key ed25519.PrivateKey) (int, error) {
	body, _ := json.Marshal(map[string]string{
		"install_id": hex.EncodeToString(key.Public().(ed25519.PublicKey)),
		"signature":  hex.EncodeToString(ed25519.Sign(key, benchfmt.WithdrawalMessage(id))),
	})
	req, _ := http.NewRequest(http.MethodDelete, strings.TrimSuffix(base, "/")+"/api/v1/submissions/"+id, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return 0, err
	}
	resp.Body.Close()
	return resp.StatusCode, nil
}

func freeAddr(t *testing.T) string {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().String()
}

func build(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("BENCHD_BIN"); p != "" {
		return p
	}
	bin := filepath.Join(t.TempDir(), "benchd")
	cmd := exec.Command("go", "build", "-o", bin, "../cmd/benchd")
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
	return bin
}

func run(t *testing.T, bin string, envs []string, args ...string) string {
	t.Helper()
	cmd := exec.Command(bin, args...)
	cmd.Env = append(os.Environ(), envs...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("benchd %v: %v\n%s", args, err, out)
	}
	return string(out)
}

func get(t *testing.T, url string) (int, string, http.Header) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	req.Header.Set("Accept-Language", "en")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b), resp.Header
}

func loadSample(t *testing.T) *benchfmt.Result {
	t.Helper()
	b, err := os.ReadFile("../testdata/sample-result.json")
	if err != nil {
		t.Fatal(err)
	}
	r, err := benchfmt.Parse(b)
	if err != nil {
		t.Fatalf("testdata/sample-result.json is no longer a valid v1 document: %v", err)
	}
	return r
}

func TestEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("builds and starts benchd")
	}
	bin := build(t)
	dir := t.TempDir()
	addr := freeAddr(t)
	base := "http://" + addr
	envs := []string{
		"BENCHD_ADDR=" + addr, "BENCHD_DB=" + filepath.Join(dir, "bench.db"), "BENCHD_PUBLIC_URL=" + base,
		"BENCHD_ADMIN_TOKEN=e2e-token", "BENCHD_RETIRED_SUITES=standard/v0", "BENCHD_BACKUP_DIR=" + filepath.Join(dir, "backups"),
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	srv := exec.CommandContext(ctx, bin, "serve")
	srv.Env = append(os.Environ(), envs...)
	var logs bytes.Buffer
	srv.Stdout, srv.Stderr = &logs, &logs
	if err := srv.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		cancel()
		_ = srv.Wait()
		if t.Failed() {
			t.Logf("benchd log:\n%s", logs.String())
		}
	}()
	up := false
	for range 100 {
		if code, _, _ := tryGet(base + "/healthz"); code == http.StatusOK {
			up = true
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !up {
		t.Fatal("benchd did not come up")
	}
	run(t, bin, envs, "healthcheck")

	_, key, _ := ed25519.GenerateKey(nil)
	rc, code, body, err := submit(base, loadSample(t), key)
	if err != nil || rc == nil {
		t.Fatalf("submit: %d %s %v", code, body, err)
	}
	if code != http.StatusCreated || rc.URL != base+"/benchmarks/s/"+rc.ID {
		t.Fatalf("receipt %d %+v", code, rc)
	}

	// The result is visible on the pages and in the API.
	for path, want := range map[string]string{
		"/benchmarks":                                       "ASUS ROG Zephyrus M16",
		"/benchmarks/s/" + rc.ID:                            "Intel Core i7-12700H",
		"/benchmarks/systems/asus-rog-zephyrus-m16":         "vram-8g",
		"/benchmarks/gpu/nvidia-geforce-rtx-3070-ti-laptop": "58.2",
		"/api/v1/results?system=asus-rog-zephyrus-m16":      `"installs": 1`,
		"/api/v1/facets":                                    "intel-core-i7-12700h",
		"/api/v1/submissions/" + rc.ID:                      "gemma-4-e4b-qat",
	} {
		code, body, _ := get(t, base+path)
		if code != http.StatusOK || !strings.Contains(body, want) {
			t.Errorf("GET %s: %d, want %q in:\n%.2000s", path, code, want, body)
		}
	}
	_, csv, h := get(t, base+"/api/v1/export?format=csv")
	if h.Get("X-Data-License") != "CC-BY-4.0" || !strings.Contains(csv, "ASUS ROG Zephyrus M16") {
		t.Errorf("export: %v\n%s", h, csv)
	}

	// Retired and unknown suite versions, as the client sees them.
	for version, want := range map[string]int{"v0": http.StatusGone, "v7": http.StatusUpgradeRequired} {
		r := loadSample(t)
		r.Suite.Version = version
		_, code, body, _ := submit(base, r, key)
		if code != want {
			t.Errorf("suite %s: %d %s, want %d", version, code, body, want)
		}
	}

	// The CLI commands against the same database while it serves.
	if out := run(t, bin, envs, "reindex"); !strings.Contains(out, "reindexed 1") {
		t.Errorf("reindex: %s", out)
	}
	if out := run(t, bin, envs, "backup"); !strings.Contains(out, "backup:") {
		t.Errorf("backup: %s", out)
	}

	// Admin.
	req, _ := http.NewRequest(http.MethodGet, base+"/admin/api/queue?state=published", nil)
	req.Header.Set("Authorization", "Bearer e2e-token")
	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("admin: %v %v", err, resp)
	}
	resp.Body.Close()

	// Withdraw, as `nanollm bench withdraw` does.
	if code, err := withdraw(base, rc.ID, key); err != nil || code != http.StatusOK {
		t.Fatalf("withdraw: %d %v", code, err)
	}
	if code, _, _ := get(t, base+"/benchmarks/s/"+rc.ID); code != http.StatusGone {
		t.Errorf("withdrawn page: %d", code)
	}
	if _, body, _ := get(t, base+"/api/v1/results"); strings.Contains(body, "Zephyrus") {
		t.Errorf("withdrawn result still aggregated: %s", body)
	}
}

func tryGet(url string) (int, string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b), nil
}
