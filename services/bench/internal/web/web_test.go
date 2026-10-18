package web

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/services/bench/internal/ingest"
	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

func sample() *benchfmt.Result {
	sha := strings.Repeat("ab", 32)
	return &benchfmt.Result{
		Schema: benchfmt.Schema, ConsentVersion: "2026-10",
		Suite:  benchfmt.Suite{ID: "standard", Version: "v1", PromptsSHA256: ingest.StandardV1Prompts},
		Client: benchfmt.Client{NanoLLM: "v1.0.0", LlamaCpp: benchfmt.LlamaCpp{Tag: "b11600"}, OS: "linux", Arch: "amd64"},
		Model:  benchfmt.Model{ID: "gemma-4-e4b-qat", SHA256: sha},
		Machine: benchfmt.Machine{
			CPU:    benchfmt.CPU{Vendor: "AuthenticAMD", Model: "AMD EPYC 9454P 48-Core Processor", Cores: 48, Threads: 96},
			Memory: benchfmt.Memory{TotalBytes: 384 << 30},
			System: benchfmt.System{Vendor: "Supermicro", Product: "AS -1015SV-TNRT", FormFactor: benchfmt.FormServer},
			OS:     benchfmt.OSInfo{Name: "linux", Version: "Ubuntu 24.04", KernelMM: "6.8"},
		},
		Runs: []benchfmt.Run{{Preset: "ram-6g", Plan: benchfmt.Plan{Mode: "ram-full", Device: "CPU"}, Reps: 3, Status: "ok",
			Metrics: benchfmt.Metrics{PPTokS: 300, TGTokS: benchfmt.TG{D0: 35}}, Output: benchfmt.Output{SHA256: sha, Tokens: 128}}},
		Date: "2026-10-18",
	}
}

type env struct {
	srv *httptest.Server
	key ed25519.PrivateKey
	st  *store.Store
}

func setup(t *testing.T) *env {
	t.Helper()
	st, err := store.Open(context.Background(), filepath.Join(t.TempDir(), "b.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	perIP := ingest.NewLimiter(100, time.Hour)
	ing := ingest.New(st, ingest.Config{PublicURL: "https://bench.test", PerIP: perIP, Retired: map[string]bool{"standard/v0": true}})
	s, err := New(Config{PublicURL: "https://bench.test", BackupRetentionDays: 30}, st, ing, perIP, nil)
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	s.Routes(mux)
	srv := httptest.NewServer(Headers(mux))
	t.Cleanup(srv.Close)
	_, key, _ := ed25519.GenerateKey(nil)
	return &env{srv, key, st}
}

func (e *env) post(t *testing.T, r *benchfmt.Result) (*http.Response, map[string]any) {
	t.Helper()
	if err := r.Sign(e.key); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(r)
	resp, err := http.Post(e.srv.URL+"/api/v1/submissions", "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var m map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&m)
	return resp, m
}

func get(t *testing.T, url string, header ...string) (*http.Response, string) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodGet, url, nil)
	for i := 0; i+1 < len(header); i += 2 {
		req.Header.Set(header[i], header[i+1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, string(b)
}

func TestAPIAndPages(t *testing.T) {
	e := setup(t)
	resp, m := e.post(t, sample())
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("%d %v", resp.StatusCode, m)
	}
	id := m["id"].(string)
	if m["url"] != "https://bench.test/benchmarks/s/"+id {
		t.Fatal(m)
	}

	_, body := get(t, e.srv.URL+"/api/v1/results?cpu=amd-epyc-9454p")
	var res struct {
		Suite  string
		Groups []store.Group
	}
	if err := json.Unmarshal([]byte(body), &res); err != nil || len(res.Groups) != 1 || res.Suite != "standard/v1" {
		t.Fatalf("%s %v", body, err)
	}
	if g := res.Groups[0]; g.Metrics["tg_d0"].Median != 35 || g.Installs != 1 || g.System != "Supermicro AS -1015SV-TNRT" {
		t.Fatalf("%+v", g)
	}
	_, body = get(t, e.srv.URL+"/api/v1/results?cpu=nothing")
	if !strings.Contains(body, `"groups": []`) {
		t.Fatal(body)
	}

	_, body = get(t, e.srv.URL+"/api/v1/facets")
	if !strings.Contains(body, `"amd-epyc-9454p"`) || !strings.Contains(body, `"server"`) {
		t.Fatal(body)
	}

	_, body = get(t, e.srv.URL+"/api/v1/submissions/"+id)
	if strings.Contains(body, "install_id") || strings.Contains(body, "signature") || !strings.Contains(body, "EPYC") {
		t.Fatalf("the public submission must not carry the install key: %s", body)
	}

	for _, c := range []struct{ path, want string }{
		{"/benchmarks", "AMD EPYC 9454P"},
		{"/benchmarks/", "AMD EPYC 9454P"},
		{"/benchmarks/cpu/amd-epyc-9454p", "Supermicro"},
		{"/benchmarks/systems/" + res.Groups[0].SystemSlug, "ram-6g"},
		{"/benchmarks/s/" + id, "Ubuntu 24.04"},
		{"/benchmarks/how-to-run", "nanollm bench withdraw"},
		{"/benchmarks/privacy", "pending legal review"},
		{"/benchmarks/static/style.css", "--accent"},
	} {
		resp, body := get(t, e.srv.URL+c.path, "Accept-Language", "en-GB,en;q=0.9")
		if resp.StatusCode != http.StatusOK || !strings.Contains(body, c.want) {
			t.Errorf("%s: %d, want %q in\n%s", c.path, resp.StatusCode, c.want, body)
		}
	}
	resp, body = get(t, e.srv.URL+"/benchmarks?lang=ru")
	if !strings.Contains(body, "Модель ноутбука/ПК") || !strings.Contains(resp.Header.Get("Set-Cookie"), "lang=ru") {
		t.Error("Russian page")
	}
	if !strings.Contains(resp.Header.Get("Content-Security-Policy"), "default-src 'self'") {
		t.Error("no CSP")
	}
	_, body = get(t, e.srv.URL+"/benchmarks/privacy", "Accept-Language", "ru")
	if !strings.Contains(body, "Черновик") || !strings.Contains(body, "30 дней") {
		t.Error("Russian privacy draft")
	}
	if resp, _ := get(t, e.srv.URL+"/benchmarks/gpu/nothing"); resp.StatusCode != http.StatusNotFound {
		t.Error(resp.StatusCode)
	}
}

func TestExport(t *testing.T) {
	e := setup(t)
	e.post(t, sample())
	resp, body := get(t, e.srv.URL+"/api/v1/export?format=csv")
	if resp.Header.Get("X-Data-License") != "CC-BY-4.0" || !strings.Contains(resp.Header.Get("Link"), "creativecommons.org/licenses/by/4.0") {
		t.Fatal(resp.Header)
	}
	lines := strings.Split(body, "\n")
	if !strings.HasPrefix(lines[0], "# NanoLLM public benchmark") || !strings.Contains(lines[0], "CC-BY-4.0") || !strings.HasPrefix(lines[1], "model,suite,preset") {
		t.Fatal(body)
	}
	_, body = get(t, e.srv.URL+"/api/v1/export?format=json&what=runs")
	var out struct {
		License string
		Rows    []store.SubmissionView
	}
	if err := json.Unmarshal([]byte(body), &out); err != nil || out.License != "CC-BY-4.0" || len(out.Rows) != 1 || len(out.Rows[0].Runs) != 1 {
		t.Fatal(body, err)
	}
	if strings.Contains(body, "install") || strings.Contains(body, "fingerprint") {
		t.Fatal("the export must not carry install keys or fingerprints")
	}
	if resp, _ := get(t, e.srv.URL+"/api/v1/export?format=xml"); resp.StatusCode != http.StatusBadRequest {
		t.Fatal(resp.StatusCode)
	}
}

func TestSubmitErrorsAndWithdraw(t *testing.T) {
	e := setup(t)
	r := sample()
	r.Suite.Version = "v0"
	if resp, _ := e.post(t, r); resp.StatusCode != http.StatusGone {
		t.Fatal(resp.StatusCode)
	}
	r = sample()
	r.Suite.Version = "v2"
	if resp, _ := e.post(t, r); resp.StatusCode != http.StatusUpgradeRequired {
		t.Fatal(resp.StatusCode)
	}
	big := bytes.Repeat([]byte("x"), ingest.MaxBody+10)
	resp, err := http.Post(e.srv.URL+"/api/v1/submissions", "application/json", bytes.NewReader(big))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatal(resp.StatusCode)
	}

	resp, m := e.post(t, sample())
	if resp.StatusCode != http.StatusCreated {
		t.Fatal(resp.StatusCode, m)
	}
	id := m["id"].(string)
	body, _ := json.Marshal(map[string]string{
		"install_id": hex.EncodeToString(e.key.Public().(ed25519.PublicKey)),
		"signature":  hex.EncodeToString(ed25519.Sign(e.key, benchfmt.WithdrawalMessage(id))),
	})
	req, _ := http.NewRequest(http.MethodDelete, e.srv.URL+"/api/v1/submissions/"+id, bytes.NewReader(body))
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatal(resp.StatusCode)
	}
	if resp, _ := get(t, e.srv.URL+"/api/v1/submissions/"+id); resp.StatusCode != http.StatusGone {
		t.Fatal(resp.StatusCode)
	}
	if resp, body := get(t, e.srv.URL+"/benchmarks/s/"+id, "Accept-Language", "en"); resp.StatusCode != http.StatusGone || !strings.Contains(body, "withdrawn") {
		t.Fatal(resp.StatusCode)
	}
}

func TestClientIP(t *testing.T) {
	s := &Server{cfg: Config{TrustProxy: true}}
	r := httptest.NewRequest(http.MethodGet, "/", nil)
	r.RemoteAddr = "10.0.0.2:5555"
	r.Header.Set("X-Forwarded-For", "1.1.1.1, 203.0.113.9")
	if got := s.clientIP(r); got != "203.0.113.9" {
		t.Fatal(got)
	}
	s.cfg.TrustProxy = false
	if got := s.clientIP(r); got != "10.0.0.2" {
		t.Fatal(got)
	}
}
