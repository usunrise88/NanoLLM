package admin

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/services/bench/internal/ingest"
	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

func setup(t *testing.T, token string) (*httptest.Server, *ingest.Service, *store.Store) {
	t.Helper()
	st, err := store.Open(context.Background(), filepath.Join(t.TempDir(), "b.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	ing := ingest.New(st, ingest.Config{PublicURL: "https://bench.test"})
	mux := http.NewServeMux()
	New(token, st, ing, nil).Routes(mux)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, ing, st
}

func submit(t *testing.T, ing *ingest.Service, userEntered bool) string {
	t.Helper()
	sha := strings.Repeat("ab", 32)
	r := &benchfmt.Result{
		Schema: benchfmt.Schema,
		Suite:  benchfmt.Suite{ID: "standard", Version: "v1", PromptsSHA256: ingest.StandardV1Prompts},
		Client: benchfmt.Client{NanoLLM: "v1.0.0", LlamaCpp: benchfmt.LlamaCpp{Tag: "b11600"}, OS: "linux", Arch: "amd64"},
		Model:  benchfmt.Model{ID: "gemma-4-e4b-qat", SHA256: sha},
		Machine: benchfmt.Machine{
			CPU:    benchfmt.CPU{Vendor: "AuthenticAMD", Model: "AMD Ryzen 7 7840HS w/ Radeon 780M Graphics", Threads: 16},
			Memory: benchfmt.Memory{TotalBytes: 32 << 30},
			System: benchfmt.System{Vendor: "Framework", Family: "Laptop", Product: "Laptop 16 (AMD Ryzen 7040 Series)", FormFactor: "laptop", UserEntered: userEntered},
			OS:     benchfmt.OSInfo{Name: "linux"},
		},
		Runs: []benchfmt.Run{{Preset: "ram-6g", Plan: benchfmt.Plan{Mode: "ram-full", Device: "CPU"}, Reps: 3, Status: "ok", Metrics: benchfmt.Metrics{TGTokS: benchfmt.TG{D0: 14}}}},
		Date: "2026-10-18",
	}
	_, key, _ := ed25519.GenerateKey(nil)
	if err := r.Sign(key); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(r)
	rc, err := ing.Submit(context.Background(), b, "")
	if err != nil {
		t.Fatal(err)
	}
	return rc.ID
}

func call(t *testing.T, srv *httptest.Server, token, method, path string, body any) (int, map[string]any) {
	t.Helper()
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, _ := http.NewRequest(method, srv.URL+path, rd)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var m map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&m)
	return resp.StatusCode, m
}

func TestAuth(t *testing.T) {
	srv, _, _ := setup(t, "s3cret")
	if code, _ := call(t, srv, "", "GET", "/admin/api/queue", nil); code != http.StatusUnauthorized {
		t.Fatal(code)
	}
	if code, _ := call(t, srv, "wrong", "GET", "/admin/api/queue", nil); code != http.StatusUnauthorized {
		t.Fatal(code)
	}
	if code, _ := call(t, srv, "s3cret", "GET", "/admin/api/queue", nil); code != http.StatusOK {
		t.Fatal(code)
	}
	resp, err := http.Get(srv.URL + "/admin")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatal(err, resp.StatusCode)
	}
	resp.Body.Close()

	off, _, _ := setup(t, "")
	resp, _ = http.Get(off.URL + "/admin")
	if resp.StatusCode != http.StatusNotFound {
		t.Fatal("admin without a token must not exist")
	}
	resp.Body.Close()
}

func TestModerationAliasesVerified(t *testing.T) {
	const tok = "s3cret"
	srv, ing, st := setup(t, tok)
	ctx := context.Background()
	id := submit(t, ing, true) // hand-entered system model → queued
	_, m := call(t, srv, tok, "GET", "/admin/api/queue", nil)
	items := m["items"].([]any)
	if len(items) != 1 || items[0].(map[string]any)["id"] != id {
		t.Fatal(m)
	}
	if gs, _ := store.Results(ctx, st.DB, store.Filter{}); len(gs) != 0 {
		t.Fatal("a queued submission is in the aggregates")
	}
	if code, _ := call(t, srv, tok, "POST", "/admin/api/submissions/"+id+"/state", map[string]string{"state": "published"}); code != http.StatusOK {
		t.Fatal(code)
	}
	gs, _ := store.Results(ctx, st.DB, store.Filter{})
	if len(gs) != 1 || gs[0].System != "Framework Laptop 16 (AMD Ryzen 7040 Series)" {
		t.Fatalf("%+v", gs)
	}
	if code, _ := call(t, srv, tok, "POST", "/admin/api/submissions/"+id+"/state", map[string]string{"state": "bogus"}); code != http.StatusBadRequest {
		t.Fatal(code)
	}

	// An alias reindexes.
	code, m := call(t, srv, tok, "POST", "/admin/api/aliases", map[string]string{"kind": "system", "pattern": "framework laptop 16 (amd ryzen 7040 series)", "name": "Framework Laptop 16"})
	if code != http.StatusOK || m["reindexed"].(float64) != 1 {
		t.Fatal(code, m)
	}
	gs, _ = store.Results(ctx, st.DB, store.Filter{})
	if gs[0].System != "Framework Laptop 16" || gs[0].SystemSlug != "framework-laptop-16" {
		t.Fatalf("%+v", gs[0])
	}
	if code, _ := call(t, srv, tok, "POST", "/admin/api/aliases", map[string]string{"kind": "nope", "pattern": "a", "name": "b"}); code != http.StatusBadRequest {
		t.Fatal(code)
	}

	// The badge, granted by a submission id.
	if code, m := call(t, srv, tok, "POST", "/admin/api/verified", map[string]string{"submission_id": id, "label": "owner"}); code != http.StatusOK {
		t.Fatal(code, m)
	}
	gs, _ = store.Results(ctx, st.DB, store.Filter{})
	if !gs[0].Verified {
		t.Fatal("no badge after verifying the install")
	}
	_, m = call(t, srv, tok, "GET", "/admin/api/submissions/"+id, nil)
	if m["install_id"] == "" || m["raw"] == nil {
		t.Fatal(m)
	}

	// Hide.
	call(t, srv, tok, "POST", "/admin/api/submissions/"+id+"/state", map[string]string{"state": "hidden"})
	if gs, _ := store.Results(ctx, st.DB, store.Filter{}); len(gs) != 0 {
		t.Fatal("a hidden submission is in the aggregates")
	}
}
