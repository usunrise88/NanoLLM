package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/tune"
)

func TestServiceAddr(t *testing.T) {
	for in, want := range map[string]string{
		"127.0.0.1:8080": "127.0.0.1:8080",
		"0.0.0.0:8080":   "127.0.0.1:8080",
		":8080":          "127.0.0.1:8080",
		"[::]:9":         "127.0.0.1:9",
		"10.0.0.5:80":    "10.0.0.5:80",
	} {
		if got := serviceAddr(in); got != want {
			t.Errorf("%q gives %q, want %q", in, got, want)
		}
	}
}

func TestAdminKey(t *testing.T) {
	cfg := config.Default()
	cfg.Auth.Mode = "apikey"
	cfg.Auth.Keys = []config.APIKey{{Name: "user", Secret: "sk-user"}, {Name: "admin", Secret: "sk-admin", Admin: true}}
	if k, err := adminKey(cfg, ""); err != nil || k != "sk-admin" {
		t.Fatalf("%q %v", k, err)
	}
	if k, _ := adminKey(cfg, "sk-flag"); k != "sk-flag" {
		t.Fatalf("a flag must win, got %q", k)
	}
	cfg.Auth.Keys = []config.APIKey{{Name: "admin", Secret: "sha256:abcd", Admin: true}}
	if _, err := adminKey(cfg, ""); err == nil || !strings.Contains(err.Error(), "hashes") {
		t.Fatalf("only hashed keys: %v", err)
	}
	cfg.Auth.Mode = "open"
	if k, err := adminKey(cfg, ""); err != nil || k != "" {
		t.Fatalf("open mode needs no key: %q %v", k, err)
	}
}

func TestParseSSE(t *testing.T) {
	in := ": keep-alive\n\nevent: probe\ndata: {\"a\":1}\n\nevent: done\ndata: {\"x\":\ndata: 2}\n\nevent: after\ndata: 1\n\n"
	var got []string
	err := parseSSE(strings.NewReader(in), func(name string, data []byte) bool {
		got = append(got, name+"="+string(data))
		return name != "done"
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, "|") != "probe={\"a\":1}|done={\"x\":\n2}" {
		t.Fatalf("%q", got)
	}
}

// The client against a server that answers like the real one.
func TestOptimizeViaService(t *testing.T) {
	var applied bool
	var sawAuth string
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"status":"ok"}`)) })
	mux.HandleFunc("POST /api/v1/optimize", func(w http.ResponseWriter, r *http.Request) {
		sawAuth = r.Header.Get("Authorization")
		var body struct{ Mode string }
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body.Mode != "optimal" {
			http.Error(w, "bad", http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusAccepted)
		w.Write([]byte(`{"id":"r1","estimate_s":45}`))
	})
	mux.HandleFunc("GET /api/v1/optimize/r1/events", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("event: probe_start\ndata: {\"type\":\"probe_start\",\"message\":\"loading x\"}\n\nevent: done\ndata: {}\n\n"))
	})
	mux.HandleFunc("GET /api/v1/optimize/r1", func(w http.ResponseWriter, _ *http.Request) {
		p := tune.Profile{Schema: tune.ProfileSchema, Mode: tune.Optimal, Key: tune.Key{Model: "m"}, Chosen: tune.Chosen{NCtx: 8192}}
		json.NewEncoder(w).Encode(map[string]any{"id": "r1", "state": "done", "profile": p})
	})
	mux.HandleFunc("POST /api/v1/optimize/r1/apply", func(w http.ResponseWriter, _ *http.Request) {
		applied = true
		w.Write([]byte(`{"profile":"optimal@abc","config_written":true,"reloaded":true}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := &optClient{base: srv.URL, key: "sk-admin", http: srv.Client()}
	if err := optimizeViaService(context.Background(), c, tune.Optimal, "", false, true, true, false); err != nil {
		t.Fatal(err)
	}
	if !applied || sawAuth != "Bearer sk-admin" {
		t.Fatalf("applied %v, authorization %q", applied, sawAuth)
	}
	// Without --yes nothing is applied.
	applied = false
	if err := optimizeViaService(context.Background(), c, tune.Optimal, "", false, true, false, false); err != nil {
		t.Fatal(err)
	}
	if applied {
		t.Fatal("applied without being asked")
	}
	// An error from the server reaches the person with its text.
	err := optimizeViaService(context.Background(), c, tune.Silent, "", false, true, true, false)
	if err == nil || !strings.Contains(err.Error(), "bad") {
		t.Fatalf("%v", err)
	}
	// detectService finds the server by /healthz.
	cfg := config.Default()
	cfg.Server.Addr = strings.TrimPrefix(srv.URL, "http://")
	if _, ok := detectService(cfg, ""); !ok {
		t.Fatal("a running server was not found")
	}
	cfg.Server.Addr = "127.0.0.1:1"
	if _, ok := detectService(cfg, ""); ok {
		t.Fatal("found a server where there is none")
	}
}
