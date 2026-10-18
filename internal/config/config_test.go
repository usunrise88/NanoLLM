package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func noEnv(string) (string, bool) { return "", false }

func envOf(m map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) { v, ok := m[k]; return v, ok }
}

func TestDefaultValidates(t *testing.T) {
	c := Default()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestExampleConfigLoads(t *testing.T) {
	cfg, err := Load(filepath.Join("..", "..", "configs", "nanollm.example.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	o := cfg.Models.Override("gemma-4-e4b-qat")
	if o.Preset != "vram-3g" || o.Speculative.Type != SpecMTP || o.Speculative.NMax != 2 {
		t.Errorf("override not decoded: %+v", o)
	}
	if !o.NCtx.IsAuto() {
		t.Errorf("n_ctx: got %v, want auto", o.NCtx)
	}
	if len(cfg.Auth.Keys) != 2 || !cfg.Auth.Keys[0].Admin || cfg.Auth.Keys[1].RPS != 10 {
		t.Errorf("keys not decoded: %+v", cfg.Auth.Keys)
	}
}

func TestUnknownKeyIsRefused(t *testing.T) {
	c := Default()
	err := decode([]byte("server:\n  adr: 0.0.0.0:80\n"), &c)
	if err == nil || !strings.Contains(err.Error(), "adr") {
		t.Fatalf("typo accepted: %v", err)
	}
}

func TestEmptyFileKeepsDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.yaml")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.Addr != Default().Server.Addr {
		t.Errorf("addr: %q", cfg.Server.Addr)
	}
}

func TestValidateRefuses(t *testing.T) {
	cases := map[string]func(*Config){
		"open on a routable address": func(c *Config) { c.Server.Addr = "0.0.0.0:8080" },
		"open on every interface":    func(c *Config) { c.Server.Addr = ":8080" },
		"apikey without keys":        func(c *Config) { c.Auth.Mode = "apikey" },
		"key without a secret": func(c *Config) {
			c.Auth.Mode = "apikey"
			c.Auth.Keys = []APIKey{{Name: "a"}}
		},
		"duplicate key names": func(c *Config) {
			c.Auth.Mode = "apikey"
			c.Auth.Keys = []APIKey{{Name: "a", Secret: "x"}, {Name: "a", Secret: "y"}}
		},
		"unknown dialect":   func(c *Config) { c.API.Dialects = []string{"nonesuch"} },
		"unknown backend":   func(c *Config) { c.Backends.Enabled = []string{"vllm"} },
		"bad device":        func(c *Config) { c.Devices.Default = "cuda0" },
		"bad device prefix": func(c *Config) { c.Devices.Default = "tensorrt:0" },
		"unknown preset": func(c *Config) {
			c.Models.Overrides = map[string]ModelOverride{"m": {Preset: "ram-5g"}}
		},
		"unknown mode": func(c *Config) {
			c.Models.Overrides = map[string]ModelOverride{"m": {Mode: "M11"}}
		},
		"bad kv type": func(c *Config) {
			c.Models.Overrides = map[string]ModelOverride{"m": {KVType: "q4_0"}}
		},
		"more preloads than slots in the pool": func(c *Config) {
			c.Models.Preload = []string{"a", "b"}
		},
		"bad overflow policy": func(c *Config) { c.Generation.ContextOverflow = "drop" },
		"nothing to serve": func(c *Config) {
			c.API.Dialects = nil
			c.UI.Enabled = false
		},
	}
	for name, mutate := range cases {
		c := Default()
		mutate(&c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestEnvOverrides(t *testing.T) {
	c := Default()
	err := applyEnv(&c, envOf(map[string]string{
		"NANOLLM_SERVER_ADDR":         "127.0.0.1:9000",
		"NANOLLM_API_DIALECTS":        "openai, native",
		"NANOLLM_DEVICES_FALLBACK":    "false",
		"NANOLLM_DEVICES_CPU_THREADS": "6",
		"NANOLLM_AUTH_KEYS":           "sk-a, sk-b",
		"NANOLLM_UI_ENABLED":          "false",
	}))
	if err != nil {
		t.Fatal(err)
	}
	if c.Server.Addr != "127.0.0.1:9000" || c.Devices.Fallback || c.Devices.CPUThreads != 6 || c.UI.Enabled {
		t.Errorf("scalars not applied: %+v %+v %+v", c.Server, c.Devices, c.UI)
	}
	if len(c.API.Dialects) != 2 || c.API.Dialects[1] != "native" {
		t.Errorf("list not applied: %q", c.API.Dialects)
	}
	if len(c.Auth.Keys) != 2 || c.Auth.Keys[1].Secret != "sk-b" || c.Auth.Keys[0].Admin {
		t.Errorf("keys not applied: %+v", c.Auth.Keys)
	}
}

func TestEnvOverrideRejectsGarbage(t *testing.T) {
	c := Default()
	if err := applyEnv(&c, envOf(map[string]string{"NANOLLM_POOL_MAX_RESIDENT_MODELS": "two"})); err == nil {
		t.Fatal("a non-integer was accepted")
	}
}

func TestEnvLeavesUnsetKeysAlone(t *testing.T) {
	c := Default()
	if err := applyEnv(&c, noEnv); err != nil {
		t.Fatal(err)
	}
	if c.Server.Addr != Default().Server.Addr || len(c.API.Dialects) != 2 {
		t.Errorf("defaults changed: %+v", c)
	}
}

func TestAuto(t *testing.T) {
	var a Auto
	for in, want := range map[string]int{"": 0, "auto": 0, "4096": 4096} {
		if err := a.UnmarshalText([]byte(in)); err != nil || a.N != want {
			t.Errorf("%q: got %d, %v", in, a.N, err)
		}
	}
	for _, in := range []string{"0", "-1", "lots"} {
		if err := a.UnmarshalText([]byte(in)); err == nil {
			t.Errorf("%q accepted", in)
		}
	}
}

func TestNewKeySecret(t *testing.T) {
	a, err := NewKeySecret()
	if err != nil {
		t.Fatal(err)
	}
	b, _ := NewKeySecret()
	if a == b || !strings.HasPrefix(a, KeyPrefix) || len(a) != len(KeyPrefix)+32 {
		t.Errorf("bad secrets %q %q", a, b)
	}
}
