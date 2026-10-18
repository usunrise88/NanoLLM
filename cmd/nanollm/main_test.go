package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/config"
)

func TestParseFlagsInAnyOrder(t *testing.T) {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	preset := fs.String("preset", "", "")
	think := fs.Bool("think", false, "")
	pos, err := parseFlags(fs, []string{"--think", "gemma-4-e4b-qat", "-preset", "ram-4g", "extra"})
	if err != nil {
		t.Fatal(err)
	}
	if *preset != "ram-4g" || !*think {
		t.Errorf("flags: preset=%q think=%v", *preset, *think)
	}
	// The boolean must not have swallowed the model name.
	if want := []string{"gemma-4-e4b-qat", "extra"}; !reflect.DeepEqual(pos, want) {
		t.Errorf("positional: %q, want %q", pos, want)
	}
}

func TestInitWritesALoadableConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nanollm.yaml")
	// A Windows-style data directory: backslashes must survive YAML quoting.
	dataDir := `C:\ProgramData\NanoLLM`

	if err := initCommand([]string{"-config", path, "-data-dir", dataDir, "-no-download"}); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Auth.Mode != "apikey" || len(cfg.Auth.Keys) != 2 || !cfg.Auth.Keys[0].Admin {
		t.Errorf("auth: %+v", cfg.Auth)
	}
	for _, k := range cfg.Auth.Keys {
		if !strings.HasPrefix(k.Secret, config.KeyPrefix) {
			t.Errorf("key %q has secret %q", k.Name, k.Secret)
		}
	}
	if cfg.Server.DataDir != dataDir {
		t.Errorf("data_dir: %q", cfg.Server.DataDir)
	}
	if cfg.Models.Default != config.DefaultModel {
		t.Errorf("model: %q", cfg.Models.Default)
	}

	// Windows has no group/other permission bits to check.
	if runtime.GOOS != "windows" {
		if st, err := os.Stat(path); err != nil || st.Mode().Perm()&0o077 != 0 {
			t.Errorf("the file holding the keys is readable by others: %v", err)
		}
	}
	if err := initCommand([]string{"-config", path, "-no-download"}); err == nil {
		t.Error("an existing configuration was overwritten without -force")
	}
}

func TestInitRefusesAnUnknownModel(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nanollm.yaml")
	if err := initCommand([]string{"-config", path, "-model", "llama-70b"}); err == nil {
		t.Fatal("a model outside the catalog was accepted")
	}
	if _, err := os.Stat(path); err == nil {
		t.Error("a configuration was written anyway")
	}
}

func TestRedactedConfigHidesSecrets(t *testing.T) {
	cfg := config.Default()
	cfg.Auth.Keys = []config.APIKey{{Name: "admin", Secret: "sk-nanollm-secret"}}
	out, err := redactedConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(fmt.Sprint(out), "sk-nanollm-secret") {
		t.Fatal("the snapshot carries a secret")
	}
	if cfg.Auth.Keys[0].Secret != "sk-nanollm-secret" {
		t.Error("redaction modified the live configuration")
	}
}
