package main

import (
	"testing"
	"time"
)

func TestFlagsFromEnv(t *testing.T) {
	t.Setenv("BENCHD_DATA_LICENSE", "CC0-1.0")
	t.Setenv("BENCHD_RETIRED_SUITES", "standard/v0, standard/v00 ,")
	t.Setenv("BENCHD_BACKUP_KEEP", "168h")
	t.Setenv("BENCHD_ADMIN_TOKEN", "tok")
	c, err := flags("serve", []string{"-addr", "127.0.0.1:9"})
	if err != nil {
		t.Fatal(err)
	}
	if c.license != "CC0-1.0" || c.addr != "127.0.0.1:9" || c.backupKeep != 7*24*time.Hour || c.adminToken != "tok" {
		t.Fatalf("%+v", c)
	}
	if r := c.retiredSet(); len(r) != 2 || !r["standard/v0"] || !r["standard/v00"] {
		t.Fatal(r)
	}
}

func TestFlagsDefaultsAndChecks(t *testing.T) {
	c, err := flags("serve", nil)
	if err != nil {
		t.Fatal(err)
	}
	if c.license != "CC-BY-4.0" || c.moderation != "auto" || c.backupKeep != 30*24*time.Hour {
		t.Fatalf("%+v", c)
	}
	if _, err := flags("serve", []string{"-moderation", "lax"}); err == nil {
		t.Fatal("bad moderation accepted")
	}
	if _, err := flags("serve", []string{"-backup-keep", "0s"}); err == nil {
		t.Fatal("unbounded retention accepted")
	}
}
