package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/plan"
)

func TestPlanCommand(t *testing.T) {
	var out bytes.Buffer
	if err := planCommand([]string{"gemma-4-e4b-qat", "--preset", "vram-4g"}, &out); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"M2 vram-ple-host", "llamacpp:cuda0", "tried", "not budgeted"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}

	out.Reset()
	err := planCommand([]string{"gemma-4-e4b-qat", "--preset", "vram-8g", "--device", "llamacpp:vulkan0",
		"--device-memory", "3g", "--json"}, &out)
	if err != nil {
		t.Fatal(err)
	}
	var p plan.Plan
	if err := json.Unmarshal(out.Bytes(), &p); err != nil {
		t.Fatal(err)
	}
	if p.Device == nil || p.Device.Backend != plan.BackendVulkan || p.Memory.VRAM.Budget != 3<<30 {
		t.Errorf("device %+v, memory %+v", p.Device, p.Memory)
	}
}

func TestPlanCommandRefuses(t *testing.T) {
	var out bytes.Buffer
	if err := planCommand([]string{"gemma-4-e4b-qat", "--preset", "ram-5g"}, &out); err == nil {
		t.Error("an undefined preset was planned")
	}
	if err := planCommand([]string{"gemma-4-e4b-qat"}, &out); err == nil {
		t.Error("planned without a preset")
	}
	// A context nothing can hold: the plan prints, and the command fails.
	out.Reset()
	err := planCommand([]string{"gemma-4-e4b-qat", "--preset", "ram-1g", "--n-ctx", "131072", "--slots", "8"}, &out)
	if err == nil || !strings.Contains(out.String(), "refused") {
		t.Errorf("err %v, output:\n%s", err, out.String())
	}
}

func TestPlanCommandOnFile(t *testing.T) {
	p := writeTiny(t, gguf.TinyGemma4())
	var out bytes.Buffer
	if err := planCommand([]string{p, "--preset", "ram-1g", "--json"}, &out); err != nil {
		t.Fatal(err)
	}
	var pl plan.Plan
	if err := json.Unmarshal(out.Bytes(), &pl); err != nil {
		t.Fatal(err)
	}
	if pl.Mode != plan.RAMFull || pl.MTP.Enabled {
		t.Errorf("tiny model: %s, MTP %v", pl.Mode, pl.MTP.Enabled)
	}
}

func TestParseSize(t *testing.T) {
	for in, want := range map[string]int64{"6g": 6 << 30, "1.5g": 3 << 29, "512m": 512 << 20, "1000": 1000} {
		if got, err := parseSize(in); err != nil || got != want {
			t.Errorf("%s = %d, %v", in, got, err)
		}
	}
	if _, err := parseSize("-1g"); err == nil {
		t.Error("negative size accepted")
	}
}
