package benchfmt

import (
	"bytes"
	"crypto/ed25519"
	"encoding/json"
	"flag"
	"os"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "rewrite schema.json from the Go types")

func sample() Result {
	sha := strings.Repeat("ab", 32)
	return Result{
		Schema: Schema, ConsentVersion: "2026-10",
		Suite:  Suite{ID: "standard", Version: "v1", PromptsSHA256: sha},
		Client: Client{NanoLLM: "v1.0.0", LlamaCpp: LlamaCpp{Tag: "b11600", Commit: "ace20f6"}, GGMLCPUVariant: "alderlake", OS: "windows", Arch: "amd64"},
		Model:  Model{ID: "gemma-4-e4b-qat", Revision: "8c5a9e4", SHA256: sha, DrafterSHA256: sha},
		Machine: Machine{
			CPU:    CPU{Vendor: "Intel", Model: "Core i7-12700H", Cores: 14, Threads: 20, PCores: 6, ECores: 8, ISA: []string{"avx2", "avx_vnni"}},
			Memory: Memory{TotalBytes: 32 << 30},
			GPUs:   []GPU{{Name: "GeForce RTX 3070 Ti Laptop GPU", Vendor: "NVIDIA", VRAMBytes: 8 << 30, DriverMajor: "560"}},
			System: System{Vendor: "ASUS", Family: "ROG Zephyrus", FormFactor: FormLaptop},
			OS:     OSInfo{Name: "windows", Version: "11"},
			Power:  Power{Source: "ac"},
		},
		Runs: []Run{
			{Preset: "vram-8g", Plan: Plan{Mode: "vram-full", GPULayers: -1, NCtx: 8192, FA: true, MTP: true}, Reps: 3, Status: StatusOK,
				Metrics: Metrics{PPTokS: 1500, TGTokS: TG{D0: 60, DN: 52, DepthN: 4096}, MTP: &MTP{Acceptance: 0.7, Speedup: 1.4}},
				Output:  Output{SHA256: sha, Tokens: 128}},
			{Preset: "ram-1g", Status: StatusRefused, Reason: "weights need 2.9 GiB"},
		},
		Date: "2026-10-18", DurationS: 600,
	}
}

func TestValidateAndParse(t *testing.T) {
	r := sample()
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(r)
	if _, err := Parse(b); err != nil {
		t.Fatal(err)
	}
	bad := sample()
	bad.Schema = "nanollm.bench/v0"
	bad.Date = "2026-10-18T10:00:00Z"
	bad.Runs[0].Metrics.TGTokS.D0 = -1
	bad.Machine.System.FormFactor = "phone"
	bad.Machine.CPU.Model = "x\ny"
	err := bad.Validate()
	for _, want := range []string{"schema", "date", "tg_tok_s.d0", "form_factor", "cpu.model"} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("expected a complaint about %s, got %v", want, err)
		}
	}
}

func TestUnknownFieldsParse(t *testing.T) {
	b, _ := json.Marshal(sample())
	b = bytes.Replace(b, []byte(`"date"`), []byte(`"future_field":{"x":1},"date"`), 1)
	if _, err := Parse(b); err != nil {
		t.Fatalf("an additive field must not break v1 readers: %v", err)
	}
}

func TestSignVerify(t *testing.T) {
	_, key, _ := ed25519.GenerateKey(nil)
	r := sample()
	if err := r.Sign(key); err != nil {
		t.Fatal(err)
	}
	if err := r.Verify(); err != nil {
		t.Fatal(err)
	}
	// Round trip through JSON keeps the signature valid.
	b, _ := json.Marshal(r)
	back, _ := Parse(b)
	if err := back.Verify(); err != nil {
		t.Fatal(err)
	}
	back.Runs[0].Metrics.TGTokS.D0 = 99
	if back.Verify() == nil {
		t.Fatal("tampering not detected")
	}
}

func TestSchemaIsCurrent(t *testing.T) {
	got, err := GenerateSchema()
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	if *update {
		if err := os.WriteFile("schema.json", got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	if !bytes.Equal(got, SchemaJSON) {
		t.Fatal("schema.json is stale: go test ./ -update, and review the diff — v1 changes must be additive")
	}
}
