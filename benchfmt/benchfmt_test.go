package benchfmt

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
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

func TestOutputPrefixes(t *testing.T) {
	r := sample()
	r.Runs[0].Output.Prefixes = []Prefix{{Tokens: 16, SHA256: strings.Repeat("cd", 32)}}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	r.Runs[0].Output.Prefixes = append(r.Runs[0].Output.Prefixes, Prefix{Tokens: 0, SHA256: "x"})
	if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "prefixes[1]") {
		t.Fatalf("a malformed prefix must be refused, got %v", err)
	}
}

// The signature covers the document, not the Go struct: a member this version
// does not know is signed too, so an older service verifies a newer client's
// result, and cannot be fooled by a changed unknown member.
func TestVerifyJSONCoversUnknownMembers(t *testing.T) {
	_, key, _ := ed25519.GenerateKey(nil)
	r := sample()
	// The newer client has a member this struct lacks.
	b, _ := json.Marshal(r)
	b = bytes.Replace(b, []byte(`"date"`), []byte(`"future_field":{"x":1.50,"y":"<a&b>"},"date"`), 1)
	signed := signDoc(t, key, b)
	if err := VerifyJSON(signed); err != nil {
		t.Fatalf("a document with a member this version does not know: %v", err)
	}
	// The struct this version reads from it would re-serialize without the
	// member, which the old scheme signed over; the new one does not care.
	if _, err := Parse(signed); err != nil {
		t.Fatal(err)
	}
	// Changing the unknown member breaks the signature.
	bad := bytes.Replace(signed, []byte(`"x":1.50`), []byte(`"x":1.5`), 1)
	if VerifyJSON(bad) == nil {
		t.Fatal("a changed unknown member was not detected")
	}
	// White space and key order do not matter.
	var m map[string]json.RawMessage
	if err := json.Unmarshal(signed, &m); err != nil {
		t.Fatal(err)
	}
	pretty, _ := json.MarshalIndent(m, "", "  ")
	if err := VerifyJSON(pretty); err != nil {
		t.Fatalf("re-formatted document: %v", err)
	}
}

// signDoc signs a document the way Sign does, for a document that has members
// the struct does not.
func signDoc(t *testing.T, key ed25519.PrivateKey, doc []byte) []byte {
	t.Helper()
	pub := hex.EncodeToString(key.Public().(ed25519.PublicKey))
	var m map[string]json.RawMessage
	if err := json.Unmarshal(doc, &m); err != nil {
		t.Fatal(err)
	}
	m["install_id"], _ = json.Marshal(pub)
	delete(m, "signature")
	unsigned, _ := json.Marshal(m)
	c, err := Canonical(unsigned)
	if err != nil {
		t.Fatal(err)
	}
	m["signature"], _ = json.Marshal(hex.EncodeToString(ed25519.Sign(key, c)))
	out, _ := json.Marshal(m)
	return out
}
