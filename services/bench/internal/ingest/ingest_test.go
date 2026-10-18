package ingest

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/services/bench/internal/store"
)

func sample(tg float64) *benchfmt.Result {
	sha := strings.Repeat("ab", 32)
	return &benchfmt.Result{
		Schema: benchfmt.Schema, ConsentVersion: "2026-10",
		Suite:  benchfmt.Suite{ID: "standard", Version: "v1", PromptsSHA256: StandardV1Prompts, Quick: true},
		Client: benchfmt.Client{NanoLLM: "v1.0.0", LlamaCpp: benchfmt.LlamaCpp{Tag: "b11600", Commit: "ace20f6"}, GGMLCPUVariant: "alderlake", OS: "windows", Arch: "amd64"},
		Model:  benchfmt.Model{ID: "gemma-4-e4b-qat", Revision: "8c5a9e4", SHA256: sha},
		Machine: benchfmt.Machine{
			CPU:    benchfmt.CPU{Vendor: "GenuineIntel", Model: "12th Gen Intel(R) Core(TM) i7-12700H", Cores: 14, Threads: 20},
			Memory: benchfmt.Memory{TotalBytes: 32 << 30, SpeedMTS: 4800},
			GPUs:   []benchfmt.GPU{{Name: "NVIDIA GeForce RTX 3070 Ti Laptop GPU", Vendor: "NVIDIA", VRAMBytes: 8 << 30, DriverMajor: "560"}},
			System: benchfmt.System{Vendor: "ASUSTeK COMPUTER INC.", Family: "ROG Zephyrus M16", Product: "ROG Zephyrus M16 GU603ZW_GU603ZW", FormFactor: benchfmt.FormLaptop},
			OS:     benchfmt.OSInfo{Name: "windows", Version: "11"},
			Power:  benchfmt.Power{Source: "ac"},
		},
		Runs: []benchfmt.Run{
			{Preset: "vram-8g", Plan: benchfmt.Plan{Mode: "vram-full", Device: "CUDA0", GPULayers: -1, NCtx: 8192, FA: true}, Reps: 3, Status: benchfmt.StatusOK,
				Metrics: benchfmt.Metrics{PPTokS: 1500, TGTokS: benchfmt.TG{D0: tg, DN: tg * 0.9, DepthN: 4096}, ChatTokS: tg},
				Output:  benchfmt.Output{SHA256: sha, Tokens: 128}},
			{Preset: "ram-6g", Plan: benchfmt.Plan{Mode: "ram-full", Device: "CPU", NCtx: 4096}, Reps: 3, Status: benchfmt.StatusOK,
				Metrics: benchfmt.Metrics{PPTokS: 80, TGTokS: benchfmt.TG{D0: 12}}},
			{Preset: "ram-1g", Status: benchfmt.StatusRefused, Reason: "weights need 2.9 GiB"},
		},
		Date: "2026-10-18", DurationS: 600,
	}
}

func signed(t *testing.T, r *benchfmt.Result, key ed25519.PrivateKey) []byte {
	t.Helper()
	if err := r.Sign(key); err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func newKey() ed25519.PrivateKey {
	_, k, _ := ed25519.GenerateKey(nil)
	return k
}

func newService(t *testing.T, cfg Config) (*Service, *store.Store) {
	t.Helper()
	st, err := store.Open(context.Background(), filepath.Join(t.TempDir(), "bench.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	if cfg.PublicURL == "" {
		cfg.PublicURL = "https://bench.test"
	}
	if cfg.Retired == nil {
		cfg.Retired = map[string]bool{"standard/v0": true}
	}
	return New(st, cfg), st
}

func status(err error) int {
	var e *Error
	if errors.As(err, &e) {
		return e.Status
	}
	return 0
}

func TestSubmitAndDedupe(t *testing.T) {
	ctx := context.Background()
	s, st := newService(t, Config{})
	key := newKey()
	body := signed(t, sample(60), key)
	rc, err := s.Submit(ctx, body, "ip")
	if err != nil {
		t.Fatal(err)
	}
	if !ValidID(rc.ID) || rc.URL != "https://bench.test/benchmarks/s/"+rc.ID || rc.State != store.StatePublished {
		t.Fatalf("%+v", rc)
	}
	// The exact bytes are kept.
	sub, _, err := store.GetSubmission(ctx, st.DB, rc.ID)
	if err != nil || string(sub.Raw) != string(body) {
		t.Fatal("raw bytes not kept as received", err)
	}
	// The same bytes again: the same id.
	again, err := s.Submit(ctx, body, "ip")
	if err != nil || again.ID != rc.ID || !again.Existing {
		t.Fatalf("%+v %v", again, err)
	}
	groups, err := store.Results(ctx, st.DB, store.Filter{Suite: "standard/v1"})
	if err != nil || len(groups) != 2 {
		t.Fatalf("%+v %v", groups, err)
	}
	g := groups[0]
	if g.Preset != "vram-8g" || g.System != "ASUS ROG Zephyrus M16" || g.CPU != "Intel Core i7-12700H" || g.Installs != 1 || g.Metrics["tg_d0"].Median != 60 {
		t.Fatalf("%+v", g)
	}
	v, err := store.GetView(ctx, st.DB, rc.ID)
	if err != nil || len(v.Runs) != 3 || v.Runs[0].Plausibility != store.PlausibleOK || v.Runs[0].Ceiling < 100 {
		t.Fatalf("%+v %v", v, err)
	}
}

func TestSuiteVersions(t *testing.T) {
	ctx := context.Background()
	s, _ := newService(t, Config{})
	for _, c := range []struct {
		id, version string
		want        int
	}{
		{"standard", "v0", http.StatusGone},
		{"standard", "v9", http.StatusUpgradeRequired},
		{"dev", "v1", http.StatusUnprocessableEntity},
	} {
		r := sample(60)
		r.Suite.ID, r.Suite.Version = c.id, c.version
		if _, err := s.Submit(ctx, signed(t, r, newKey()), ""); status(err) != c.want {
			t.Errorf("%s/%s: want %d, got %v", c.id, c.version, c.want, err)
		}
	}
	r := sample(60)
	r.Schema = "nanollm.bench/v2"
	b, _ := json.Marshal(r)
	if _, err := s.Submit(ctx, b, ""); status(err) != http.StatusUpgradeRequired {
		t.Errorf("v2 format: %v", err)
	}
}

func TestRefusals(t *testing.T) {
	ctx := context.Background()
	s, _ := newService(t, Config{})
	// Unsigned.
	b, _ := json.Marshal(sample(60))
	if _, err := s.Submit(ctx, b, ""); status(err) != http.StatusUnprocessableEntity {
		t.Errorf("unsigned: %v", err)
	}
	// Tampered after signing.
	r := sample(60)
	b = signed(t, r, newKey())
	b = []byte(strings.Replace(string(b), `"d0":60`, `"d0":61`, 1))
	if _, err := s.Submit(ctx, b, ""); status(err) != http.StatusUnprocessableEntity || !strings.Contains(err.Error(), "signature") {
		t.Errorf("tampered: %v", err)
	}
	// Prompts of another suite.
	r = sample(60)
	r.Suite.PromptsSHA256 = strings.Repeat("cd", 32)
	if _, err := s.Submit(ctx, signed(t, r, newKey()), ""); status(err) != http.StatusUnprocessableEntity {
		t.Errorf("prompts: %v", err)
	}
	// Too big.
	if _, err := s.Submit(ctx, make([]byte, MaxBody+1), ""); status(err) != http.StatusRequestEntityTooLarge {
		t.Errorf("size: %v", err)
	}
	// Implausibly fast for the GPU: 448 GB/s over 2.6 GB per token.
	if _, err := s.Submit(ctx, signed(t, sample(900), newKey()), ""); status(err) != http.StatusUnprocessableEntity || !strings.Contains(err.Error(), "ceiling") {
		t.Errorf("implausible: %v", err)
	}
}

func TestUnknownHardwareIsFlaggedNotRefused(t *testing.T) {
	ctx := context.Background()
	s, st := newService(t, Config{})
	r := sample(900)
	r.Machine.GPUs[0].Name = "Imaginary GPU 9000"
	r.Runs = r.Runs[:1]
	rc, err := s.Submit(ctx, signed(t, r, newKey()), "")
	if err != nil {
		t.Fatal(err)
	}
	if len(rc.Flags) == 0 || rc.Flags[0] != "plausibility:unknown" {
		t.Fatalf("%+v", rc)
	}
	v, _ := store.GetView(ctx, st.DB, rc.ID)
	if v.Runs[0].Plausibility != store.PlausibleUnknown {
		t.Fatal(v.Runs[0])
	}
}

func TestRateLimits(t *testing.T) {
	ctx := context.Background()
	s, _ := newService(t, Config{PerIP: NewLimiter(3, time.Hour), PerKey: NewLimiter(1, 24*time.Hour)})
	key := newKey()
	if _, err := s.Submit(ctx, signed(t, sample(60), key), "ip1"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Submit(ctx, signed(t, sample(61), key), "ip1"); status(err) != http.StatusTooManyRequests {
		t.Fatalf("per key: %v", err)
	}
	if _, err := s.Submit(ctx, signed(t, sample(62), newKey()), "ip1"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Submit(ctx, signed(t, sample(63), newKey()), "ip1"); status(err) != http.StatusTooManyRequests {
		t.Fatalf("per address: %v", err)
	}
}

func TestLimiterSaltRotates(t *testing.T) {
	l := NewLimiter(1, time.Hour)
	day := time.Date(2026, 10, 18, 23, 0, 0, 0, time.UTC)
	l.now = func() time.Time { return day }
	k1 := l.Key("203.0.113.7")
	if strings.Contains(k1, "203") || k1 != l.Key("203.0.113.7") {
		t.Fatal("the key must be a stable hash within a day")
	}
	day = day.Add(2 * time.Hour)
	if l.Key("203.0.113.7") == k1 {
		t.Fatal("the salt did not rotate with the day")
	}
}

func TestDedupeInAggregates(t *testing.T) {
	ctx := context.Background()
	s, st := newService(t, Config{})
	// One machine, a fresh key per run: counted once.
	for _, tg := range []float64{60, 62, 64} {
		if _, err := s.Submit(ctx, signed(t, sample(tg), newKey()), ""); err != nil {
			t.Fatal(err)
		}
	}
	// One install, repeats: counted once.
	other := newKey()
	for _, tg := range []float64{40, 42} {
		r := sample(tg)
		r.Machine.Memory.TotalBytes = 16 << 30 // another machine
		if _, err := s.Submit(ctx, signed(t, r, other), ""); err != nil {
			t.Fatal(err)
		}
	}
	gs, _ := store.Results(ctx, st.DB, store.Filter{Suite: "standard/v1", Preset: "vram-8g"})
	if len(gs) != 1 || gs[0].Installs != 2 {
		t.Fatalf("%+v", gs)
	}
	if m := gs[0].Metrics["tg_d0"]; m.Median != (62+41)/2.0 {
		t.Fatalf("%+v", m)
	}
}

func TestOutliersQueue(t *testing.T) {
	ctx := context.Background()
	s, st := newService(t, Config{})
	for i, tg := range []float64{60, 61, 59, 62, 60, 58} {
		r := sample(tg)
		r.Machine.Memory.TotalBytes = int64(16+i) << 30 // distinct machines
		if _, err := s.Submit(ctx, signed(t, r, newKey()), ""); err != nil {
			t.Fatal(err)
		}
	}
	r := sample(180)
	r.Machine.Memory.TotalBytes = 64 << 30
	rc, err := s.Submit(ctx, signed(t, r, newKey()), "")
	if err != nil {
		t.Fatal(err)
	}
	if rc.State != store.StateQueued {
		t.Fatalf("an outlier was published: %+v", rc)
	}
	q, _ := store.ListModeration(ctx, st.DB, store.StateQueued, 10)
	if len(q) != 1 || q[0].ID != rc.ID {
		t.Fatal(q)
	}
	// Not in the aggregates until a moderator publishes it.
	gs, _ := store.Results(ctx, st.DB, store.Filter{Preset: "vram-8g"})
	if gs[0].Installs != 6 {
		t.Fatal(gs[0].Installs)
	}
	if err := s.Moderate(ctx, rc.ID, store.StatePublished, "checked"); err != nil {
		t.Fatal(err)
	}
	gs, _ = store.Results(ctx, st.DB, store.Filter{Preset: "vram-8g"})
	if gs[0].Installs != 7 {
		t.Fatal(gs[0].Installs)
	}
}

func TestWithdraw(t *testing.T) {
	ctx := context.Background()
	s, st := newService(t, Config{})
	key := newKey()
	rc, err := s.Submit(ctx, signed(t, sample(60), key), "")
	if err != nil {
		t.Fatal(err)
	}
	pub := hex.EncodeToString(key.Public().(ed25519.PublicKey))
	sign := func(k ed25519.PrivateKey, id string) string {
		return hex.EncodeToString(ed25519.Sign(k, benchfmt.WithdrawalMessage(id)))
	}
	if err := s.Withdraw(ctx, rc.ID, pub, sign(newKey(), rc.ID)); status(err) != http.StatusForbidden {
		t.Fatalf("someone else's key: %v", err)
	}
	if err := s.Withdraw(ctx, rc.ID, pub, sign(key, "aaaaaaaaaaaaa")); status(err) != http.StatusForbidden {
		t.Fatalf("a signature over another id: %v", err)
	}
	if err := s.Withdraw(ctx, rc.ID, pub, sign(key, rc.ID)); err != nil {
		t.Fatal(err)
	}
	if err := s.Withdraw(ctx, rc.ID, pub, sign(key, rc.ID)); err != nil {
		t.Fatalf("a repeated withdrawal: %v", err)
	}
	if gs, _ := store.Results(ctx, st.DB, store.Filter{}); len(gs) != 0 {
		t.Fatal("the aggregates still count a withdrawn submission")
	}
	if _, _, err := store.GetSubmission(ctx, st.DB, rc.ID); !errors.Is(err, store.ErrNotFound) {
		t.Fatal(err)
	}
}

func TestReindexAppliesAliases(t *testing.T) {
	ctx := context.Background()
	s, st := newService(t, Config{})
	if _, err := s.Submit(ctx, signed(t, sample(60), newKey()), ""); err != nil {
		t.Fatal(err)
	}
	before, _ := store.Results(ctx, st.DB, store.Filter{})
	if err := store.SetAlias(ctx, st.DB, store.Alias{Kind: "system", Pattern: "ASUS ROG Zephyrus M16", Name: "ASUS ROG Zephyrus M16 (2022)"}); err != nil {
		t.Fatal(err)
	}
	n, err := s.Reindex(ctx)
	if err != nil || n != 1 {
		t.Fatal(n, err)
	}
	after, _ := store.Results(ctx, st.DB, store.Filter{})
	if len(after) != len(before) || after[0].System != "ASUS ROG Zephyrus M16 (2022)" || after[0].SystemSlug != "asus-rog-zephyrus-m16-2022" {
		t.Fatalf("%+v", after[0])
	}
	if after[0].Metrics["tg_d0"] != before[0].Metrics["tg_d0"] {
		t.Fatal("reindex changed the numbers")
	}
	if need, _ := s.NeedsReindex(ctx); need {
		t.Fatal("fresh index reported stale")
	}
}

func TestOutputMatch(t *testing.T) {
	ctx := context.Background()
	s, st := newService(t, Config{})
	r := sample(60)
	r.Runs[0].Output.Prefixes = []benchfmt.Prefix{{Tokens: 16, SHA256: strings.Repeat("11", 32)}, {Tokens: 64, SHA256: strings.Repeat("22", 32)}}
	first, err := s.Submit(ctx, signed(t, r, newKey()), "")
	if err != nil {
		t.Fatal(err)
	}
	v, _ := store.GetView(ctx, st.DB, first.ID)
	if v.Runs[0].OutputMatch != -1 {
		t.Fatal("nothing to compare with yet", v.Runs[0].OutputMatch)
	}
	r2 := sample(61)
	r2.Machine.Memory.TotalBytes = 16 << 30
	r2.Runs[0].Output.SHA256 = strings.Repeat("ef", 32)
	r2.Runs[0].Output.Prefixes = []benchfmt.Prefix{{Tokens: 16, SHA256: strings.Repeat("11", 32)}, {Tokens: 64, SHA256: strings.Repeat("33", 32)}}
	second, err := s.Submit(ctx, signed(t, r2, newKey()), "")
	if err != nil {
		t.Fatal(err)
	}
	v, _ = store.GetView(ctx, st.DB, second.ID)
	if v.Runs[0].OutputMatch != 16 {
		t.Fatal(v.Runs[0].OutputMatch)
	}
	if second.State != store.StatePublished {
		t.Fatal("a differing output must not block publication")
	}
}
