package normalize

import (
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/benchfmt"
)

func TestCPU(t *testing.T) {
	for _, c := range []struct{ vendor, model, want string }{
		{"GenuineIntel", "12th Gen Intel(R) Core(TM) i7-12700H", "Intel Core i7-12700H"},
		{"GenuineIntel", "Intel(R) Core(TM) i5-1035G1 CPU @ 1.00GHz", "Intel Core i5-1035G1"},
		{"Intel", "Core i7-12700H", "Intel Core i7-12700H"},
		{"AuthenticAMD", "AMD Ryzen 9 5900X 12-Core Processor", "AMD Ryzen 9 5900X"},
		{"AuthenticAMD", "AMD Ryzen 7 7840HS w/ Radeon 780M Graphics", "AMD Ryzen 7 7840HS"},
		{"AuthenticAMD", "AMD EPYC 9454P 48-Core Processor", "AMD EPYC 9454P"},
		{"Apple", "Apple M2 Pro", "Apple M2 Pro"},
		{"AuthenticAMD", "AMD Ryzen AI 9 HX 370 w/ Radeon 890M", "AMD Ryzen AI 9 HX 370"},
		{"GenuineIntel", "Intel(R) Core(TM) Ultra 7 155H", "Intel Core Ultra 7 155H"},
		{"GenuineIntel", "13th Gen Intel(R) Core(TM) i9-13900K", "Intel Core i9-13900K"},
	} {
		if got := CPU(c.vendor, c.model); got != c.want {
			t.Errorf("CPU(%q, %q) = %q, want %q", c.vendor, c.model, got, c.want)
		}
	}
}

func TestGPU(t *testing.T) {
	for _, c := range []struct {
		vendor, name string
		cores        int
		want         string
	}{
		{"NVIDIA", "NVIDIA GeForce RTX 3070 Ti Laptop GPU", 0, "NVIDIA GeForce RTX 3070 Ti Laptop"},
		{"", "GeForce RTX 4090", 0, "NVIDIA GeForce RTX 4090"},
		{"AMD", "AMD Radeon 780M Graphics", 0, "AMD Radeon 780M"},
		{"", "Intel(R) Iris(R) Xe Graphics", 0, "Intel Iris Xe"},
		{"Apple", "Apple M2 Pro", 19, "Apple M2 Pro (19-core GPU)"},
	} {
		if got := GPU(c.vendor, c.name, c.cores); got != c.want {
			t.Errorf("GPU(%q, %q) = %q, want %q", c.vendor, c.name, got, c.want)
		}
	}
}

func TestVendor(t *testing.T) {
	for in, want := range map[string]string{
		"ASUSTeK COMPUTER INC.":              "ASUS",
		"LENOVO":                             "Lenovo",
		"Dell Inc.":                          "Dell",
		"Micro-Star International Co., Ltd.": "MSI",
		"To Be Filled By O.E.M.":             "",
		"Some Maker Co., Ltd.":               "Some Maker",
		"HP":                                 "HP",
	} {
		if got := Vendor(in); got != want {
			t.Errorf("Vendor(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSystem(t *testing.T) {
	for _, c := range []struct {
		vendor, family, product string
		cores                   int
		want                    string
	}{
		// SKU → family.
		{"LENOVO", "ThinkPad X1 Carbon Gen 11", "21HMCTO1WW", 0, "Lenovo ThinkPad X1 Carbon Gen 11"},
		// A trailing SKU after the family goes.
		{"ASUSTeK COMPUTER INC.", "ROG Zephyrus G14", "ROG Zephyrus G14 GA402RJ_GA402RJ", 0, "ASUS ROG Zephyrus G14"},
		// A coarse family loses to the product.
		{"Dell Inc.", "XPS", "XPS 13 9310", 0, "Dell XPS 13 9310"},
		// HP's family carries a board id; the product a "Notebook PC" suffix.
		{"HP", "103C_5336AN HP EliteBook", "HP EliteBook 840 G8 Notebook PC", 0, "HP EliteBook 840 G8"},
		{"System manufacturer", "To be filled by O.E.M.", "System Product Name", 0, ""},
		// A board named by the client (ASRockRack EPYCD8-2T, whose system strings are placeholders).
		{"ASRockRack", "EPYCD8-2T", "EPYCD8-2T", 0, "ASRockRack EPYCD8-2T"},
		// Mac: hw.model plus GPU cores.
		{"Apple Inc.", "", "Mac14,5", 30, "MacBook Pro 14\" (2023), 30-core GPU"},
		{"Apple Inc.", "", "Mac99,1", 0, "Apple Mac99,1"},
		{"QEMU", "", "Standard PC (Q35 + ICH9, 2009)", 0, "QEMU Standard PC (Q35 + ICH9, 2009)"},
	} {
		if got := System(c.vendor, c.family, c.product, c.cores); got != c.want {
			t.Errorf("System(%q, %q, %q) = %q, want %q", c.vendor, c.family, c.product, got, c.want)
		}
	}
}

func TestSlugAndOS(t *testing.T) {
	if got := Slug("ASUS ROG Zephyrus G14"); got != "asus-rog-zephyrus-g14" {
		t.Error(got)
	}
	if got := Slug("  ★ "); got != "unknown" {
		t.Error(got)
	}
	for _, c := range []struct {
		name, version string
		wsl           bool
		want          string
	}{
		{"windows", "11", false, "Windows 11"},
		{"darwin", "15.1", false, "macOS 15"},
		{"linux", "", true, "Linux (WSL)"},
		{"linux", "Ubuntu 24.04", false, "Ubuntu 24.04"},
	} {
		if got := OS(c.name, c.version, c.wsl); got != c.want {
			t.Errorf("OS(%q, %q) = %q, want %q", c.name, c.version, got, c.want)
		}
	}
	if Backend("CUDA0") != "CUDA" || Backend("") != "CPU" || Backend("Metal") != "Metal" {
		t.Error("backend")
	}
}

func sample() *benchfmt.Result {
	sha := strings.Repeat("ab", 32)
	return &benchfmt.Result{
		Schema: benchfmt.Schema,
		Suite:  benchfmt.Suite{ID: "standard", Version: "v1", PromptsSHA256: sha},
		Client: benchfmt.Client{NanoLLM: "v1.0.0", LlamaCpp: benchfmt.LlamaCpp{Tag: "b11600"}, OS: "windows", Arch: "amd64"},
		Model:  benchfmt.Model{ID: "gemma-4-e4b-qat", SHA256: sha},
		Machine: benchfmt.Machine{
			CPU:    benchfmt.CPU{Vendor: "GenuineIntel", Model: "12th Gen Intel(R) Core(TM) i7-12700H", Cores: 14, Threads: 20},
			Memory: benchfmt.Memory{TotalBytes: 32 << 30},
			GPUs:   []benchfmt.GPU{{Name: "NVIDIA GeForce RTX 3070 Ti Laptop GPU", Vendor: "NVIDIA", VRAMBytes: 8 << 30, DriverMajor: "560"}},
			System: benchfmt.System{Vendor: "ASUSTeK COMPUTER INC.", Family: "ROG Zephyrus M16", Product: "ROG Zephyrus M16 GU603ZW_GU603ZW", FormFactor: benchfmt.FormLaptop},
			OS:     benchfmt.OSInfo{Name: "windows", Version: "11"},
		},
		Runs: []benchfmt.Run{{Preset: "vram-8g", Plan: benchfmt.Plan{Mode: "vram-full", Device: "CUDA0", GPULayers: -1}, Reps: 3, Status: "ok",
			Metrics: benchfmt.Metrics{TGTokS: benchfmt.TG{D0: 60}}}},
		Date: "2026-10-18",
	}
}

func TestNormalizeAndAliases(t *testing.T) {
	r := sample()
	rec := Normalize(r, nil)
	if rec.CPU != "Intel Core i7-12700H" || rec.GPU != "NVIDIA GeForce RTX 3070 Ti Laptop" || rec.System != "ASUS ROG Zephyrus M16" {
		t.Fatalf("%+v", rec)
	}
	if rec.SystemSlug != "asus-rog-zephyrus-m16" || rec.Suite != "standard/v1" || rec.Runs[0].Backend != "CUDA" {
		t.Fatalf("%+v", rec)
	}
	al := Aliases{}
	al.Set(KindSystem, "ASUS ROG Zephyrus M16", "ASUS ROG Zephyrus M16 (2022)")
	if got := Normalize(r, al).System; got != "ASUS ROG Zephyrus M16 (2022)" {
		t.Fatal(got)
	}

	// The fingerprint ignores versions and identity, not hardware.
	other := sample()
	other.Client.NanoLLM, other.Machine.GPUs[0].DriverMajor, other.InstallID = "v1.0.1", "570", "ff"
	if Normalize(other, nil).Fingerprint != rec.Fingerprint {
		t.Error("a driver update changed the hardware fingerprint")
	}
	other.Machine.Memory.TotalBytes = 16 << 30
	if Normalize(other, nil).Fingerprint == rec.Fingerprint {
		t.Error("different memory, same fingerprint")
	}
}
