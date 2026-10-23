package hwinfo

import "testing"

func TestVendorOf(t *testing.T) {
	for desc, want := range map[string]string{
		"NVIDIA GeForce RTX 3070 Ti Laptop GPU": "NVIDIA",
		"Intel(R) Iris(R) Xe Graphics":          "Intel",
		"AMD Radeon(TM) Graphics":               "AMD",
		"Microsoft Basic Render Driver":         "Microsoft",
		"Some Unknown Adapter":                  "",
	} {
		if got := vendorOf(desc); got != want {
			t.Errorf("%q is %q, want %q", desc, got, want)
		}
	}
}

func TestParseDriverDate(t *testing.T) {
	for in, want := range map[string]string{
		"9-15-2025":  "2025-09-15",
		"9/15/2025":  "2025-09-15",
		"2025-09-15": "2025-09-15",
		"15.09.2025": "",
		"":           "",
	} {
		if got := parseDriverDate(in); got != want {
			t.Errorf("%q is %q, want %q", in, got, want)
		}
	}
}

func TestBusName(t *testing.T) {
	for in, want := range map[uint32]string{0x07: "usb", 0x0B: "sata", 0x11: "nvme", 0x0A: "sas"} {
		if got := busName(in); got != want {
			t.Errorf("bus %#x is %q, want %q", in, got, want)
		}
	}
	if got := busName(0xFF); got != "" {
		t.Errorf("an unknown bus is %q", got)
	}
}
