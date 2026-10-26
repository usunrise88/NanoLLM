package hwinfo

import (
	"strings"
	"time"
)

// GPUDriver is a display driver the OS has installed. Intel and AMD drivers on
// Windows are not versioned by nvidia-smi, and the registry is where their
// version and date are.
type GPUDriver struct {
	Name    string `json:"name"`
	Vendor  string `json:"vendor,omitempty"`
	Version string `json:"version,omitempty"`
	// Date is the driver's release date (YYYY-MM-DD), empty when unknown.
	Date string `json:"date,omitempty"`
}

// Age is how old the driver is at now, zero when its date is unknown.
func (d GPUDriver) Age(now time.Time) time.Duration {
	t, err := time.Parse("2006-01-02", d.Date)
	if err != nil {
		return 0
	}
	return now.Sub(t)
}

// vendorOf names the maker from a display adapter's description.
func vendorOf(desc string) string {
	d := strings.ToLower(desc)
	switch {
	case strings.Contains(d, "nvidia"), strings.Contains(d, "geforce"), strings.Contains(d, "quadro"):
		return "NVIDIA"
	case strings.Contains(d, "intel"):
		return "Intel"
	case strings.Contains(d, "amd"), strings.Contains(d, "radeon"), strings.Contains(d, "ati "):
		return "AMD"
	case strings.Contains(d, "apple"):
		return "Apple"
	case strings.Contains(d, "microsoft basic"), strings.Contains(d, "basic render"):
		return "Microsoft"
	}
	return ""
}

// parseDriverDate turns the registry's "9-15-2025" or "9/15/2025" (month-day-
// year) or a "2025-09-15" into YYYY-MM-DD.
func parseDriverDate(s string) string {
	for _, layout := range []string{"1-2-2006", "1/2/2006", "2006-01-02"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.Format("2006-01-02")
		}
	}
	return ""
}
