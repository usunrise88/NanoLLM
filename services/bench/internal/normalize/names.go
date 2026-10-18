// Package normalize turns the hardware strings a submission carries into the
// names the public pages group by: "12th Gen Intel(R) Core(TM) i7-12700H"
// and "Intel Core i7-12700H @ 2.30GHz" are one CPU, "ASUSTeK COMPUTER INC."
// and "ASUS" are one vendor, a Lenovo SKU is its ThinkPad family, a Mac's
// hw.model is its marketing name plus the GPU core count.
//
// The normalizer is versioned (Version). Its output lives in derived tables
// that `benchd reindex` rebuilds from the raw submissions, so a better rule
// or a new alias is a version bump and a reindex, never an edit of stored
// rows.
package normalize

import (
	"fmt"
	"regexp"
	"strings"
)

// Version is the normalizer's version. Bump it with any rule change: the
// server reindexes when the stored rows were made by an older one.
const Version = 1

// Alias kinds, as the admin alias table names them.
const (
	KindCPU    = "cpu"
	KindGPU    = "gpu"
	KindVendor = "vendor"
	KindSystem = "system"
)

// Aliases maps kind → normalized name (lower case) → the name to show. The
// admin edits it; normalization applies it after the built-in rules.
type Aliases map[string]map[string]string

func (a Aliases) apply(kind, name string) string {
	if to, ok := a[kind][strings.ToLower(name)]; ok {
		return to
	}
	return name
}

// Set adds one alias.
func (a Aliases) Set(kind, from, to string) {
	if a[kind] == nil {
		a[kind] = map[string]string{}
	}
	a[kind][strings.ToLower(strings.TrimSpace(from))] = strings.TrimSpace(to)
}

var (
	trademarks  = regexp.MustCompile(`(?i)\((r|tm|c)\)|[®™©]`)
	frequency   = regexp.MustCompile(`(?i)\s*@?\s*\d+(\.\d+)?\s*[gm]hz`)
	generation  = regexp.MustCompile(`(?i)\b\d+(st|nd|rd|th)\s+gen(eration)?\b`)
	coreCount   = regexp.MustCompile(`(?i)\b\d+-core\b`)
	cpuNoise    = regexp.MustCompile(`(?i)\b(cpu|processor|apu)\b`)
	withGfx     = regexp.MustCompile(`(?i)\s+(with|w/)\s+.*(graphics|radeon).*$`)
	spaces      = regexp.MustCompile(`\s+`)
	gpuSuffix   = regexp.MustCompile(`(?i)\s+(gpu|graphics)$`)
	slugBad     = regexp.MustCompile(`[^a-z0-9]+`)
	skuToken    = regexp.MustCompile(`^[A-Za-z0-9_-]{4,}$`)
	hasLetter   = regexp.MustCompile(`[A-Za-z]`)
	hasDigit    = regexp.MustCompile(`[0-9]`)
	hpFamilyPfx = regexp.MustCompile(`^(?i)103C_\w+\s+`)
	pcSuffix    = regexp.MustCompile(`(?i)\s+(notebook|laptop|desktop|mobile workstation|all-in-one)\s+pc$`)
	macModel    = regexp.MustCompile(`^(Mac|MacBook|MacBookPro|MacBookAir|iMac|iMacPro|Macmini|MacPro)\d+,\d+$`)
)

func squash(s string) string { return strings.TrimSpace(spaces.ReplaceAllString(s, " ")) }

// CPUVendor maps the CPUID vendor strings to the names people use.
func CPUVendor(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "genuineintel", "intel":
		return "Intel"
	case "authenticamd", "amd":
		return "AMD"
	case "apple":
		return "Apple"
	case "qualcomm":
		return "Qualcomm"
	case "arm", "arm limited":
		return "Arm"
	}
	return squash(trademarks.ReplaceAllString(v, ""))
}

// CPU normalizes a processor name: trademarks, frequency, "12th Gen",
// "8-Core Processor" and "with Radeon Graphics" go, and the vendor leads.
func CPU(vendor, model string) string {
	s := trademarks.ReplaceAllString(model, "")
	s = withGfx.ReplaceAllString(s, "")
	s = frequency.ReplaceAllString(s, "")
	s = generation.ReplaceAllString(s, "")
	s = coreCount.ReplaceAllString(s, "")
	s = cpuNoise.ReplaceAllString(s, "")
	s = squash(s)
	v := CPUVendor(vendor)
	if v != "" && !strings.HasPrefix(strings.ToLower(s), strings.ToLower(v)) {
		s = v + " " + s
	}
	return squash(s)
}

// GPUVendor names the GPU's maker from the vendor field or the name.
func GPUVendor(vendor, name string) string {
	for _, s := range []string{vendor, name} {
		l := strings.ToLower(s)
		switch {
		case strings.Contains(l, "nvidia") || strings.Contains(l, "geforce") || strings.Contains(l, "quadro"):
			return "NVIDIA"
		case strings.Contains(l, "amd") || strings.Contains(l, "radeon") || strings.Contains(l, "ati "):
			return "AMD"
		case strings.Contains(l, "intel") || strings.Contains(l, "arc") || strings.Contains(l, "iris"):
			return "Intel"
		case strings.Contains(l, "apple"):
			return "Apple"
		case strings.Contains(l, "adreno") || strings.Contains(l, "qualcomm"):
			return "Qualcomm"
		}
	}
	return ""
}

// GPU normalizes a graphics adapter's name; an Apple GPU carries its core
// count, which is what tells two chips of one name apart.
func GPU(vendor, name string, cores int) string {
	s := squash(trademarks.ReplaceAllString(name, ""))
	s = gpuSuffix.ReplaceAllString(s, "")
	s = squash(gpuSuffix.ReplaceAllString(s, "")) // "Laptop GPU" → "Laptop", "Graphics GPU" twice
	v := GPUVendor(vendor, name)
	if v != "" && !strings.HasPrefix(strings.ToLower(s), strings.ToLower(v)) {
		s = v + " " + s
	}
	if v == "Apple" && cores > 0 {
		s = fmt.Sprintf("%s (%d-core GPU)", s, cores)
	}
	return squash(s)
}

// vendorTable is the built-in vendor aliases; the admin table extends it.
var vendorTable = map[string]string{
	"asustek computer inc.":              "ASUS",
	"asustek computer inc":               "ASUS",
	"asus":                               "ASUS",
	"lenovo":                             "Lenovo",
	"dell inc.":                          "Dell",
	"dell":                               "Dell",
	"hp":                                 "HP",
	"hewlett-packard":                    "HP",
	"micro-star international co., ltd.": "MSI",
	"micro-star international co., ltd":  "MSI",
	"msi":                                "MSI",
	"acer":                               "Acer",
	"apple inc.":                         "Apple",
	"apple":                              "Apple",
	"microsoft corporation":              "Microsoft",
	"gigabyte technology co., ltd.":      "Gigabyte",
	"razer":                              "Razer",
	"framework":                          "Framework",
	"huawei":                             "Huawei",
	"xiaomi":                             "Xiaomi",
	"samsung electronics co., ltd.":      "Samsung",
	"lg electronics":                     "LG",
	"tongfang hongkong limited":          "Tongfang",
	"supermicro":                         "Supermicro",
	"super micro computer, inc.":         "Supermicro",
	"intel corporation":                  "Intel",
	"qemu":                               "QEMU",
	"innotek gmbh":                       "VirtualBox",
	"vmware, inc.":                       "VMware",
	"microsoft corporation (hyper-v)":    "Hyper-V",
	"amazon ec2":                         "Amazon EC2",
	"google":                             "Google",
	"hetzner":                            "Hetzner",
	"medion":                             "Medion",
	"system76":                           "System76",
	"tuxedo computers gmbh":              "TUXEDO",
	"chuwi innovation and technology(shenzhen)co.,ltd": "Chuwi",
}

var corporate = regexp.MustCompile(`(?i)[\s,]+(inc\.?|co\.,?\s*ltd\.?|ltd\.?|corporation|corp\.?|gmbh|limited|computer inc\.?)$`)

// placeholders are what firmware vendors leave in unfilled SMBIOS fields.
var placeholders = map[string]bool{
	"": true, "to be filled by o.e.m.": true, "default string": true, "system product name": true,
	"system version": true, "system manufacturer": true, "not applicable": true, "none": true,
	"o.e.m.": true, "oem": true, "not specified": true, "type1productconfigid": true, "invalid": true,
	"unknown": true, "x.x": true, "0123456789": true, "1234567890": true,
}

// Placeholder reports whether an SMBIOS string carries no information.
func Placeholder(s string) bool { return placeholders[strings.ToLower(strings.TrimSpace(s))] }

// Vendor normalizes a system manufacturer.
func Vendor(v string) string {
	v = squash(trademarks.ReplaceAllString(v, ""))
	if Placeholder(v) {
		return ""
	}
	if to, ok := vendorTable[strings.ToLower(v)]; ok {
		return to
	}
	for {
		s := corporate.ReplaceAllString(v, "")
		if s == v {
			break
		}
		v = s
	}
	if to, ok := vendorTable[strings.ToLower(v)]; ok {
		return to
	}
	return v
}

// skuLike: a part number rather than a name ("21HMCTO1WW", "81YQ").
func skuLike(s string) bool {
	return !strings.Contains(s, " ") && skuToken.MatchString(s) && hasLetter.MatchString(s) && hasDigit.MatchString(s)
}

func stripVendor(s, vendor string) string {
	if vendor == "" {
		return s
	}
	for _, p := range []string{vendor + " ", strings.ToUpper(vendor) + " "} {
		if len(s) > len(p) && strings.EqualFold(s[:len(p)], p) {
			return strings.TrimSpace(s[len(p):])
		}
	}
	return s
}

// System names a machine: vendor and marketing name. The firmware's family
// wins over a SKU-like product name (Lenovo: "21HMCTO1WW" is a ThinkPad X1
// Carbon Gen 11), the product wins over a coarse family (Dell: "XPS" vs "XPS
// 13 9310"), and a trailing SKU after the family goes (ASUS: "ROG Zephyrus
// G14 GA402RJ_GA402RJ"). A Mac is its hw.model's name and GPU core count.
func System(vendor, family, product string, gpuCores int) string {
	v := Vendor(vendor)
	if m := strings.TrimSpace(product); macModel.MatchString(m) {
		name, ok := macNames[m]
		if !ok {
			name = "Apple " + m
		}
		if gpuCores > 0 {
			name = fmt.Sprintf("%s, %d-core GPU", name, gpuCores)
		}
		return name
	}
	f := squash(trademarks.ReplaceAllString(family, ""))
	p := squash(trademarks.ReplaceAllString(product, ""))
	if Placeholder(f) {
		f = ""
	}
	if Placeholder(p) {
		p = ""
	}
	f = hpFamilyPfx.ReplaceAllString(f, "")
	f, p = stripVendor(f, v), stripVendor(p, v)
	p = pcSuffix.ReplaceAllString(p, "")
	var name string
	switch {
	case p == "" || skuLike(p):
		name = f
		if name == "" {
			name = p
		}
	case f != "" && len(p) > len(f) && strings.EqualFold(p[:len(f)], f):
		rest := strings.TrimSpace(p[len(f):])
		if skuLike(rest) {
			name = f
		} else {
			name = p
		}
	default:
		name = p
	}
	return squash(v + " " + name)
}

// macNames maps hw.model to the marketing name. It is not exhaustive: an
// unknown model shows as "Apple <hw.model>" until an alias names it.
var macNames = map[string]string{
	"MacBookAir10,1": "MacBook Air (M1, 2020)",
	"MacBookPro17,1": "MacBook Pro 13\" (M1, 2020)",
	"MacBookPro18,1": "MacBook Pro 16\" (2021)",
	"MacBookPro18,2": "MacBook Pro 16\" (2021)",
	"MacBookPro18,3": "MacBook Pro 14\" (2021)",
	"MacBookPro18,4": "MacBook Pro 14\" (2021)",
	"Macmini9,1":     "Mac mini (M1, 2020)",
	"iMac21,1":       "iMac 24\" (M1, 2021)",
	"iMac21,2":       "iMac 24\" (M1, 2021)",
	"Mac13,1":        "Mac Studio (2022)",
	"Mac13,2":        "Mac Studio (2022)",
	"Mac14,2":        "MacBook Air 13\" (M2, 2022)",
	"Mac14,7":        "MacBook Pro 13\" (M2, 2022)",
	"Mac14,3":        "Mac mini (M2, 2023)",
	"Mac14,12":       "Mac mini (M2 Pro, 2023)",
	"Mac14,5":        "MacBook Pro 14\" (2023)",
	"Mac14,9":        "MacBook Pro 14\" (2023)",
	"Mac14,6":        "MacBook Pro 16\" (2023)",
	"Mac14,10":       "MacBook Pro 16\" (2023)",
	"Mac14,15":       "MacBook Air 15\" (M2, 2023)",
	"Mac14,13":       "Mac Studio (2023)",
	"Mac14,14":       "Mac Studio (2023)",
	"Mac15,3":        "MacBook Pro 14\" (M3, 2023)",
	"Mac15,6":        "MacBook Pro 14\" (2023)",
	"Mac15,8":        "MacBook Pro 14\" (2023)",
	"Mac15,10":       "MacBook Pro 14\" (2023)",
	"Mac15,7":        "MacBook Pro 16\" (2023)",
	"Mac15,9":        "MacBook Pro 16\" (2023)",
	"Mac15,11":       "MacBook Pro 16\" (2023)",
	"Mac15,12":       "MacBook Air 13\" (M3, 2024)",
	"Mac15,13":       "MacBook Air 15\" (M3, 2024)",
	"Mac16,1":        "MacBook Pro 14\" (M4, 2024)",
	"Mac16,6":        "MacBook Pro 14\" (2024)",
	"Mac16,8":        "MacBook Pro 14\" (2024)",
	"Mac16,5":        "MacBook Pro 16\" (2024)",
	"Mac16,7":        "MacBook Pro 16\" (2024)",
	"Mac16,10":       "Mac mini (M4, 2024)",
	"Mac16,11":       "Mac mini (M4 Pro, 2024)",
	"Mac16,12":       "MacBook Air 13\" (M4, 2025)",
	"Mac16,13":       "MacBook Air 15\" (M4, 2025)",
}

// Slug is the URL form of a name: lower case ASCII, words joined by "-".
func Slug(s string) string {
	s = strings.Trim(slugBad.ReplaceAllString(strings.ToLower(s), "-"), "-")
	if s == "" {
		return "unknown"
	}
	if len(s) > 80 {
		s = strings.TrimRight(s[:80], "-")
	}
	return s
}

// OS is the display name of the operating system, coarse on purpose:
// Windows and macOS keep their major version, Linux its distribution when the
// client sent one.
func OS(name, version string, wsl bool) string {
	major := strings.SplitN(strings.TrimSpace(version), ".", 2)[0]
	var s string
	switch strings.ToLower(name) {
	case "windows":
		s = squash("Windows " + major)
	case "darwin", "macos":
		s = squash("macOS " + major)
	case "linux":
		s = "Linux"
		if v := strings.TrimSpace(version); v != "" && hasLetter.MatchString(v) {
			s = squash(v) // a distribution name such as "Ubuntu 24.04"
		}
	case "":
		s = "unknown"
	default:
		s = squash(name + " " + major)
	}
	if wsl {
		s += " (WSL)"
	}
	return s
}

// Backend is the ggml backend a plan's device runs on: "CUDA0" → "CUDA".
func Backend(device string) string {
	d := strings.TrimRight(strings.TrimSpace(device), "0123456789")
	if d == "" {
		return "CPU"
	}
	return d
}
