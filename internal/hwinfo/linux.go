package hwinfo

import (
	"bufio"
	"context"
	"io/fs"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
)

// linuxProfile reads a Linux machine through fsys rooted at "/", so tests can
// hand it a synthetic tree.
func linuxProfile(fsys fs.FS) benchfmt.Machine {
	var m benchfmt.Machine
	read := func(p string) string {
		b, err := fs.ReadFile(fsys, strings.TrimPrefix(p, "/"))
		if err != nil {
			return ""
		}
		return strings.TrimSpace(string(b))
	}

	// CPU: the brand string, the ISA flags that pick ggml's kernels, and the
	// topology.
	flags := map[string]bool{}
	logical := 0
	cores := map[string]bool{}
	physID, coreID := "", ""
	if f, err := fsys.Open("proc/cpuinfo"); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 64<<10), 1<<20)
		for sc.Scan() {
			k, v, ok := strings.Cut(sc.Text(), ":")
			if !ok {
				continue
			}
			k, v = strings.TrimSpace(k), strings.TrimSpace(v)
			switch k {
			case "processor":
				logical++
			case "model name", "Model":
				if m.CPU.Model == "" {
					m.CPU.Model = v
				}
			case "vendor_id":
				if m.CPU.Vendor == "" {
					m.CPU.Vendor = map[string]string{"GenuineIntel": "Intel", "AuthenticAMD": "AMD"}[v]
				}
			case "flags", "Features":
				if len(flags) == 0 {
					for _, fl := range strings.Fields(v) {
						flags[fl] = true
					}
				}
			case "physical id":
				physID = v
			case "core id":
				coreID = v
				cores[physID+"/"+coreID] = true
			}
		}
		f.Close()
	}
	m.CPU.Threads = logical
	m.CPU.Cores = len(cores)
	if m.CPU.Cores == 0 {
		m.CPU.Cores = logical
	}
	for _, isa := range []string{"avx", "avx2", "fma", "f16c", "avx512f", "avx512_vnni", "avx512_bf16", "avx_vnni", "amx_tile", "neon", "asimd", "sve", "i8mm", "dotprod"} {
		if flags[isa] {
			m.CPU.ISA = append(m.CPU.ISA, isa)
		}
	}
	m.CPU.PCores, m.CPU.ECores = hybridCores(fsys)

	if mem := read("/proc/meminfo"); mem != "" {
		for _, line := range strings.Split(mem, "\n") {
			if rest, ok := strings.CutPrefix(line, "MemTotal:"); ok {
				kb, _ := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(rest), " kB"), 10, 64)
				m.Memory.TotalBytes = kb << 10
			}
		}
	}

	// The system: DMI without the serial and UUID files, which are never
	// opened.
	m.System.Vendor = read("/sys/class/dmi/id/sys_vendor")
	product := read("/sys/class/dmi/id/product_name")
	version := read("/sys/class/dmi/id/product_version")
	family := read("/sys/class/dmi/id/product_family")
	m.System.Product = product
	m.System.Family = marketingName(family, version, product)
	if t, err := strconv.Atoi(read("/sys/class/dmi/id/chassis_type")); err == nil {
		m.System.FormFactor = FormFactorOfChassis(t)
		m.System.Chassis = ChassisName(t)
	}
	if looksVirtual(m.System.Vendor, product) || (m.System.Vendor == "" && flags["hypervisor"]) {
		m.System.FormFactor = benchfmt.FormVM
	}

	// OS.
	osr := parseOSRelease(read("/etc/os-release"))
	m.OS.Name = "linux"
	if osr["ID"] != "" {
		m.OS.Name = osr["ID"]
	}
	m.OS.Version = osr["VERSION_ID"]
	release := read("/proc/sys/kernel/osrelease")
	m.OS.KernelMM = majorMinor(release)
	m.OS.WSL = strings.Contains(strings.ToLower(release), "microsoft")

	m.Power = linuxPower(fsys, read)
	m.Disk.Type = linuxDisk(fsys, read)
	return m
}

// marketingName picks the name people know the machine by. Lenovo puts the
// marketing name in product_version ("ThinkPad X1 Carbon Gen 11") and a part
// number in product_name; most others use product_family or product_name.
func marketingName(family, version, product string) string {
	junk := func(s string) bool {
		l := strings.ToLower(s)
		return s == "" || strings.Contains(l, "to be filled") || strings.Contains(l, "default string") ||
			strings.Contains(l, "not specified") || strings.Contains(l, "system version") || l == "none" || l == "0" || l == "1.0"
	}
	if !junk(version) && strings.ContainsAny(version, " ") && !strings.HasPrefix(version, "V") {
		return version
	}
	if !junk(family) {
		return family
	}
	if !junk(product) {
		return product
	}
	return ""
}

// hybridCores counts performance and efficiency cores where the kernel
// exposes them: Intel's cpu_core/cpu_atom PMUs, or differing max frequencies.
func hybridCores(fsys fs.FS) (p, e int) {
	count := func(list string) int {
		n := 0
		for _, part := range strings.Split(strings.TrimSpace(list), ",") {
			lo, hi, isRange := strings.Cut(part, "-")
			a, err1 := strconv.Atoi(lo)
			if err1 != nil {
				continue
			}
			if !isRange {
				n++
				continue
			}
			if b, err := strconv.Atoi(hi); err == nil {
				n += b - a + 1
			}
		}
		return n
	}
	core, _ := fs.ReadFile(fsys, "sys/devices/cpu_core/cpus")
	atom, _ := fs.ReadFile(fsys, "sys/devices/cpu_atom/cpus")
	if len(core) == 0 || len(atom) == 0 {
		return 0, 0
	}
	// Logical CPUs: P-cores are hyperthreaded on Alder Lake and later, E-cores
	// are not.
	pl, el := count(string(core)), count(string(atom))
	sib, _ := fs.ReadFile(fsys, "sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
	if strings.ContainsAny(string(sib), ",-") {
		pl /= 2
	}
	return pl, el
}

func parseOSRelease(s string) map[string]string {
	out := map[string]string{}
	for _, line := range strings.Split(s, "\n") {
		k, v, ok := strings.Cut(line, "=")
		if ok {
			out[k] = strings.Trim(v, `"'`)
		}
	}
	return out
}

func linuxPower(fsys fs.FS, read func(string) string) benchfmt.Power {
	var p benchfmt.Power
	p.Profile = read("/sys/firmware/acpi/platform_profile")
	entries, _ := fs.ReadDir(fsys, "sys/class/power_supply")
	hasBattery, online := false, false
	for _, e := range entries {
		dir := "/sys/class/power_supply/" + e.Name()
		switch read(dir + "/type") {
		case "Mains":
			if read(dir+"/online") == "1" {
				online = true
			}
		case "Battery":
			hasBattery = true
		}
	}
	switch {
	case online:
		p.Source = "ac"
	case hasBattery:
		p.Source = "battery"
	}
	return p
}

func linuxDisk(fsys fs.FS, read func(string) string) string {
	entries, _ := fs.ReadDir(fsys, "sys/block")
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, "loop") || strings.HasPrefix(name, "zram") || strings.HasPrefix(name, "ram") || strings.HasPrefix(name, "dm-") {
			continue
		}
		switch {
		case strings.HasPrefix(name, "nvme"):
			return "nvme"
		case strings.HasPrefix(name, "vd") || strings.HasPrefix(name, "xvd"):
			return "virtual"
		case read(path.Join("/sys/block", name, "queue/rotational")) == "1":
			return "hdd"
		case read(path.Join("/sys/block", name, "queue/rotational")) == "0":
			return "ssd"
		}
	}
	return "unknown"
}

// wslHost asks Windows, through WSL interop, what machine this is: inside
// WSL2 the Linux DMI files describe a Hyper-V VM, not the laptop.
func wslHost(ctx context.Context, m *benchfmt.Machine) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
		"$c=Get-CimInstance Win32_ComputerSystem; $p=Get-CimInstance Win32_ComputerSystemProduct; $e=Get-CimInstance Win32_SystemEnclosure;"+
			"Write-Output ('{0}|{1}|{2}|{3}|{4}' -f $c.Manufacturer,$c.SystemFamily,$p.Version,$c.Model,($e.ChassisTypes -join ','))").Output()
	if err != nil {
		return
	}
	f := strings.Split(strings.TrimSpace(string(out)), "|")
	if len(f) < 5 {
		return
	}
	m.System.Vendor = strings.TrimSpace(f[0])
	m.System.Product = strings.TrimSpace(f[3])
	m.System.Family = marketingName(strings.TrimSpace(f[1]), strings.TrimSpace(f[2]), m.System.Product)
	m.System.FormFactor = benchfmt.FormUnknown
	if t, err := strconv.Atoi(strings.Split(strings.TrimSpace(f[4]), ",")[0]); err == nil {
		m.System.FormFactor = FormFactorOfChassis(t)
		m.System.Chassis = ChassisName(t)
	}
}
