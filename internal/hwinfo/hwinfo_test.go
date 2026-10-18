package hwinfo

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/usunrise88/nanollm/benchfmt"
)

func file(s string) *fstest.MapFile { return &fstest.MapFile{Data: []byte(s)} }

func TestLinuxLaptop(t *testing.T) {
	cpuinfo := ""
	for i := 0; i < 20; i++ {
		core := i
		if i < 12 {
			core = i / 2 // P-cores are hyperthreaded
		}
		cpuinfo += "processor\t: " + itoa(i) + "\nvendor_id\t: GenuineIntel\nmodel name\t: 12th Gen Intel(R) Core(TM) i7-12700H\n" +
			"flags\t\t: fpu sse avx avx2 fma f16c avx_vnni\nphysical id\t: 0\ncore id\t\t: " + itoa(core) + "\n\n"
	}
	fsys := fstest.MapFS{
		"proc/cpuinfo":                                              file(cpuinfo),
		"proc/meminfo":                                              file("MemTotal:       32617148 kB\nMemFree: 1 kB\n"),
		"sys/class/dmi/id/sys_vendor":                               file("LENOVO\n"),
		"sys/class/dmi/id/product_name":                             file("21HMCTO1WW\n"),
		"sys/class/dmi/id/product_version":                          file("ThinkPad X1 Carbon Gen 11\n"),
		"sys/class/dmi/id/product_family":                           file("ThinkPad X1 Carbon Gen 11\n"),
		"sys/class/dmi/id/product_serial":                           file("MUST-NOT-BE-READ\n"),
		"sys/class/dmi/id/product_uuid":                             file("MUST-NOT-BE-READ\n"),
		"sys/class/dmi/id/chassis_type":                             file("10\n"),
		"sys/devices/cpu_core/cpus":                                 file("0-11\n"),
		"sys/devices/cpu_atom/cpus":                                 file("12-19\n"),
		"sys/devices/system/cpu/cpu0/topology/thread_siblings_list": file("0-1\n"),
		"etc/os-release":                                            file("ID=ubuntu\nVERSION_ID=\"24.04\"\n"),
		"proc/sys/kernel/osrelease":                                 file("6.8.0-45-generic\n"),
		"sys/class/power_supply/AC/type":                            file("Mains\n"),
		"sys/class/power_supply/AC/online":                          file("1\n"),
		"sys/class/power_supply/BAT0/type":                          file("Battery\n"),
		"sys/firmware/acpi/platform_profile":                        file("balanced\n"),
		"sys/block/nvme0n1/queue/rotational":                        file("0\n"),
	}
	m := linuxProfile(fsys)
	if m.CPU.Threads != 20 || m.CPU.Cores != 14 || m.CPU.PCores != 6 || m.CPU.ECores != 8 {
		t.Errorf("cpu %+v", m.CPU)
	}
	if strings.Join(m.CPU.ISA, ",") != "avx,avx2,fma,f16c,avx_vnni" {
		t.Errorf("isa %v", m.CPU.ISA)
	}
	if m.Memory.TotalBytes != 32617148<<10 {
		t.Errorf("memory %d", m.Memory.TotalBytes)
	}
	if m.System.Vendor != "LENOVO" || m.System.Family != "ThinkPad X1 Carbon Gen 11" || m.System.FormFactor != benchfmt.FormLaptop {
		t.Errorf("system %+v", m.System)
	}
	if m.OS.Name != "ubuntu" || m.OS.Version != "24.04" || m.OS.KernelMM != "6.8" || m.OS.WSL {
		t.Errorf("os %+v", m.OS)
	}
	if m.Power.Source != "ac" || m.Power.Profile != "balanced" || m.Disk.Type != "nvme" {
		t.Errorf("power %+v disk %+v", m.Power, m.Disk)
	}
	b, _ := json.Marshal(m)
	if strings.Contains(string(b), "MUST-NOT-BE-READ") {
		t.Fatal("a serial or UUID leaked into the profile")
	}
}

func TestLinuxVMAndWSL(t *testing.T) {
	m := linuxProfile(fstest.MapFS{
		"proc/cpuinfo":                  file("processor : 0\nmodel name : AMD EPYC 7351P 16-Core Processor\nflags : avx2 hypervisor\n"),
		"sys/class/dmi/id/sys_vendor":   file("QEMU\n"),
		"sys/class/dmi/id/chassis_type": file("1\n"),
		"proc/sys/kernel/osrelease":     file("5.15.153.1-microsoft-standard-WSL2\n"),
	})
	if m.System.FormFactor != benchfmt.FormVM || !m.OS.WSL {
		t.Fatalf("%+v %+v", m.System, m.OS)
	}
}

// smbiosStruct builds one SMBIOS structure: formatted area, then strings.
func smbiosStruct(typ byte, formatted []byte, strs ...string) []byte {
	b := append([]byte{typ, byte(4 + len(formatted)), 0, 0}, formatted...)
	if len(strs) == 0 {
		return append(b, 0, 0)
	}
	for _, s := range strs {
		b = append(append(b, s...), 0)
	}
	return append(b, 0)
}

func TestParseSMBIOS(t *testing.T) {
	// Type 1: manufacturer=1, product=2, version=3, serial=4 (never read),
	// uuid (16 bytes), wake-up, sku=5, family=6.
	t1 := make([]byte, 0x1B-4)
	t1[0], t1[1], t1[2], t1[3] = 1, 2, 3, 4
	t1[0x19-4], t1[0x1A-4] = 5, 6
	copy(t1[4:20], "UUID-NOT-READ!!!")
	sys := smbiosStruct(1, t1, "ASUSTeK COMPUTER INC.", "ROG Zephyrus G14 GA402RJ", "1.0", "SERIAL-NOT-READ", "SKU", "ROG Zephyrus G14")
	t3 := make([]byte, 5)
	t3[1] = 0x80 | 10 // lock bit + notebook
	enc := smbiosStruct(3, t3, "ASUS")
	t17 := make([]byte, 0x22-4)
	binary.LittleEndian.PutUint16(t17[0x15-4:], 4800)
	binary.LittleEndian.PutUint16(t17[0x20-4:], 4800)
	mem := smbiosStruct(17, t17)
	end := smbiosStruct(127, nil)
	table := append(append(append(sys, enc...), mem...), end...)
	raw := append([]byte{0, 3, 4, 0, 0, 0, 0, 0}, table...)
	binary.LittleEndian.PutUint32(raw[4:], uint32(len(table)))

	info := parseSMBIOS(raw)
	if info.vendor != "ASUSTeK COMPUTER INC." || info.product != "ROG Zephyrus G14 GA402RJ" || info.family != "ROG Zephyrus G14" {
		t.Fatalf("%+v", info)
	}
	if info.chassis != 10 || info.memMTS != 4800 {
		t.Fatalf("%+v", info)
	}
	if marketingName(info.family, info.version, info.product) != "ROG Zephyrus G14" {
		t.Fatal("marketing name")
	}
	if parseSMBIOS(raw[:5]) != (smbiosInfo{}) || parseSMBIOS(append(raw[:8:8], 1, 200)) != (smbiosInfo{}) {
		t.Fatal("truncated tables must not panic or invent data")
	}
}

func TestCleanCPUName(t *testing.T) {
	for in, want := range map[string]string{
		"12th Gen Intel(R) Core(TM) i7-12700H":      "Intel Core i7-12700H",
		"Intel(R) Core(TM) i5-1035G1 CPU @ 1.00GHz": "Intel Core i5-1035G1",
		"AMD EPYC 7351P 16-Core Processor":          "AMD EPYC 7351P",
		"AMD Ryzen AI 9 HX 370 w/ Radeon 890M":      "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
		"Apple M3 Pro":                              "Apple M3 Pro",
	} {
		if got := CleanCPUName(in); got != want {
			t.Errorf("%q → %q, want %q", in, got, want)
		}
	}
}

func TestNvidiaSMI(t *testing.T) {
	rows := parseNvidiaSMI("NVIDIA GeForce RTX 3070 Ti Laptop GPU, 560.94, 125.00\n")
	if len(rows) != 1 || rows[0].powerW != 125 || majorOf(rows[0].driver) != "560" {
		t.Fatalf("%+v", rows)
	}
}

// TestCollectHere runs the real collector on the machine the tests run on:
// it must not fail, and must produce a profile benchfmt accepts.
func TestCollectHere(t *testing.T) {
	m := Collect(context.Background(), nil)
	if m.CPU.Threads == 0 && m.CPU.Model == "" {
		t.Skip("this platform exposes no CPU information")
	}
	if m.System.FormFactor == "" || m.OS.Name == "" {
		t.Fatalf("%+v", m)
	}
	t.Logf("%+v", m)
}

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}
