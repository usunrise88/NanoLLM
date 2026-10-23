//go:build windows

package hwinfo

import (
	"context"
	"encoding/binary"
	"os/exec"
	"regexp"
	"strings"
	"time"
	"unsafe"

	"golang.org/x/sys/cpu"
	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"

	"github.com/usunrise88/nanollm/benchfmt"
)

var (
	kernel32                      = windows.NewLazySystemDLL("kernel32.dll")
	procGetSystemFirmwareTable    = kernel32.NewProc("GetSystemFirmwareTable")
	procGetLogicalProcessorInfoEx = kernel32.NewProc("GetLogicalProcessorInformationEx")
	procGlobalMemoryStatusEx      = kernel32.NewProc("GlobalMemoryStatusEx")
	procGetSystemPowerStatus      = kernel32.NewProc("GetSystemPowerStatus")
)

func collect(ctx context.Context) benchfmt.Machine {
	var m benchfmt.Machine
	m.OS.Name = "windows"
	v := windows.RtlGetVersion()
	m.OS.Version = map[bool]string{true: "11", false: "10"}[v.BuildNumber >= 22000]
	m.OS.KernelMM = itoa(int(v.MajorVersion)) + "." + itoa(int(v.BuildNumber))

	if k, err := registry.OpenKey(registry.LOCAL_MACHINE, `HARDWARE\DESCRIPTION\System\CentralProcessor\0`, registry.QUERY_VALUE); err == nil {
		m.CPU.Model, _, _ = k.GetStringValue("ProcessorNameString")
		vendor, _, _ := k.GetStringValue("VendorIdentifier")
		m.CPU.Vendor = map[string]string{"GenuineIntel": "Intel", "AuthenticAMD": "AMD"}[vendor]
		_ = k.Close()
	}
	m.CPU.Cores, m.CPU.Threads, m.CPU.PCores, m.CPU.ECores = processorCores()
	m.CPU.ISA = isa()

	var ms struct {
		Length                                     uint32
		MemoryLoad                                 uint32
		TotalPhys, AvailPhys, TotalPage, AvailPage uint64
		TotalVirtual, AvailVirtual, AvailExtended  uint64
	}
	ms.Length = uint32(unsafe.Sizeof(ms))
	if r, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&ms))); r != 0 {
		m.Memory.TotalBytes = int64(ms.TotalPhys)
	}

	if info, ok := smbios(); ok {
		m.System.Vendor, m.System.Product = info.vendor, info.product
		m.System.Family = marketingName(info.family, info.version, info.product)
		m.System.FormFactor = FormFactorOfChassis(info.chassis)
		m.System.Chassis = ChassisName(info.chassis)
		m.Memory.SpeedMTS = info.memMTS
		if looksVirtual(info.vendor, info.product) {
			m.System.FormFactor = benchfmt.FormVM
		}
		if cleanSystem(&m.System, info.boardVendor, info.boardProduct) {
			m.Extra = append(m.Extra, benchfmt.Field{Key: "system_name_from", Value: "baseboard"})
		}
	}

	var ps struct {
		ACLineStatus, BatteryFlag, BatteryLifePercent, SystemStatusFlag byte
		BatteryLifeTime, BatteryFullLifeTime                            uint32
	}
	if r, _, _ := procGetSystemPowerStatus.Call(uintptr(unsafe.Pointer(&ps))); r != 0 {
		switch ps.ACLineStatus {
		case 1:
			m.Power.Source = "ac"
		case 0:
			m.Power.Source = "battery"
		}
	}
	m.Power.Profile = powerScheme(ctx)
	m.Disk.Type = windowsDisk(ctx)
	return m
}

// smbios reads the raw SMBIOS table; no administrator rights are needed.
func smbios() (smbiosInfo, bool) {
	const rsmb = 'R'<<24 | 'S'<<16 | 'M'<<8 | 'B'
	n, _, _ := procGetSystemFirmwareTable.Call(rsmb, 0, 0, 0)
	if n == 0 {
		return smbiosInfo{}, false
	}
	buf := make([]byte, n)
	if r, _, _ := procGetSystemFirmwareTable.Call(rsmb, 0, uintptr(unsafe.Pointer(&buf[0])), n); r == 0 {
		return smbiosInfo{}, false
	}
	return parseSMBIOS(buf), true
}

// processorCores walks RelationProcessorCore records: one per physical core,
// with the logical processors in its group mask and an efficiency class that
// is higher for performance cores on hybrid parts.
func processorCores() (cores, threads, p, e int) {
	const relationProcessorCore = 0
	var size uint32
	_, _, _ = procGetLogicalProcessorInfoEx.Call(relationProcessorCore, 0, uintptr(unsafe.Pointer(&size)))
	if size == 0 {
		return
	}
	buf := make([]byte, size)
	if r, _, _ := procGetLogicalProcessorInfoEx.Call(relationProcessorCore, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size))); r == 0 {
		return
	}
	type core struct{ class, threads int }
	var list []core
	maxClass := 0
	for off := 0; off+8 <= int(size); {
		rel := binary.LittleEndian.Uint32(buf[off:])
		recSize := int(binary.LittleEndian.Uint32(buf[off+4:]))
		if recSize == 0 || off+recSize > int(size) {
			break
		}
		if rel == relationProcessorCore && recSize >= 8+24 {
			// PROCESSOR_RELATIONSHIP: Flags, EfficiencyClass, Reserved[20],
			// GroupCount (u16), GroupMask[] {KAFFINITY Mask; WORD Group; WORD Reserved[3]}.
			body := buf[off+8:]
			class := int(body[1])
			groups := int(binary.LittleEndian.Uint16(body[22:]))
			t := 0
			for g := 0; g < groups; g++ {
				at := 24 + g*16
				if at+8 > len(body) {
					break
				}
				mask := binary.LittleEndian.Uint64(body[at:])
				for ; mask != 0; mask &= mask - 1 {
					t++
				}
			}
			list = append(list, core{class, t})
			maxClass = max(maxClass, class)
		}
		off += recSize
	}
	for _, c := range list {
		cores++
		threads += c.threads
		if maxClass > 0 {
			if c.class == maxClass {
				p++
			} else {
				e++
			}
		}
	}
	return
}

func isa() []string {
	var out []string
	for _, f := range []struct {
		name string
		ok   bool
	}{
		{"avx", cpu.X86.HasAVX}, {"avx2", cpu.X86.HasAVX2}, {"fma", cpu.X86.HasFMA},
		{"avx512f", cpu.X86.HasAVX512F}, {"avx512_vnni", cpu.X86.HasAVX512VNNI},
		{"avx512_bf16", cpu.X86.HasAVX512BF16}, {"avx_vnni", cpu.X86.HasAVXVNNI},
		{"asimd", cpu.ARM64.HasASIMD}, {"dotprod", cpu.ARM64.HasASIMDDP}, {"i8mm", cpu.ARM64.HasI8MM}, {"sve", cpu.ARM64.HasSVE},
	} {
		if f.ok {
			out = append(out, f.name)
		}
	}
	return out
}

var schemeName = regexp.MustCompile(`\(([^)]+)\)\s*$`)

func powerScheme(ctx context.Context) string {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "powercfg", "/getactivescheme").Output()
	if err != nil {
		return ""
	}
	if m := schemeName.FindStringSubmatch(strings.TrimSpace(string(out))); m != nil {
		return m[1]
	}
	return ""
}

func windowsDisk(ctx context.Context) string {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
		"$d=Get-PhysicalDisk | Sort-Object DeviceId | Select-Object -First 1; Write-Output ('{0}|{1}' -f $d.BusType,$d.MediaType)").Output()
	if err != nil {
		return "unknown"
	}
	bus, media, _ := strings.Cut(strings.TrimSpace(string(out)), "|")
	switch {
	case strings.EqualFold(bus, "NVMe"):
		return "nvme"
	case strings.EqualFold(media, "SSD"):
		return "ssd"
	case strings.EqualFold(media, "HDD"):
		return "hdd"
	}
	return "unknown"
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// quickCores is the physical core count without the PowerShell and powercfg
// calls the rest of collect makes.
func quickCores() int {
	cores, _, _, _ := processorCores()
	return cores
}

func cpuNameRaw() string {
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, `HARDWARE\DESCRIPTION\System\CentralProcessor\0`, registry.QUERY_VALUE)
	if err != nil {
		return ""
	}
	defer k.Close()
	s, _, _ := k.GetStringValue("ProcessorNameString")
	return s
}
