//go:build windows

package hwinfo

import (
	"strings"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	powrprof                     = windows.NewLazySystemDLL("powrprof.dll")
	procPowerGetActiveScheme     = powrprof.NewProc("PowerGetActiveScheme")
	procPowerGetEffectiveOverlay = powrprof.NewProc("PowerGetEffectiveOverlayScheme")
	procPowerReadFriendlyName    = powrprof.NewProc("PowerReadFriendlyName")
)

const (
	// The Windows "Power saver" plan and the overlays of the power slider.
	guidPowerSaver     = "a1841308-3541-4fab-bc81-f71556f20b4a"
	guidBestEfficiency = "961cc777-2547-4f9d-8174-7d86181b8a7a"
	guidBetterBattery  = "3af9b8d9-7c97-431d-ad78-34a8bfea439f"
	guidBestPerf       = "ded574b5-45a0-4f42-8737-46345c09c238"
)

func guidString(g *windows.GUID) string { return strings.ToLower(g.String()[1 : len(g.String())-1]) }

// friendlyName is the name of a power plan as the OS shows it, in its language.
func friendlyName(g *windows.GUID) string {
	if procPowerReadFriendlyName.Find() != nil {
		return ""
	}
	var size uint32
	// A first call without a buffer says how big it must be.
	procPowerReadFriendlyName.Call(0, uintptr(unsafe.Pointer(g)), 0, 0, 0, uintptr(unsafe.Pointer(&size)))
	if size < 2 || size > 1<<12 {
		return ""
	}
	buf := make([]uint16, size/2+1)
	if r, _, _ := procPowerReadFriendlyName.Call(0, uintptr(unsafe.Pointer(g)), 0, 0, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&size))); r != 0 {
		return ""
	}
	return windows.UTF16ToString(buf)
}

// ReadPowerMode reads the active power plan and the effective slider overlay
// of Windows 10 and 11. The plan names are localized, the GUIDs are not.
func ReadPowerMode() PowerMode {
	var pm PowerMode
	var plan *windows.GUID
	if r, _, _ := procPowerGetActiveScheme.Call(0, uintptr(unsafe.Pointer(&plan))); r == 0 && plan != nil {
		g := guidString(plan)
		name := friendlyName(plan)
		windows.LocalFree(windows.Handle(unsafe.Pointer(plan)))
		pm.Name = g
		if name != "" {
			pm.Name = name + " (" + g + ")"
		}
		pm.Saver = g == guidPowerSaver
		if !pm.Saver {
			standard := windows.GUID{Data1: 0xa1841308, Data2: 0x3541, Data3: 0x4fab, Data4: [8]byte{0xbc, 0x81, 0xf7, 0x15, 0x56, 0xf2, 0x0b, 0x4a}}
			pm.Saver = sameSaverName(name, friendlyName(&standard))
		}
	}
	if procPowerGetEffectiveOverlay.Find() == nil {
		var ov windows.GUID
		if r, _, _ := procPowerGetEffectiveOverlay.Call(uintptr(unsafe.Pointer(&ov))); r == 0 {
			switch g := guidString(&ov); g {
			case guidBestEfficiency, guidBetterBattery:
				pm.Saver = true
				pm.Name = g
			}
		}
	}
	return pm
}
