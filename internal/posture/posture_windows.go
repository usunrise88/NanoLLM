//go:build windows

package posture

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	kernel32                  = windows.NewLazySystemDLL("kernel32.dll")
	procSetProcessInformation = kernel32.NewProc("SetProcessInformation")
	procGetProcessInformation = kernel32.NewProc("GetProcessInformation")
)

const (
	processPowerThrottling = 4
	throttlingVersion      = 1
	executionSpeed         = 0x1
)

// powerThrottlingState is PROCESS_POWER_THROTTLING_STATE.
type powerThrottlingState struct {
	Version     uint32
	ControlMask uint32
	StateMask   uint32
}

// Apply puts the process below normal priority and in the EcoQoS class when low
// is set, and back to normal and out of it when it is not. Both work in either
// direction while the process runs.
func Apply(low bool) error {
	class := uint32(windows.NORMAL_PRIORITY_CLASS)
	state := uint32(0)
	if low {
		class, state = windows.BELOW_NORMAL_PRIORITY_CLASS, executionSpeed
	}
	proc := windows.CurrentProcess()
	if err := windows.SetPriorityClass(proc, class); err != nil {
		return err
	}
	st := powerThrottlingState{Version: throttlingVersion, ControlMask: executionSpeed, StateMask: state}
	// EcoQoS needs Windows 10 1709; an older system keeps the priority alone.
	if procSetProcessInformation.Find() == nil {
		r, _, err := procSetProcessInformation.Call(uintptr(proc), processPowerThrottling, uintptr(unsafe.Pointer(&st)), unsafe.Sizeof(st))
		if r == 0 {
			return err
		}
	}
	return nil
}

// Reapply is for the systems that give each thread its own priority; here the
// class covers the whole process.
func Reapply() error { return nil }

// Current reads the posture back.
func Current() State {
	var s State
	if class, err := windows.GetPriorityClass(windows.CurrentProcess()); err == nil {
		s.Low = class == windows.BELOW_NORMAL_PRIORITY_CLASS || class == windows.IDLE_PRIORITY_CLASS
	}
	if procGetProcessInformation.Find() == nil {
		var st powerThrottlingState
		st.Version = throttlingVersion
		r, _, _ := procGetProcessInformation.Call(uintptr(windows.CurrentProcess()), processPowerThrottling, uintptr(unsafe.Pointer(&st)), unsafe.Sizeof(st))
		s.Eco = r != 0 && st.StateMask&executionSpeed != 0
	}
	return s
}
