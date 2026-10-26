package hwinfo

import (
	"context"
	"runtime"
	"sync"
)

// PhysicalCores is the number of physical cores this process may run on, the
// efficiency cores of a hybrid CPU included (DefaultThreads counts only the
// performance cores there). The optimal mode takes half of them. It does not
// run the slow parts of Collect, so it is cheap to call.
func PhysicalCores() int {
	physicalOnce.Do(func() { physical = max(1, detectPhysicalCores()) })
	return physical
}

var (
	physicalOnce sync.Once
	physical     int
)

func detectPhysicalCores() int {
	if n := allowedPhysicalCores(); n > 0 {
		return n
	}
	if n := quickCores(); n > 0 {
		return n
	}
	m := collect(context.Background())
	if m.CPU.Cores > 0 {
		return m.CPU.Cores
	}
	return runtime.NumCPU()
}

// CPUName is the processor's model as the OS names it, cleaned of decoration
// (CleanCPUName). It reads only the one value, without the slow parts of
// Collect.
func CPUName() string { return CleanCPUName(cpuNameRaw()) }
