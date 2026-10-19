//go:build linux

package hwinfo

import (
	"os"

	"golang.org/x/sys/unix"
)

// allowedPhysicalCores counts the physical cores among the CPUs in this
// process's affinity mask.
func allowedPhysicalCores() int {
	var set unix.CPUSet
	if err := unix.SchedGetaffinity(0, &set); err != nil {
		return 0
	}
	var cpus []int
	for i := 0; i < len(set)*64; i++ {
		if set.IsSet(i) {
			cpus = append(cpus, i)
		}
	}
	return physicalCores(os.DirFS("/"), cpus)
}
