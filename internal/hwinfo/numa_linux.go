//go:build linux

package hwinfo

import "os"

// NUMATopology lists this machine's NUMA nodes and their CPUs; empty when the
// kernel exposes none.
func NUMATopology() []NUMANode { return numaNodes(os.DirFS("/")) }

// FirstThreads returns one logical CPU per physical core among cpus, the
// first thread of each: the CPUs to pin to so that no two threads share a
// core's vector units.
func FirstThreads(cpus []int) []int { return firstThreads(os.DirFS("/"), cpus) }
