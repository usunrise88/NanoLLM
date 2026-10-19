//go:build !linux

package hwinfo

// NUMATopology is empty off Linux: pinning and memory policy are Linux-only.
func NUMATopology() []NUMANode { return nil }

// FirstThreads returns cpus unchanged where the topology is not read.
func FirstThreads(cpus []int) []int { return cpus }
