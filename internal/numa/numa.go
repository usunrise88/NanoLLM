// Package numa places the process on the machine's memory nodes and CPUs.
//
// ggml allocates a model's weights wherever the thread that reads them runs,
// so on a machine with several NUMA nodes (an EPYC, a Threadripper, a dual
// Xeon) the whole model lands on one node, and every thread on the others
// reads it across the interconnect: the node's memory channels, not the
// machine's, set the decode speed. Interleaving the pages over all nodes
// (what `numactl --interleave=all` does) spreads that traffic over every
// channel.
//
// A memory policy and a CPU affinity are inherited across execve, and a
// thread started by the Go runtime inherits from the thread that creates it,
// which no program controls. So Enter sets them on one locked thread and then
// re-executes the binary: every thread of the new process image starts under
// the policy, with no numactl and no cgo.
package numa

import (
	"fmt"
	"strconv"
	"strings"
)

// Environment variables of the hand-off between the process that decides and
// its re-executed image.
const (
	// EnvPlace asks a process to place itself ("interleave", "node:1",
	// "interleave;cpus=0-3"); `nanollm bench` sets it on a worker.
	EnvPlace = "NANOLLM_PLACE"
	// EnvApplied marks a process that has been placed, so it does not
	// place itself again.
	EnvApplied = "NANOLLM_PLACED"
)

// Spec is a placement request.
type Spec struct {
	// Mode is "" or "off" (leave the kernel's policy), "interleave" (pages
	// round-robin over Nodes, or all nodes) or "node" (memory on Node only).
	Mode string
	Node int
	// CPUs pins the process to these logical CPUs; empty leaves its affinity.
	CPUs []int
}

// IsZero reports whether the spec asks for nothing.
func (s Spec) IsZero() bool { return (s.Mode == "" || s.Mode == "off") && len(s.CPUs) == 0 }

// Parse reads "interleave", "node:N" or "off", optionally followed by
// ";cpus=LIST".
func Parse(s string, parseCPUs func(string) []int) (Spec, error) {
	mode, rest, _ := strings.Cut(s, ";")
	var sp Spec
	switch {
	case mode == "" || mode == "off":
		sp.Mode = "off"
	case mode == "interleave":
		sp.Mode = "interleave"
	case strings.HasPrefix(mode, "node:"):
		n, err := strconv.Atoi(strings.TrimPrefix(mode, "node:"))
		if err != nil || n < 0 {
			return sp, fmt.Errorf("numa %q: want node:N", mode)
		}
		sp.Mode, sp.Node = "node", n
	default:
		return sp, fmt.Errorf("numa %q: want off, interleave or node:N", mode)
	}
	if list, ok := strings.CutPrefix(rest, "cpus="); ok {
		if sp.CPUs = parseCPUs(list); sp.CPUs == nil {
			return sp, fmt.Errorf("numa: cpus %q is not a cpu list", list)
		}
	}
	return sp, nil
}
