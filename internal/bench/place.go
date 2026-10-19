package bench

import (
	"fmt"
	"strconv"
	"strings"
)

// NUMA policies a bench worker can run under.
//
//	off          the kernel's default (first touch, with automatic balancing)
//	interleave   pages spread round-robin over the nodes the worker may use
//	node:N       the worker's CPUs and memory on node N alone
type NUMA struct {
	Policy string // "", off, interleave, node
	Node   int
}

// ParseNUMA reads a --numa value.
func ParseNUMA(s string) (NUMA, error) {
	switch {
	case s == "" || s == "off":
		return NUMA{Policy: "off"}, nil
	case s == "interleave":
		return NUMA{Policy: "interleave"}, nil
	case strings.HasPrefix(s, "node:"):
		n, err := strconv.Atoi(strings.TrimPrefix(s, "node:"))
		if err != nil || n < 0 {
			return NUMA{}, fmt.Errorf("numa %q: want node:N", s)
		}
		return NUMA{Policy: "node", Node: n}, nil
	}
	return NUMA{}, fmt.Errorf("numa %q: want off, interleave or node:N", s)
}

func (n NUMA) String() string {
	if n.Policy == "node" {
		return "node:" + strconv.Itoa(n.Node)
	}
	return n.Policy
}

// Placement is where a worker runs. The zero value is the whole machine
// under the kernel's defaults.
type Placement struct {
	// NUMA is the memory policy ("" is off).
	NUMA string `json:"numa,omitempty"`
	// CPUs is the cpu list the worker is pinned to ("" is every CPU).
	CPUs string `json:"cpus,omitempty"`
}

// Spec is the placement in the form a worker's $NANOLLM_PLACE takes
// (numa.Parse): "interleave", "node:1;cpus=4-7", …
func (p Placement) Spec() string {
	s := p.NUMA
	if s == "" {
		s = "off"
	}
	if p.CPUs != "" {
		s += ";cpus=" + p.CPUs
	}
	return s
}

// IsZero reports whether the placement leaves the worker unconstrained.
func (p Placement) IsZero() bool { return (p.NUMA == "" || p.NUMA == "off") && p.CPUs == "" }

func sharesCPU(a, b []int) bool {
	in := map[int]bool{}
	for _, c := range a {
		in[c] = true
	}
	for _, c := range b {
		if in[c] {
			return true
		}
	}
	return false
}
