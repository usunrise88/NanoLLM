//go:build linux

package bench

import (
	"fmt"

	"github.com/usunrise88/nanollm/internal/hwinfo"
)

// Resolve completes a placement against the machine's topology: node:N pins
// to the CPUs of node N. It also returns the thread count a pinned worker
// should default to, the physical cores among its CPUs (0 when the placement
// does not pin; this process sees the whole machine, the worker will not).
func (p Placement) Resolve() (Placement, int, error) {
	pol, err := ParseNUMA(p.NUMA)
	if err != nil {
		return p, 0, err
	}
	nodes := hwinfo.NUMATopology()
	if (pol.Policy != "off" || p.CPUs != "") && len(nodes) == 0 {
		return p, 0, fmt.Errorf("bench: no NUMA topology to place on")
	}
	cpus := hwinfo.ParseCPUList(p.CPUs)
	if p.CPUs != "" && cpus == nil {
		return p, 0, fmt.Errorf("bench: cpus %q: not a cpu list", p.CPUs)
	}
	if pol.Policy == "node" {
		found := false
		for _, n := range nodes {
			if n.ID == pol.Node {
				found = true
				if cpus == nil {
					cpus = n.CPUs
				}
			}
		}
		if !found {
			return p, 0, fmt.Errorf("bench: no NUMA node %d", pol.Node)
		}
	}
	out := Placement{NUMA: pol.String(), CPUs: hwinfo.FormatCPUList(cpus)}
	if pol.Policy == "off" {
		out.NUMA = ""
	}
	threads := 0
	if len(cpus) > 0 {
		threads = len(hwinfo.FirstThreads(cpus))
	}
	return out, threads, nil
}
