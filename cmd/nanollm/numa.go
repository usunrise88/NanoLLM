package main

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/numa"
)

// placeProcess places this process on the machine's NUMA nodes and CPUs and
// re-executes it under that placement (internal/numa). raw is a numa.Parse
// spec, or "auto": interleave over every node when the machine has several,
// nothing otherwise. It returns when there is nothing to do, and never
// returns after a successful placement: the process image is replaced.
func placeProcess(raw string) error {
	topo := hwinfo.NUMATopology()
	if raw == "auto" {
		raw = "off"
		if len(topo) > 1 {
			raw = "interleave"
		}
	}
	sp, err := numa.Parse(raw, hwinfo.ParseCPUList)
	if err != nil {
		return err
	}
	var nodes []int
	for _, n := range topo {
		if len(sp.CPUs) == 0 || intersects(n.CPUs, sp.CPUs) {
			nodes = append(nodes, n.ID)
		}
	}
	if sp.Mode == "interleave" && len(nodes) < 2 {
		sp.Mode = "off" // one node: nothing to spread over
	}
	if sp.Mode == "node" && len(topo) == 0 {
		return fmt.Errorf("numa: this machine has no NUMA nodes to place on")
	}
	return numa.Enter(sp, nodes)
}

// placeFromEnv honours $NANOLLM_PLACE, which `nanollm bench` sets on its
// workers; a failure is logged and the process runs unplaced.
func placeFromEnv() {
	if raw := os.Getenv(numa.EnvPlace); raw != "" {
		if err := placeProcess(raw); err != nil {
			slog.Warn("NUMA placement failed; running without it", "place", raw, "err", err)
		}
	}
}

func intersects(a, b []int) bool {
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
