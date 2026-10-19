package bench

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/usunrise88/nanollm/internal/hwinfo"
)

// Profile is a hardware class to measure on this machine, so an optimization
// is judged on the machine it is meant for and on a much weaker one (the
// project aims to run on nearly any hardware): where the worker may run,
// which preset to ask for, and whether the disk is read cold.
type Profile struct {
	Name   string
	Place  Placement
	Preset string
	Cold   bool
}

// Profile names.
const (
	ProfileStrong = "strong"
	ProfileWeak   = "weak"
)

// Weak profile: four cores of one NUMA node, 2 GB of RAM, a cold disk.
const (
	weakCores  = 4
	weakPreset = "ram-2g"
	// strongPreset is large enough to be M6 (everything resident) on the
	// E4B.
	strongPreset = "ram-16g"
)

// ParseProfile reads --profile: strong, weak, or "<cpus>:<preset>" such as
// "0-3:ram-1g" (a cold run on those CPUs and their node).
func ParseProfile(s string) (Profile, error) {
	nodes := hwinfo.NUMATopology()
	switch s {
	case ProfileStrong:
		p := Profile{Name: s, Preset: strongPreset}
		if len(nodes) > 1 {
			p.Place.NUMA = "interleave"
		}
		return p, nil
	case ProfileWeak:
		p := Profile{Name: s, Preset: weakPreset, Cold: true}
		if len(nodes) == 0 {
			return p, nil // nothing to pin to; the preset's budget still applies
		}
		cores := hwinfo.FirstThreads(nodes[0].CPUs)
		if len(cores) > weakCores {
			cores = cores[:weakCores]
		}
		p.Place = Placement{CPUs: hwinfo.FormatCPUList(cores)}
		if len(nodes) > 1 {
			p.Place.NUMA = "node:" + strconv.Itoa(nodes[0].ID)
		}
		return p, nil
	}
	cpus, preset, ok := strings.Cut(s, ":")
	if !ok || preset == "" || hwinfo.ParseCPUList(cpus) == nil {
		return Profile{}, fmt.Errorf("profile %q: want strong, weak or <cpus>:<preset>, e.g. 0-3:ram-2g", s)
	}
	p := Profile{Name: s, Preset: preset, Cold: true, Place: Placement{CPUs: hwinfo.FormatCPUList(hwinfo.ParseCPUList(cpus))}}
	if len(nodes) > 1 {
		// Memory follows the CPUs when they sit on one node.
		var on []int
		for _, n := range nodes {
			if sharesCPU(n.CPUs, hwinfo.ParseCPUList(cpus)) {
				on = append(on, n.ID)
			}
		}
		if len(on) == 1 {
			p.Place.NUMA = "node:" + strconv.Itoa(on[0])
		} else {
			p.Place.NUMA = "interleave"
		}
	}
	return p, nil
}
