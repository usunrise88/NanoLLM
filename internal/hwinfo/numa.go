package hwinfo

import (
	"io/fs"
	"sort"
	"strconv"
	"strings"
)

// NUMANode is a memory node and the logical CPUs local to it.
type NUMANode struct {
	ID   int
	CPUs []int
}

// numaNodes reads the nodes from sysfs. A machine without NUMA (or without
// sysfs) yields no nodes.
func numaNodes(fsys fs.FS) []NUMANode {
	ents, err := fs.ReadDir(fsys, "sys/devices/system/node")
	if err != nil {
		return nil
	}
	var out []NUMANode
	for _, e := range ents {
		id, err := strconv.Atoi(strings.TrimPrefix(e.Name(), "node"))
		if err != nil || !strings.HasPrefix(e.Name(), "node") {
			continue
		}
		b, err := fs.ReadFile(fsys, "sys/devices/system/node/"+e.Name()+"/cpulist")
		if err != nil {
			continue
		}
		out = append(out, NUMANode{ID: id, CPUs: ParseCPUList(strings.TrimSpace(string(b)))})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// ParseCPUList reads the kernel's cpu list syntax: "0-3,16-19".
func ParseCPUList(s string) []int {
	var out []int
	for part := range strings.SplitSeq(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		lo, hi, isRange := strings.Cut(part, "-")
		a, err := strconv.Atoi(lo)
		if err != nil {
			return nil
		}
		b := a
		if isRange {
			if b, err = strconv.Atoi(hi); err != nil || b < a {
				return nil
			}
		}
		for i := a; i <= b; i++ {
			out = append(out, i)
		}
	}
	return out
}

// FormatCPUList is ParseCPUList's inverse.
func FormatCPUList(cpus []int) string {
	s := append([]int(nil), cpus...)
	sort.Ints(s)
	var parts []string
	for i := 0; i < len(s); {
		j := i
		for j+1 < len(s) && s[j+1] == s[j]+1 {
			j++
		}
		if j > i {
			parts = append(parts, strconv.Itoa(s[i])+"-"+strconv.Itoa(s[j]))
		} else {
			parts = append(parts, strconv.Itoa(s[i]))
		}
		i = j + 1
	}
	return strings.Join(parts, ",")
}

// firstThreads picks one logical CPU per physical core among cpus: the lowest
// numbered, which is how the kernel numbers SMT siblings on Intel and AMD
// alike (the first thread of every core, then the second).
func firstThreads(fsys fs.FS, cpus []int) []int {
	seen := map[string]bool{}
	var out []int
	s := append([]int(nil), cpus...)
	sort.Ints(s)
	for _, cpu := range s {
		base := "sys/devices/system/cpu/cpu" + strconv.Itoa(cpu) + "/topology/"
		core, err := fs.ReadFile(fsys, base+"core_id")
		if err != nil {
			out = append(out, cpu)
			continue
		}
		pkg, _ := fs.ReadFile(fsys, base+"physical_package_id")
		key := strings.TrimSpace(string(pkg)) + "/" + strings.TrimSpace(string(core))
		if !seen[key] {
			seen[key] = true
			out = append(out, cpu)
		}
	}
	return out
}
