//go:build linux

package memlock

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// fileMappings lists the mappings of path in this process, from
// /proc/self/maps. A file that is not mapped (yet) has none.
func fileMappings(path string) ([]mapping, error) {
	want, err := filepath.EvalSymlinks(path)
	if err != nil {
		return nil, err
	}
	f, err := os.Open("/proc/self/maps")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []mapping
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		// start-end perms offset dev inode path
		fields := strings.Fields(sc.Text())
		if len(fields) < 6 || fields[5] != want {
			continue
		}
		lo, hi, ok := strings.Cut(fields[0], "-")
		if !ok {
			continue
		}
		start, err1 := strconv.ParseUint(lo, 16, 64)
		end, err2 := strconv.ParseUint(hi, 16, 64)
		off, err3 := strconv.ParseInt(fields[2], 16, 64)
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}
		out = append(out, mapping{start, end, off})
	}
	return out, sc.Err()
}
