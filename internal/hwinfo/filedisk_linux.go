package hwinfo

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/usunrise88/nanollm/benchfmt"
)

// FileDisk describes the block device that holds path, as the fields of a
// result's machine: its name, whether it spins, its readahead window and its
// largest request. The two sizes bound what one readahead request reads (the
// layer streamer's MADV_WILLNEED, docs/AUTOTUNE.md OPT-8), and the device is
// the one the weights come from, which the machine's disk type need not be.
func FileDisk(path string) []benchfmt.Field {
	var st unix.Stat_t
	if path == "" || unix.Stat(path, &st) != nil {
		return nil
	}
	dir := fmt.Sprintf("/sys/dev/block/%d:%d", unix.Major(st.Dev), unix.Minor(st.Dev))
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return nil
	}
	q := filepath.Join(real, "queue")
	if _, err := os.Stat(q); err != nil {
		q = filepath.Join(filepath.Dir(real), "queue") // a partition: its disk's queue
	}
	read := func(name string) string {
		b, _ := os.ReadFile(filepath.Join(q, name))
		return strings.TrimSpace(string(b))
	}
	var out []benchfmt.Field
	add := func(k, v string) {
		if v != "" {
			out = append(out, benchfmt.Field{Key: k, Value: v})
		}
	}
	add("model_disk", filepath.Base(real))
	add("model_disk_rotational", read("rotational"))
	add("model_read_ahead_kb", read("read_ahead_kb"))
	add("model_max_sectors_kb", read("max_sectors_kb"))
	return out
}
