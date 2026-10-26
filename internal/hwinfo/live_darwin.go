//go:build darwin

package hwinfo

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sys/unix"
)

func memoryNow() (total, available int64) {
	b, err := unix.SysctlUint64("hw.memsize")
	if err != nil {
		return 0, 0
	}
	total = int64(b)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "vm_stat").Output()
	if err != nil {
		return total, 0
	}
	return total, parseVMStat(string(out))
}

// parseVMStat is the memory macOS can give to a program without paging:
// free, inactive and speculative pages ("Pages free:  1234.").
func parseVMStat(s string) int64 {
	page := int64(16384)
	if i := strings.Index(s, "page size of "); i >= 0 {
		if n, err := strconv.ParseInt(strings.Fields(s[i+len("page size of "):])[0], 10, 64); err == nil {
			page = n
		}
	}
	var pages int64
	for _, line := range strings.Split(s, "\n") {
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch k {
		case "Pages free", "Pages inactive", "Pages speculative":
			if n, err := strconv.ParseInt(strings.TrimSuffix(strings.TrimSpace(v), "."), 10, 64); err == nil {
				pages += n
			}
		}
	}
	return pages * page
}

func cpuBusy(context.Context, time.Duration) float64 { return -1 }
