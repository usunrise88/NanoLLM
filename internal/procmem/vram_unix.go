//go:build !windows

package procmem

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// ProcessVRAM reads the memory nvidia-smi says one process holds on NVIDIA
// GPUs, summed over the cards. There is no shared part to report on Linux:
// the driver does not lend system memory to the GPU there. Other vendors have
// no per-process counter an ordinary user can read, and get ErrUnavailable.
func ProcessVRAM(pid int) (VRAM, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return VRAM{}, fmt.Errorf("%w: nvidia-smi: %v", ErrUnavailable, err)
	}
	return parseComputeApps(string(out), pid)
}

// parseComputeApps sums the used_memory (MiB) of the rows for pid.
func parseComputeApps(out string, pid int) (VRAM, error) {
	var sum int64
	for _, line := range strings.Split(out, "\n") {
		f := strings.Split(strings.TrimSpace(line), ",")
		if len(f) != 2 {
			continue
		}
		p, err := strconv.Atoi(strings.TrimSpace(f[0]))
		if err != nil || p != pid {
			continue
		}
		mib, err := strconv.ParseInt(strings.TrimSpace(f[1]), 10, 64)
		if err != nil {
			continue // "[N/A]" under some drivers and virtualisation
		}
		sum += mib << 20
	}
	return VRAM{Dedicated: sum, Source: SourceNvidiaSMI}, nil
}
