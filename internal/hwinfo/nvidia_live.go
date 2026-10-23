package hwinfo

import (
	"context"
	"os/exec"
	"strings"
	"time"
)

// nvidiaLive reads every NVIDIA GPU nvidia-smi knows; none when it is absent
// or fails. The throttle field was renamed from clocks_throttle_reasons to
// clocks_event_reasons in newer drivers, so both are tried.
func nvidiaLive(ctx context.Context) []GPULive {
	return NVIDIALive(ctx)
}

// NVIDIALive reads every NVIDIA GPU nvidia-smi knows: its state, driver and
// compute capability. None when nvidia-smi is absent or fails.
func NVIDIALive(ctx context.Context) []GPULive {
	ctx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	var gpus []GPULive
	// Newer drivers know compute_cap and clocks_event_reasons; older ones
	// refuse a query with a field they do not know, so the fields are dropped
	// one by one until the query is accepted.
	for _, fields := range []string{
		nvidiaLiveFields + ",clocks_event_reasons.active",
		nvidiaLiveFields + ",clocks_throttle_reasons.active",
		strings.Replace(nvidiaLiveFields, ",compute_cap", "", 1) + ",clocks_throttle_reasons.active",
	} {
		out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu="+fields, "--format=csv,noheader,nounits").Output()
		if err != nil {
			continue
		}
		if gpus = parseNvidiaLive(string(out)); len(gpus) > 0 {
			break
		}
	}
	if len(gpus) == 0 {
		return nil
	}
	if out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits").Output(); err == nil {
		// The listing does not say which GPU a process is on; with one GPU
		// (the usual laptop) it is that one.
		if procs := parseComputeApps(string(out)); len(gpus) == 1 {
			gpus[0].Processes = procs
		}
	}
	return gpus
}
