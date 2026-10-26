package hwinfo

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Live is what the machine is doing at one moment: the memory that is free to
// use, how busy the CPU is, and the state of each NVIDIA GPU. The autotuner
// reads it before a measurement to tell a quiet machine from a busy one and
// to see how much memory a mode may count on (docs/OPTIMIZE.md, §3.1).
type Live struct {
	RAMTotal     int64 `json:"ram_total"`
	RAMAvailable int64 `json:"ram_available"`
	// CPUPercent is the share of CPU time that was not idle over the sample
	// window, 0..100; -1 when the platform gave no reading.
	CPUPercent float64   `json:"cpu_percent"`
	GPUs       []GPULive `json:"gpus,omitempty"`
}

// GPULive is one NVIDIA GPU as nvidia-smi reports it.
type GPULive struct {
	Index         int     `json:"index"`
	Name          string  `json:"name"`
	Driver        string  `json:"driver,omitempty"`
	ComputeCap    string  `json:"compute_cap,omitempty"` // "8.6"
	UtilPercent   int     `json:"util_percent"`
	TempC         int     `json:"temp_c"`
	SMClockMHz    int     `json:"sm_clock_mhz"`
	MaxSMClockMHz int     `json:"max_sm_clock_mhz"`
	MemUsedMiB    int64   `json:"mem_used_mib"`
	MemTotalMiB   int64   `json:"mem_total_mib"`
	PowerW        float64 `json:"power_w,omitempty"`
	// Throttle names what holds the clocks down right now (sw_power_cap,
	// hw_thermal, …); empty when nothing does.
	Throttle  []string  `json:"throttle,omitempty"`
	Processes []GPUProc `json:"processes,omitempty"`
}

// GPUProc is a process nvidia-smi lists on a GPU. Under WDDM (Windows) it
// lists compute processes only, and without their memory.
type GPUProc struct {
	PID    int    `json:"pid"`
	Name   string `json:"name"`
	MemMiB int64  `json:"mem_mib,omitempty"`
}

// ReadLive samples the machine. The CPU load is measured over window (one
// second when zero), during which the GPUs are read too.
func ReadLive(ctx context.Context, window time.Duration) Live {
	if window <= 0 {
		window = time.Second
	}
	l := Live{CPUPercent: -1}
	l.RAMTotal, l.RAMAvailable = memoryNow()
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		l.CPUPercent = cpuBusy(ctx, window)
	}()
	go func() {
		defer wg.Done()
		l.GPUs = nvidiaLive(ctx)
	}()
	wg.Wait()
	return l
}

// MemoryNow is the physical memory in total and available for programs to
// allocate right now (Windows ullAvailPhys, Linux MemAvailable). It is a few
// microseconds and safe to call often.
func MemoryNow() (total, available int64) { return memoryNow() }

// throttleBits are the bits of nvidia-smi's clocks_throttle_reasons.active.
var throttleBits = []struct {
	bit  uint64
	name string
}{
	{0x4, "sw_power_cap"}, {0x8, "hw_slowdown"}, {0x20, "sw_thermal"}, {0x40, "hw_thermal"}, {0x80, "hw_power_brake"},
}

// parseThrottle names the throttle reasons set in nvidia-smi's hex mask
// ("0x0000000000000004"). Idle (0x1), application clock settings (0x2),
// sync boost (0x10) and display clocks (0x100) are not throttling.
func parseThrottle(mask string) []string {
	mask = strings.TrimSpace(mask)
	v, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(mask), "0x"), 16, 64)
	if err != nil {
		return nil
	}
	var out []string
	for _, b := range throttleBits {
		if v&b.bit != 0 {
			out = append(out, b.name)
		}
	}
	return out
}

// nvidiaLiveFields are the columns of the query, in order; the last one is the
// throttle mask, whose name depends on the driver (nvidia_live.go).
const nvidiaLiveFields = "index,name,driver_version,utilization.gpu,temperature.gpu,clocks.sm,clocks.max.sm,memory.used,memory.total,power.draw,compute_cap"

// parseNvidiaLive reads the rows of
//
//	nvidia-smi --query-gpu=<nvidiaLiveFields>,<throttle> --format=csv,noheader,nounits
//
// "[N/A]" and "[Not Supported]" fields read as zero. The compute_cap column is
// absent on old drivers; a row with one column fewer is read without it.
func parseNvidiaLive(s string) []GPULive {
	var out []GPULive
	for _, line := range strings.Split(strings.TrimSpace(s), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 11 {
			continue
		}
		for i := range f {
			f[i] = strings.TrimSpace(f[i])
		}
		num := func(i int) float64 {
			v, err := strconv.ParseFloat(f[i], 64)
			if err != nil {
				return 0
			}
			return v
		}
		g := GPULive{
			Index: int(num(0)), Name: f[1], Driver: f[2], UtilPercent: int(num(3)), TempC: int(num(4)),
			SMClockMHz: int(num(5)), MaxSMClockMHz: int(num(6)), MemUsedMiB: int64(num(7)), MemTotalMiB: int64(num(8)),
			PowerW: num(9),
		}
		if len(f) >= 12 {
			if cc := f[10]; cc != "" && !strings.HasPrefix(cc, "[") {
				g.ComputeCap = cc
			}
			g.Throttle = parseThrottle(f[11])
		} else {
			g.Throttle = parseThrottle(f[10])
		}
		out = append(out, g)
	}
	return out
}

// parseComputeApps reads the rows of nvidia-smi --query-compute-apps=pid,process_name,used_memory.
func parseComputeApps(s string) []GPUProc {
	var out []GPUProc
	for _, line := range strings.Split(strings.TrimSpace(s), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 3 {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(f[0]))
		if err != nil {
			continue
		}
		p := GPUProc{PID: pid, Name: baseName(strings.TrimSpace(f[1]))}
		if mib, err := strconv.ParseInt(strings.TrimSpace(f[2]), 10, 64); err == nil {
			p.MemMiB = mib
		}
		out = append(out, p)
	}
	return out
}

func baseName(p string) string {
	if i := strings.LastIndexAny(p, `\/`); i >= 0 {
		return p[i+1:]
	}
	return p
}
