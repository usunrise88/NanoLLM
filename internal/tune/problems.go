package tune

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/hwinfo"
)

// Severity says how much a problem matters (docs/OPTIMIZE.md, §3.5).
type Severity string

const (
	// SevInfo is for the reader's information.
	SevInfo Severity = "info"
	// SevWarn lowers the speed or the accuracy of the measurement.
	SevWarn Severity = "warn"
	// SevError means the mode cannot be met as asked; the module still picks the
	// best it can and says what is not met.
	SevError Severity = "error"
)

// Code names a problem. The UI translates the text by code and shows the
// numbers from Params.
type Code string

const (
	SlowDisk          Code = "slow_disk"
	FasterDisk        Code = "faster_disk_available"
	OldDriver         Code = "old_driver"
	CUDA13Available   Code = "cuda13_available"
	CUDAAddonMissing  Code = "cuda_addon_missing"
	CUDANotOwn        Code = "cuda_not_own"
	NoVulkanIGPU      Code = "no_vulkan_igpu"
	Session0NoIGPU    Code = "session0_no_igpu"
	VRAMBusy          Code = "vram_busy"
	VRAMSpill         Code = "vram_spill"
	LowFreeRAM        Code = "low_free_ram"
	ModeDegraded      Code = "mode_degraded"
	SilentUnavailable Code = "silent_unavailable"
	OnBattery         Code = "on_battery"
	PowerSaver        Code = "power_saver"
	BusyMachine       Code = "busy_machine"
	HotGPU            Code = "hot_gpu"
	NoFastKernels     Code = "no_fast_kernels"
	SingleChannel     Code = "single_channel"
	VM                Code = "vm"
	WSL               Code = "wsl"
	DraftMismatch     Code = "draft_mismatch"
)

// Problem is one finding: a code, how bad it is, a sentence with the numbers,
// advice and the numbers on their own.
type Problem struct {
	Code     Code           `json:"code"`
	Severity Severity       `json:"severity"`
	Message  string         `json:"message"`
	Advice   string         `json:"advice,omitempty"`
	Params   map[string]any `json:"params,omitempty"`
}

func (p Problem) String() string {
	if p.Advice == "" {
		return p.Message
	}
	return p.Message + " " + p.Advice
}

// Thresholds of the survey's problems.
const (
	// SlowDiskMiBs is the sequential rate below which a drive is slow.
	SlowDiskMiBs = 300
	// OldNvidiaDriver is the first NVIDIA driver major that runs the CUDA
	// add-on (CUDA 12.x needs 525).
	OldNvidiaDriver = 525
	// CUDA13Driver is the first driver major that runs the CUDA 13 build.
	CUDA13Driver = 580
	// OldDriverAge is how old an Intel or AMD driver may be before the module
	// calls it old. There are no measurements by version yet, so the bound is
	// generous.
	OldDriverAge = 2 * 365 * 24 * time.Hour
	// BusyCPUPercent and BusyGPUPercent are the load above which a timing is
	// suspect.
	BusyCPUPercent = 20
	BusyGPUPercent = 10
	// HotGPUC is the idle temperature above which a GPU is called hot.
	HotGPUC = 75
	// VRAMBusyShare is the share of video memory other programs may hold
	// before the module reports it.
	VRAMBusyShare = 0.25
)

func gib(b int64) string { return fmt.Sprintf("%.1f GiB", float64(b)/float64(GiB)) }

// degraded is the powerful mode that found too little free memory.
func degraded(free int64) Problem {
	return Problem{
		Code: ModeDegraded, Severity: SevError,
		Message: fmt.Sprintf("Only %s of memory is free, less than the %s the powerful mode needs: it runs with the optimal mode's memory limit (%s) and all the cores.", gib(free), gib(PowerfulMinFree), gib(OptimalMemLimit)),
		Advice:  "Close other programs and repeat the check to use more.",
		Params:  map[string]any{"free_bytes": free, "needed_bytes": PowerfulMinFree, "limit_bytes": OptimalMemLimit},
	}
}

// Problems lists what the survey shows is wrong with this machine for the mode.
func (s *Survey) Problems(c Constraints) []Problem {
	var out []Problem
	add := func(p Problem) { out = append(out, p) }
	// A mode that runs on the CPU only has no use for what is wrong with the
	// GPUs.
	if c.CPUOnly {
		defer func() { out = dropGPUProblems(out) }()
	}

	if d := s.Disk; d.Path != "" {
		slow := d.Kind.Type == "hdd"
		mibs := 0.0
		if d.Speed != nil {
			mibs = d.Speed.SeqMiBs
			slow = slow || (mibs > 0 && mibs < SlowDiskMiBs)
		}
		if slow {
			msg := "The model is on a slow disk"
			if mibs > 0 {
				msg = fmt.Sprintf("The model is on a slow disk (%.0f MB/s)", mibs*1.048576)
			}
			add(Problem{
				Code: SlowDisk, Severity: SevWarn,
				Message: msg + ": setups that read weights from the disk answer below 1 token per second.",
				Advice:  "Move the data directory (server.data_dir) to an SSD.",
				Params:  map[string]any{"seq_mib_s": mibs, "disk": d.Kind.Type},
			})
		}
		if (d.Kind.Type == "hdd" || d.Kind.Bus == "sata") && d.SystemDisk == "nvme" {
			add(Problem{
				Code: FasterDisk, Severity: SevInfo,
				Message: "This machine has a faster disk (NVMe): moving the model there speeds up loading and the quiet mode.",
				Params:  map[string]any{"disk": d.Kind.Type},
			})
		}
	}

	cuda := false
	for _, d := range s.Devices {
		if d.Kind == "cuda" {
			cuda = true
			if !d.Has("cuda-nano") {
				add(Problem{
					Code: CUDANotOwn, Severity: SevWarn,
					Message: fmt.Sprintf("The CUDA library is not NanoLLM's own: decoding is slower (on an RTX 3070 Ti 93 against 106 tokens per second) on %s.", d.Name),
					Advice:  "Update the CUDA add-on to the release of this build.",
					Params:  map[string]any{"device": d.Selector},
				})
			}
		}
		if d.Vendor == "NVIDIA" && d.DriverMajor() > 0 && d.DriverMajor() < OldNvidiaDriver {
			add(Problem{
				Code: OldDriver, Severity: SevWarn,
				Message: fmt.Sprintf("NVIDIA driver %s is too old for the CUDA add-on: the GPU runs through Vulkan or not at all.", d.Driver),
				Advice:  "Update the driver to 580 or newer.",
				Params:  map[string]any{"device": d.Selector, "driver": d.Driver, "min_major": OldNvidiaDriver},
			})
		}
		if d.Kind == "cuda" && s.Versions.CUDABuild == "cuda12" && d.DriverMajor() >= OldNvidiaDriver && d.DriverMajor() < CUDA13Driver && d.CC >= 75 {
			add(Problem{
				Code: CUDA13Available, Severity: SevInfo,
				Message: fmt.Sprintf("Driver %s runs the CUDA 12 build; driver %d or newer gives the CUDA 13 build (about 2%% faster decoding on an RTX).", d.Driver, CUDA13Driver),
				Params:  map[string]any{"device": d.Selector, "driver": d.Driver},
			})
		}
		if d.Type == "gpu" && d.Vendor != "NVIDIA" && d.DriverDate != "" {
			if age := time.Since(parseDay(d.DriverDate)); age > OldDriverAge {
				add(Problem{
					Code: OldDriver, Severity: SevWarn,
					Message: fmt.Sprintf("The %s driver (%s, %s) is %.0f months old.", d.Name, d.Driver, d.DriverDate, age.Hours()/24/30),
					Advice:  "Update the graphics driver.",
					Params:  map[string]any{"device": d.Selector, "driver": d.Driver, "date": d.DriverDate},
				})
			}
		}
	}

	nvidia := false
	for _, g := range s.GPUs {
		if cc := parseCC(g.ComputeCap); cc == 0 || cc >= 50 {
			nvidia = true
		}
	}
	if nvidia && !cuda {
		add(Problem{
			Code: CUDAAddonMissing, Severity: SevWarn,
			Message: "An NVIDIA GPU is present but the CUDA add-on is not installed: the card runs through Vulkan, slower.",
			Advice:  "Reinstall with the CUDA option (install.ps1 -Cuda).",
		})
	}

	if hasIntegrated(s) {
		if s.OS.Service {
			add(Problem{
				Code: Session0NoIGPU, Severity: SevWarn,
				Message: "The service runs in session 0 and does not see the integrated GPU.",
				Advice:  "Run the service as a user account (service install -account).",
			})
		} else {
			add(Problem{
				Code: NoVulkanIGPU, Severity: SevWarn,
				Message: "An integrated GPU is present but no Vulkan device shows it: the driver or the way NanoLLM was started hides it.",
				Advice:  "Update the graphics driver.",
			})
		}
	}

	for _, g := range s.GPUs {
		others := g.MemUsedMiB
		for _, p := range g.Processes {
			others -= p.MemMiB
		}
		if g.MemTotalMiB > 0 && float64(others) > VRAMBusyShare*float64(g.MemTotalMiB) {
			p := Problem{
				Code: VRAMBusy, Severity: SevWarn,
				Message: fmt.Sprintf("Other programs hold %.1f GiB of the %.1f GiB of video memory on %s: the GPU budget is smaller.", float64(others)/1024, float64(g.MemTotalMiB)/1024, g.Name),
				Advice:  "Close the programs that use the GPU (browser, games, streaming) and repeat the check.",
				Params:  map[string]any{"busy_mib": others, "total_mib": g.MemTotalMiB},
			}
			if names := procNames(g); len(names) > 0 {
				p.Params["apps"] = names
			}
			add(p)
		}
		if g.TempC > HotGPUC {
			add(Problem{
				Code: HotGPU, Severity: SevInfo,
				Message: fmt.Sprintf("%s is at %d C when idle.", g.Name, g.TempC),
				Advice:  "Check the cooling: a hot GPU lowers its clocks within a minute.",
				Params:  map[string]any{"temp_c": g.TempC},
			})
		}
	}

	if s.Power.Source == "battery" {
		add(Problem{
			Code: OnBattery, Severity: SevWarn,
			Message: "The machine runs on battery: it is slower, and a profile measured now will be too low.",
			Advice:  "Connect the charger and repeat the check.",
		})
	}
	if s.Power.Saver {
		add(Problem{
			Code: PowerSaver, Severity: SevWarn,
			Message: "Power saving is on: the CPU and the GPU run at lower clocks.",
			Advice:  "Switch the power mode to Balanced or Best performance.",
			Params:  map[string]any{"mode": s.Power.Name},
		})
	}
	if s.Load.CPUPercent > BusyCPUPercent || s.Load.GPUPercent > BusyGPUPercent {
		add(Problem{
			Code: BusyMachine, Severity: SevWarn,
			Message: fmt.Sprintf("The machine is busy (CPU %.0f%%, GPU %d%%): the measurement may come out too low.", max(s.Load.CPUPercent, 0), max(s.Load.GPUPercent, 0)),
			Advice:  "Close other programs and repeat the check.",
			Params:  map[string]any{"cpu_percent": s.Load.CPUPercent, "gpu_percent": s.Load.GPUPercent},
		})
	}

	if noFastKernels(s.CPU) {
		add(Problem{
			Code: NoFastKernels, Severity: SevWarn,
			Message: "The processor has no AVX2: the CPU runs several times slower and the repacked kernels are not available.",
		})
	}
	if s.RAM.Modules == 1 {
		add(Problem{
			Code: SingleChannel, Severity: SevInfo,
			Message: "Single-channel memory: decoding on the CPU is about twice as slow as with two modules.",
			Advice:  "A second memory module speeds it up.",
		})
	}
	switch {
	case s.OS.WSL:
		add(Problem{Code: WSL, Severity: SevInfo, Message: "Running inside WSL: the GPU may be unavailable and the disk is slower."})
	case s.OS.VM:
		add(Problem{Code: VM, Severity: SevInfo, Message: "Running in a virtual machine: the GPU may be unavailable and the disk is slower."})
	}

	sortProblems(out)
	return out
}

// noFastKernels is an x86 CPU without AVX2 (the generic ggml variants).
func noFastKernels(c CPUInfo) bool {
	if c.Arch != "amd64" && c.Arch != "386" {
		return false
	}
	if len(c.ISA) == 0 && c.Variant == "" {
		return false // nothing known
	}
	if c.Variant == "x64" || c.Variant == "sse42" || c.Variant == "sandybridge" || c.Variant == "ivybridge" || c.Variant == "piledriver" {
		return true
	}
	return len(c.ISA) > 0 && !c.Has("avx2")
}

// hasIntegrated: a display adapter of Intel or AMD that is not discrete is
// there, and no unified Vulkan device shows it. Only Windows lists adapters
// (hwinfo.GPUDrivers), so elsewhere nothing is claimed.
func hasIntegrated(s *Survey) bool {
	for _, a := range s.Adapters {
		if a.Vendor != "Intel" && a.Vendor != "AMD" {
			continue
		}
		seen := false
		for _, d := range s.Devices {
			if d.Type == "gpu" && sameAdapter(a.Name, d.Name) {
				seen = true
			}
		}
		if !seen {
			return true
		}
	}
	return false
}

// procNames lists the programs nvidia-smi shows on a GPU.
func procNames(g hwinfo.GPULive) []string {
	var out []string
	for _, p := range g.Processes {
		out = append(out, p.Name)
	}
	return out
}

func parseDay(s string) time.Time {
	t, _ := time.Parse("2006-01-02", s)
	return t
}

func sortProblems(ps []Problem) {
	rank := map[Severity]int{SevError: 0, SevWarn: 1, SevInfo: 2}
	sort.SliceStable(ps, func(i, j int) bool { return rank[ps[i].Severity] < rank[ps[j].Severity] })
}

// Names joins the problems' codes, for logs.
func Names(ps []Problem) string {
	var s []string
	for _, p := range ps {
		s = append(s, string(p.Code))
	}
	return strings.Join(s, ",")
}

// gpuProblems are the findings that are about a GPU.
var gpuProblems = map[Code]bool{
	OldDriver: true, CUDA13Available: true, CUDAAddonMissing: true, CUDANotOwn: true, NoVulkanIGPU: true,
	Session0NoIGPU: true, VRAMBusy: true, VRAMSpill: true, HotGPU: true,
}

func dropGPUProblems(ps []Problem) []Problem {
	var out []Problem
	for _, p := range ps {
		if !gpuProblems[p.Code] {
			out = append(out, p)
		}
	}
	return out
}
