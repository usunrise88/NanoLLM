package tune

import (
	"cmp"
	"context"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/registry"
)

// SurveyTimeout bounds the whole look around (docs/OPTIMIZE.md, §3.1).
const SurveyTimeout = 10 * time.Second

// Survey is what the module found out about the machine. Everything in it is
// read with ordinary rights.
type Survey struct {
	Taken    time.Time `json:"taken"`
	Elapsed  float64   `json:"elapsed_s"`
	Versions Versions  `json:"versions"`
	OS       OSInfo    `json:"os"`
	CPU      CPUInfo   `json:"cpu"`
	RAM      RAMInfo   `json:"ram"`
	Devices  []Device  `json:"devices"`
	// Adapters are the display adapters the OS knows, whether or not a backend
	// could open them: a card the backends do not list is a finding.
	Adapters []hwinfo.GPUDriver `json:"adapters,omitempty"`
	GPUs     []hwinfo.GPULive   `json:"gpus,omitempty"`
	Disk     DiskInfo           `json:"disk"`
	Power    PowerInfo          `json:"power"`
	Load     LoadInfo           `json:"load"`
	Model    ModelInfo          `json:"model"`
	// Notes name the parts of the survey that failed or were skipped.
	Notes []string `json:"notes,omitempty"`
}

// Versions are the builds the profile is tied to.
type Versions struct {
	Nanollm     string `json:"nanollm"`
	LibsRelease string `json:"libs_release"`
	LlamaTag    string `json:"llama_tag,omitempty"`
	LlamaCommit string `json:"llama_commit,omitempty"`
	// CUDABuild is "cuda12" or "cuda13" by the runtime library found in the
	// library directory; empty without CUDA.
	CUDABuild string `json:"cuda_build,omitempty"`
}

// OSInfo is the system the process runs in.
type OSInfo struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
	VM      bool   `json:"vm,omitempty"`
	WSL     bool   `json:"wsl,omitempty"`
	// Service says the process lives in Windows session 0 (a service).
	Service bool `json:"service,omitempty"`
}

// CPUInfo is the processor and the kernels the CPU backend picked for it.
type CPUInfo struct {
	Model    string   `json:"model"`
	Vendor   string   `json:"vendor,omitempty"`
	Arch     string   `json:"arch"`
	Physical int      `json:"physical"`
	Logical  int      `json:"logical"`
	PCores   int      `json:"p_cores,omitempty"`
	ECores   int      `json:"e_cores,omitempty"`
	ISA      []string `json:"isa,omitempty"`
	// Variant is the ggml CPU backend variant the library loaded (alderlake,
	// haswell, zen4, …), Features what the build says about its kernels
	// (backend.FeatureMultiRowQ4_0).
	Variant   string   `json:"variant,omitempty"`
	Features  []string `json:"features,omitempty"`
	NUMANodes int      `json:"numa_nodes,omitempty"`
}

// Hybrid is a CPU with performance and efficiency cores.
func (c CPUInfo) Hybrid() bool { return c.PCores > 0 && c.ECores > 0 }

// Has reports whether the CPU lists an ISA extension.
func (c CPUInfo) Has(isa string) bool {
	for _, x := range c.ISA {
		if x == isa {
			return true
		}
	}
	return false
}

// RAMInfo is the memory of the machine.
type RAMInfo struct {
	Total     int64 `json:"total"`
	Available int64 `json:"available"`
	SpeedMTS  int   `json:"speed_mts,omitempty"`
	// Modules is the number of populated slots, 0 when the firmware does not
	// say.
	Modules int `json:"modules,omitempty"`
}

// Device is a place a model can run, as a backend lists it plus what the OS
// knows about its driver.
type Device struct {
	Selector string `json:"selector"`
	Backend  string `json:"backend"`
	ID       string `json:"id"`
	// Kind is cpu, cuda, vulkan or metal: the device id without its index.
	Kind        string   `json:"kind"`
	Type        string   `json:"type"`
	Name        string   `json:"name"`
	Vendor      string   `json:"vendor,omitempty"`
	MemoryTotal int64    `json:"memory_total"`
	MemoryFree  int64    `json:"memory_free"`
	Unified     bool     `json:"unified,omitempty"`
	Features    []string `json:"features,omitempty"`
	Driver      string   `json:"driver,omitempty"`
	DriverDate  string   `json:"driver_date,omitempty"`
	// ComputeCap is NVIDIA's "8.6"; CC the same as 86.
	ComputeCap string `json:"compute_cap,omitempty"`
	CC         int    `json:"cc,omitempty"`
}

// Has reports whether the device lists a feature.
func (d Device) Has(f string) bool {
	for _, x := range d.Features {
		if x == f {
			return true
		}
	}
	return false
}

// Discrete is a GPU with memory of its own.
func (d Device) Discrete() bool { return d.Type == string(backend.GPU) && !d.Unified }

// DriverMajor is the leading number of the driver version ("581.57" is 581);
// 0 when it has none.
func (d Device) DriverMajor() int {
	s := d.Driver
	if i := strings.IndexAny(s, ".- "); i > 0 {
		s = s[:i]
	}
	n, _ := strconv.Atoi(s)
	return n
}

// DiskInfo is the drive that holds the model.
type DiskInfo struct {
	Path string          `json:"path"`
	Kind hwinfo.DiskKind `json:"kind"`
	// Speed is nil when the probe was skipped or failed (SpeedErr says why).
	Speed    *hwinfo.DiskSpeed `json:"speed,omitempty"`
	SpeedErr string            `json:"speed_err,omitempty"`
	// SystemDisk is the type of the first drive of the machine, for the
	// "a faster disk is there" hint.
	SystemDisk string `json:"system_disk,omitempty"`
}

// PowerInfo is the power source and mode.
type PowerInfo struct {
	// Source is ac or battery; empty when unknown (desktops report ac).
	Source string `json:"source,omitempty"`
	Plan   string `json:"plan,omitempty"`
	Saver  bool   `json:"saver,omitempty"`
	Name   string `json:"name,omitempty"`
}

// LoadInfo is how busy the machine was while it was looked over.
type LoadInfo struct {
	CPUPercent float64 `json:"cpu_percent"`
	// GPUPercent is the busiest NVIDIA GPU; -1 when none was read.
	GPUPercent int `json:"gpu_percent"`
	GPUTempC   int `json:"gpu_temp_c,omitempty"`
}

// ModelInfo is the model the profile is for.
type ModelInfo struct {
	ID            string `json:"id"`
	Revision      string `json:"revision,omitempty"`
	SHA256        string `json:"sha256,omitempty"`
	DrafterSHA256 string `json:"drafter_sha256,omitempty"`
	Path          string `json:"path,omitempty"`
	DrafterPath   string `json:"drafter_path,omitempty"`
	Bytes         int64  `json:"bytes,omitempty"`
}

// SurveyOptions say what to look at.
type SurveyOptions struct {
	Backends    []backend.Backend
	LibDir      string
	Manifest    registry.Manifest
	ModelPath   string
	DrafterPath string
	Version     string
	// LibsRelease, LlamaTag and LlamaCommit are the builds this binary was
	// made for (llamacpp.LibsRelease, PinnedTag, PinnedCommit).
	LibsRelease string
	LlamaTag    string
	LlamaCommit string
	// LiveWindow is how long the CPU load is sampled; a second by default.
	LiveWindow time.Duration
	// SkipDisk leaves out the disk speed probe (tests, repeated looks).
	SkipDisk bool
	// SkipMachine leaves out everything the OS is asked about but the devices
	// and the memory: hwinfo.Collect, which starts PowerShell on Windows, the
	// GPU readings, the drivers, the power mode (tests).
	SkipMachine bool
}

// TakeSurvey looks the machine over. The parts run side by side and the whole
// takes SurveyTimeout at most; a part that fails leaves its fields empty and a
// note.
func TakeSurvey(ctx context.Context, o SurveyOptions) *Survey {
	start := time.Now()
	ctx, cancel := context.WithTimeout(ctx, SurveyTimeout)
	defer cancel()
	s := &Survey{Taken: start.UTC()}
	s.Versions = Versions{Nanollm: o.Version, LibsRelease: o.LibsRelease, LlamaTag: o.LlamaTag, LlamaCommit: o.LlamaCommit, CUDABuild: cudaBuild(o.LibDir)}
	s.Model = modelInfo(o)
	s.Disk.Path = o.ModelPath

	// The devices come first: Collect wants the GPUs the backends found.
	var cpuVariant string
	var cpuFeatures []string
	for _, b := range o.Backends {
		devs, err := b.Devices(ctx)
		if err != nil {
			s.Notes = append(s.Notes, "backend "+b.Name()+" lists no devices: "+err.Error())
			continue
		}
		if cv, ok := b.(interface{ CPUVariant() string }); ok && cpuVariant == "" {
			cpuVariant = cv.CPUVariant()
		}
		for _, d := range devs {
			s.Devices = append(s.Devices, deviceOf(d))
			if d.Type == backend.CPU && cpuFeatures == nil {
				cpuFeatures = d.Features
			}
		}
	}

	var machine benchfmt.Machine
	var live hwinfo.Live
	var speed hwinfo.DiskSpeed
	var speedErr error
	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		var gpus []benchfmt.GPU
		for _, d := range s.Devices {
			if d.Type == string(backend.GPU) {
				gpus = append(gpus, benchfmt.GPU{Name: d.Name, VRAMBytes: d.MemoryTotal, Unified: d.Unified})
			}
		}
		if !o.SkipMachine {
			machine = hwinfo.Collect(ctx, gpus)
		}
	}()
	go func() {
		defer wg.Done()
		if o.SkipMachine {
			live.CPUPercent = -1
			live.RAMTotal, live.RAMAvailable = hwinfo.MemoryNow()
			return
		}
		live = hwinfo.ReadLive(ctx, o.LiveWindow)
	}()
	go func() {
		defer wg.Done()
		if o.SkipDisk || o.ModelPath == "" {
			return
		}
		speed, speedErr = hwinfo.ReadDiskSpeed(ctx, o.ModelPath)
	}()
	s.Disk.Kind = hwinfo.DiskOf(o.ModelPath)
	if o.ModelPath != "" {
		if st, err := os.Stat(o.ModelPath); err == nil {
			s.Model.Bytes = st.Size()
		}
	}
	var modules int
	var power hwinfo.PowerMode
	var service bool
	if !o.SkipMachine {
		s.Adapters = hwinfo.GPUDrivers()
		modules = hwinfo.MemoryModules()
		power = hwinfo.ReadPowerMode()
		service = hwinfo.InServiceSession()
	}
	wg.Wait()

	s.OS = OSInfo{Name: machine.OS.Name, Version: machine.OS.Version, VM: machine.System.FormFactor == benchfmt.FormVM, WSL: machine.OS.WSL, Service: service}
	s.CPU = CPUInfo{
		Model: cmp.Or(hwinfo.CPUName(), machine.CPU.Model), Vendor: machine.CPU.Vendor, Arch: runtime.GOARCH,
		Physical: machine.CPU.Cores, Logical: machine.CPU.Threads, PCores: machine.CPU.PCores, ECores: machine.CPU.ECores,
		ISA: machine.CPU.ISA, Variant: cpuVariant, Features: cpuFeatures, NUMANodes: max(len(hwinfo.NUMATopology()), 1),
	}
	if s.CPU.Physical == 0 {
		s.CPU.Physical = hwinfo.PhysicalCores()
	}
	if s.CPU.Logical == 0 {
		s.CPU.Logical = runtime.NumCPU()
	}
	s.RAM = RAMInfo{Total: machine.Memory.TotalBytes, Available: live.RAMAvailable, SpeedMTS: machine.Memory.SpeedMTS, Modules: modules}
	if live.RAMTotal > 0 {
		s.RAM.Total = live.RAMTotal
	}
	s.Disk.SystemDisk = machine.Disk.Type
	if s.Disk.Kind.Type == hwinfo.DiskUnknown && o.ModelPath != "" && machine.Disk.Type != "" && machine.Disk.Type != "unknown" {
		// The OS would not say what the model's drive is; the first drive of
		// the machine is the best guess there is.
		s.Disk.Kind.Type = machine.Disk.Type
	}
	if o.ModelPath != "" {
		if speedErr != nil {
			s.Disk.SpeedErr = speedErr.Error()
		} else if !o.SkipDisk {
			s.Disk.Speed = &speed
		}
	}
	s.Power = PowerInfo{Source: machine.Power.Source, Plan: machine.Power.Profile, Saver: power.Saver, Name: power.Name}
	s.GPUs = live.GPUs
	s.Load = LoadInfo{CPUPercent: live.CPUPercent, GPUPercent: -1}
	for _, g := range live.GPUs {
		s.Load.GPUPercent = max(s.Load.GPUPercent, g.UtilPercent)
		s.Load.GPUTempC = max(s.Load.GPUTempC, g.TempC)
	}
	s.attachDrivers()
	s.Elapsed = time.Since(start).Seconds()
	return s
}

func modelInfo(o SurveyOptions) ModelInfo {
	mi := ModelInfo{ID: o.Manifest.ID, Path: o.ModelPath, DrafterPath: o.DrafterPath}
	if f, ok := o.Manifest.File(registry.RoleModel); ok {
		mi.Revision, mi.SHA256 = f.Revision, f.SHA256
	}
	if f, ok := o.Manifest.File(registry.RoleDrafter); ok {
		mi.DrafterSHA256 = f.SHA256
	}
	return mi
}

// deviceOf converts a backend device and works out its vendor and kind.
func deviceOf(d backend.Device) Device {
	return Device{
		Selector: d.Selector(), Backend: d.Backend, ID: d.ID, Kind: strings.TrimRight(d.ID, "0123456789"), Type: string(d.Type),
		Name: d.Name, Vendor: vendorOfName(d.Name), MemoryTotal: d.MemoryTotal, MemoryFree: d.MemoryFree, Unified: d.Unified, Features: d.Features,
	}
}

func vendorOfName(n string) string {
	l := strings.ToLower(n)
	switch {
	case strings.Contains(l, "nvidia"), strings.Contains(l, "geforce"), strings.Contains(l, "quadro"), strings.Contains(l, "tesla"):
		return "NVIDIA"
	case strings.Contains(l, "intel"):
		return "Intel"
	case strings.Contains(l, "amd"), strings.Contains(l, "radeon"):
		return "AMD"
	case strings.Contains(l, "apple"):
		return "Apple"
	}
	return ""
}

// attachDrivers gives every GPU device its driver: NVIDIA's from nvidia-smi
// (matched by CUDA index, or by name for Vulkan), Intel's and AMD's from the
// display class of the registry (matched by name) on Windows.
func (s *Survey) attachDrivers() {
	for i := range s.Devices {
		d := &s.Devices[i]
		if d.Type != string(backend.GPU) {
			continue
		}
		switch d.Vendor {
		case "NVIDIA":
			for _, g := range s.GPUs {
				if (d.Kind == "cuda" && g.Index == deviceIndex(d.ID)) || (d.Kind != "cuda" && g.Name == d.Name) {
					d.Driver, d.ComputeCap = g.Driver, g.ComputeCap
					d.CC = parseCC(g.ComputeCap)
					break
				}
			}
		default:
			for _, a := range s.Adapters {
				if sameAdapter(a.Name, d.Name) {
					d.Driver, d.DriverDate = a.Version, a.Date
					break
				}
			}
		}
	}
}

func deviceIndex(id string) int {
	n, _ := strconv.Atoi(strings.TrimLeft(id, "abcdefghijklmnopqrstuvwxyz"))
	return n
}

// parseCC turns "8.6" into 86.
func parseCC(s string) int {
	maj, min, ok := strings.Cut(s, ".")
	if !ok {
		return 0
	}
	a, err1 := strconv.Atoi(maj)
	b, err2 := strconv.Atoi(min)
	if err1 != nil || err2 != nil {
		return 0
	}
	return a*10 + b
}

var nonAlnum = regexp.MustCompile(`[^a-z0-9]+`)

// sameAdapter compares two adapter names the way a driver and a backend each
// spell them ("Intel(R) Iris(R) Xe Graphics" against "Intel Iris Xe Graphics").
func sameAdapter(a, b string) bool {
	norm := func(s string) string { return nonAlnum.ReplaceAllString(strings.ToLower(s), "") }
	na, nb := norm(a), norm(b)
	return na != "" && nb != "" && (na == nb || strings.Contains(na, nb) || strings.Contains(nb, na))
}

var cudartRe = regexp.MustCompile(`^(?:cudart64_(\d+)\.dll|libcudart\.so\.(\d+).*)$`)

// cudaBuild names the CUDA runtime that ships in the library directory:
// cuda12, cuda13 or empty.
func cudaBuild(libDir string) string {
	if libDir == "" {
		return ""
	}
	entries, err := os.ReadDir(libDir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if m := cudartRe.FindStringSubmatch(e.Name()); m != nil {
			return "cuda" + m[1] + m[2]
		}
	}
	// A system-wide CUDA (Linux) keeps no runtime in the directory; the backend
	// library's presence is all there is to say.
	if _, err := os.Stat(filepath.Join(libDir, backend.SharedLibrary("ggml-cuda"))); err == nil {
		return "cuda"
	}
	return ""
}
