package tune

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/backend/fake"
	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/registry"
)

// machine is a scripted computer: every device runs the fake engine at its own
// speed, the numbers of bench/results/*/optimize-stage0-* made quicker by the
// same factor, so a run takes seconds and the ratios stay the real ones.
type machine struct {
	survey  *Survey
	engines map[string]*fake.Backend
}

// scaleUp is how much faster than the real device the scripted one runs.
const scaleUp = 5

func engine(pp, tg float64, batchMS ...float64) *fake.Backend {
	return fake.New(fake.Options{
		PrefillRate: pp * scaleUp, DecodeRate: tg * scaleUp, MaxDraft: 3, Acceptance: 0.6, Memory: 16 << 30,
		BatchCost: func(tokens, _ int) time.Duration {
			ms := batchMS[min(tokens, len(batchMS))-1]
			return time.Duration(ms / scaleUp * float64(time.Millisecond))
		},
	})
}

// scriptedBackend lists the machine's devices and loads on the engine of the
// device named in the spec.
type scriptedBackend struct {
	*fake.Backend
	devices []backend.Device
	engines map[string]*fake.Backend
}

func (b *scriptedBackend) Devices(context.Context) ([]backend.Device, error) { return b.devices, nil }

func (b *scriptedBackend) CPUVariant() string { return "scripted" }

func (b *scriptedBackend) Load(ctx context.Context, spec backend.LoadSpec) (backend.Model, error) {
	e, ok := b.engines[spec.Device]
	if !ok {
		e = b.Backend
	}
	return e.Load(ctx, spec)
}

// rusunrisebook is an i7-12700H laptop with an RTX 3070 Ti and an Iris Xe.
func rusunrisebook(t *testing.T, m registry.Manifest) (*Survey, *scriptedBackend) {
	t.Helper()
	cpu := backend.Device{Backend: fake.Name, ID: "cpu", Type: backend.CPU, Name: "12th Gen Intel(R) Core(TM) i7-12700H", MemoryTotal: 32 * int64(GiB), MemoryFree: 12 * int64(GiB), Features: []string{backend.FeatureMultiRowQ4_0}}
	rtx := backend.Device{Backend: fake.Name, ID: "cuda0", Type: backend.GPU, Name: "NVIDIA GeForce RTX 3070 Ti Laptop GPU", MemoryTotal: 8 * int64(GiB), MemoryFree: 7 * int64(GiB), Features: []string{backend.FeatureCudaNano, backend.FeatureBackendSampling}}
	iris := backend.Device{Backend: fake.Name, ID: "vulkan1", Type: backend.GPU, Name: "Intel(R) Iris(R) Xe Graphics", MemoryTotal: 16 * int64(GiB), MemoryFree: 15 * int64(GiB), Unified: true}
	b := &scriptedBackend{
		Backend: fake.New(fake.Options{}),
		devices: []backend.Device{cpu, rtx, iris},
		engines: map[string]*fake.Backend{
			"fake:cpu":     engine(202, 11.1, 95.5, 99, 108, 125),
			"fake:cuda0":   engine(5693, 100, 9.9, 10.5, 11.5, 12.0),
			"fake:vulkan1": engine(449, 11.3, 112, 125, 137, 143),
		},
	}
	s := &Survey{
		Versions: Versions{LibsRelease: "b1-n1"},
		OS:       OSInfo{Name: "windows"},
		CPU:      CPUInfo{Model: cpu.Name, Arch: "amd64", Physical: 14, Logical: 20, PCores: 6, ECores: 8, Variant: "alderlake", Features: cpu.Features, NUMANodes: 1},
		RAM:      RAMInfo{Total: 32 * int64(GiB), Available: 12 * int64(GiB)},
		Disk:     DiskInfo{Path: "fake.gguf", Speed: &hwinfo.DiskSpeed{SeqMiBs: 3295}},
		Model:    modelInfo(SurveyOptions{Manifest: m, ModelPath: "fake.gguf", DrafterPath: "fake-drafter.gguf"}),
		Load:     LoadInfo{CPUPercent: 3, GPUPercent: 2},
	}
	for _, d := range b.devices {
		s.Devices = append(s.Devices, deviceOf(d))
	}
	for i := range s.Devices {
		if s.Devices[i].Kind == "cuda" {
			s.Devices[i].ComputeCap, s.Devices[i].CC = "8.6", 86
		}
	}
	return s, b
}

// hyper is a 16-core EPYC server with four memory nodes and a hard disk.
func hyper(t *testing.T, m registry.Manifest) (*Survey, *scriptedBackend) {
	t.Helper()
	cpu := backend.Device{Backend: fake.Name, ID: "cpu", Type: backend.CPU, Name: "AMD EPYC 7351P 16-Core Processor", MemoryTotal: 62 * int64(GiB), MemoryFree: 50 * int64(GiB)}
	b := &scriptedBackend{
		Backend: fake.New(fake.Options{}),
		devices: []backend.Device{cpu},
		engines: map[string]*fake.Backend{"fake:cpu": engine(145, 24.8, 40, 52, 61, 72)},
	}
	s := &Survey{
		Versions: Versions{LibsRelease: "b1-n1"},
		OS:       OSInfo{Name: "linux"},
		CPU:      CPUInfo{Model: cpu.Name, Arch: "amd64", Physical: 16, Logical: 32, Variant: "haswell", NUMANodes: 4},
		RAM:      RAMInfo{Total: 62 * int64(GiB), Available: 50 * int64(GiB)},
		Disk:     DiskInfo{Path: "fake.gguf", Speed: &hwinfo.DiskSpeed{SeqMiBs: 130}},
		Model:    modelInfo(SurveyOptions{Manifest: m, ModelPath: "fake.gguf", DrafterPath: "fake-drafter.gguf"}),
		Load:     LoadInfo{CPUPercent: 2, GPUPercent: -1},
	}
	for _, d := range b.devices {
		s.Devices = append(s.Devices, deviceOf(d))
	}
	return s, b
}

func runScripted(t *testing.T, mode Mode, s *Survey, b *scriptedBackend, m registry.Manifest, mx *Matrix) *Result {
	t.Helper()
	res, err := Run(context.Background(), RunOptions{
		Options: Options{
			Mode:   mode,
			Survey: SurveyOptions{Backends: []backend.Backend{b}, Manifest: m, ModelPath: "fake.gguf", DrafterPath: "fake-drafter.gguf"},
			Plan:   PlanInput{Manifest: m, ModelPath: "fake.gguf", DrafterPath: "fake-drafter.gguf", Matrix: mx},
		},
		Runner:   InProcessRunner{Backend: b},
		Survey:   s,
		Cooldown: func(context.Context, Device, int) {},
		Budget:   2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("%s: %v", mode, err)
	}
	return res
}

// TestBlindRun is the blind test of docs/OPTIMIZE.md, section 7: the cells of
// the classes of the machine itself are taken out of the matrix, as on hardware
// nobody has measured, and the autotuner must still reach 95 % of the speed of
// the typical answer it gets with the whole matrix, and never find the drafter
// changing the answer. What is not known about a class is inherited from the
// nearest one and checked by the probes; it must not be switched off.
func TestBlindRun(t *testing.T) {
	if testing.Short() {
		t.Skip("scripted devices run for a few seconds")
	}
	cat, err := registry.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := registry.Find(cat, "gemma-4-e4b-qat")
	if !ok {
		t.Fatal("no gemma-4-e4b-qat in the catalog")
	}
	for _, tc := range []struct {
		name  string
		build func(*testing.T, registry.Manifest) (*Survey, *scriptedBackend)
		modes []Mode
		blind []Class
	}{
		{
			name: "rusunrisebook", build: rusunrisebook, modes: []Mode{Optimal, Powerful},
			blind: []Class{
				{AttrCPUVariant: "alderlake", AttrHybrid: "true"},
				{AttrGPUBackend: "vulkan", AttrGPUVendor: "intel", AttrGPUMemory: "shared", AttrOS: "windows"},
				{AttrGPUCC: "86", AttrGPUBackend: "cuda", AttrGPUMemory: "own", AttrGPUVendor: "nvidia"},
			},
		},
		{
			name: "hyper", build: hyper, modes: []Mode{Optimal},
			blind: []Class{{AttrCPUVariant: "haswell", AttrHybrid: "false"}, {AttrNUMA: "4"}},
		},
	} {
		for _, mode := range tc.modes {
			t.Run(tc.name+"/"+string(mode), func(t *testing.T) {
				s, b := tc.build(t, m)
				full := runScripted(t, mode, s, b, m, Default)
				s, b = tc.build(t, m)
				blind := runScripted(t, mode, s, b, m, Default.Without(tc.blind...))

				for _, r := range []*Result{full, blind} {
					for _, p := range append(append([]Problem{}, r.Report.Problems...), r.Profile.Problems...) {
						if p.Code == DraftMismatch {
							t.Errorf("the drafter changed the answer: %s", p.Message)
						}
					}
				}
				f, bl := full.Profile.Answer, blind.Profile.Answer
				if f.AnswerS <= 0 || bl.AnswerS <= 0 {
					t.Fatalf("answers: full %+v, blind %+v", f, bl)
				}
				t.Logf("full:  %s %.2f s %+v", full.Decision.Chosen.Candidate.ID, f.AnswerS, full.Profile.Chosen)
				t.Logf("blind: %s %.2f s %+v", blind.Decision.Chosen.Candidate.ID, bl.AnswerS, blind.Profile.Chosen)
				// The same setup is the same speed; the times of two runs of the scripted
				// devices differ by their timers, which is not what is tested.
				setup := func(p *Profile) string {
					return fmt.Sprint(p.Chosen.Device.Value, "|", p.Chosen.PlanMode.Value, "|", p.Chosen.NCtx, "|", p.Chosen.MTP.Value, "|", p.Chosen.Threads.Value)
				}
				if setup(full.Profile) != setup(blind.Profile) {
					if speed := f.AnswerS / bl.AnswerS; speed < 0.95 {
						t.Errorf("blind chose another setup and reaches %.0f %% of the speed of the full matrix: %.2f s against %.2f s", speed*100, bl.AnswerS, f.AnswerS)
					}
				}
				// Optimizations the machine's class never had a cell for are not lost
				// by not being known.
				fd, _ := decodeDraft(full.Profile.Chosen.MTP.Value)
				bd, _ := decodeDraft(blind.Profile.Chosen.MTP.Value)
				if fd.Enabled && !bd.Enabled {
					t.Errorf("the blind run left the drafter off: full %+v, blind %+v", fd, bd)
				}
			})
		}
	}
}
