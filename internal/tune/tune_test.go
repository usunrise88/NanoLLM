package tune

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/plan"
)

// The reference request is the yardstick of every profile and of the matrix's
// baselines: its text does not change without a new SuiteVersion.
func TestReferenceTextIsFrozen(t *testing.T) {
	sum := sha256.Sum256([]byte(strings.ReplaceAll(refText, "\r\n", "\n")))
	if got := hex.EncodeToString(sum[:]); got != RefTextSHA256 {
		t.Fatalf("refprompt.txt hashes to %s, not %s: the reference request changed; bump SuiteVersion and calibrate the baselines of matrix.yaml again", got, RefTextSHA256)
	}
	if body := refBody(); len(body) < 20000 || strings.Contains(body, "\r") {
		t.Fatalf("the reference text has %d bytes", len(body))
	}
}

func TestParseMode(t *testing.T) {
	for _, s := range []string{"silent", " Optimal ", "POWERFUL"} {
		if _, err := ParseMode(s); err != nil {
			t.Errorf("%q: %v", s, err)
		}
	}
	for _, s := range []string{"", "fast", "auto"} {
		if _, err := ParseMode(s); err == nil {
			t.Errorf("%q was accepted", s)
		}
	}
}

func TestConstraints(t *testing.T) {
	c := Cores{Physical: 14, Logical: 20}
	k, p := ConstraintsFor(Optimal, c, 20*GiB)
	if p != nil || k.MemCap != 4*GiB || k.MemTolerance != 200*MiB || k.Threads != 7 || k.ThreadsBatch != 7 || k.MinNCtx != 4608 || k.LowPriority {
		t.Fatalf("optimal: %+v %v", k, p)
	}
	k, _ = ConstraintsFor(Optimal, Cores{Physical: 1}, 0)
	if k.Threads != 1 {
		t.Fatalf("one core, %d threads", k.Threads)
	}
	k, _ = ConstraintsFor(Silent, c, 20*GiB)
	if !k.CPUOnly || k.Threads != 2 || k.ThreadsBatch != 2 || !k.LowPriority || k.Goal != GoalMinMemory || k.MemCap != 0 {
		t.Fatalf("silent: %+v", k)
	}
	k, _ = ConstraintsFor(Powerful, c, 20*GiB)
	if k.CPUOnly || k.MemCap != 0 || k.Threads != 0 || k.Goal != GoalMinTime {
		t.Fatalf("powerful: %+v", k)
	}
	// Less than 6 GiB free: the powerful mode takes the optimal mode's memory
	// limit and keeps all the cores, and says so.
	k, p = ConstraintsFor(Powerful, c, 5*GiB)
	if p == nil || p.Code != ModeDegraded || p.Severity != SevError || k.MemCap != 4*GiB || k.Threads != 0 {
		t.Fatalf("degraded: %+v %v", k, p)
	}
}

func TestDiskFormula(t *testing.T) {
	// 130 MB/s of a hard disk against 1.77 GiB of cold weights per token.
	got := DiskTokS(130/1.048576, 1772*MiB)
	if got < 0.06 || got > 0.08 {
		t.Fatalf("hdd, ram-2g: %.3f tok/s", got)
	}
	// A 3.3 GB/s NVMe against the ram-1g budget's 2131 MiB.
	if got := DiskTokS(3295, 2131*MiB); got < 1.4 || got > 1.7 {
		t.Fatalf("nvme, ram-1g: %.2f tok/s", got)
	}
	if DiskTokS(100, 0) != 0 || DiskTokS(0, 100) != 0 {
		t.Fatal("no data, no formula")
	}
}

func TestAnswerSeconds(t *testing.T) {
	// 4096 tokens of prompt at 5600 and 300 of answer at 167: the sketch of
	// docs/OPTIMIZE.md section 4.1.
	if got := AnswerSeconds(5600, 167); math.Abs(got-2.53) > 0.02 {
		t.Fatalf("%.2f", got)
	}
	if !math.IsInf(AnswerSeconds(0, 10), 1) {
		t.Fatal("no prefill rate must not be a time")
	}
}

func TestEstimateScalesCPUThreads(t *testing.T) {
	cls := Class{AttrCPUVariant: "alderlake", AttrHybrid: "true"}
	six := Default.EstimateFor(cls, "cpu", 6, false, &plan.Plan{}, 0)
	two := Default.EstimateFor(cls, "cpu", 2, false, &plan.Plan{}, 0)
	if !six.Known || two.PP >= six.PP/2 || two.TG >= six.TG || two.TG < six.TG*0.7 {
		t.Fatalf("six %+v two %+v: prefill follows threads, decode barely", six, two)
	}
	mtp := Default.EstimateFor(cls, "cpu", 6, true, &plan.Plan{}, 0)
	if mtp.TG <= six.TG*1.5 {
		t.Fatalf("drafting estimate %v against %v", mtp.TG, six.TG)
	}
	if Default.EstimateFor(Class{AttrGPUBackend: "metal"}, "metal", 0, false, &plan.Plan{}, 0).Known {
		t.Fatal("an estimate without a baseline")
	}
}

func TestEstimateCapsDecodeAtTheDisk(t *testing.T) {
	p := &plan.Plan{StreamRanges: []plan.StreamRange{{Layer: 0, Length: 1772 * MiB}}}
	e := Default.EstimateFor(Class{AttrCPUVariant: "haswell", AttrHybrid: "false"}, "cpu", 4, false, p, 130/1.048576)
	if !e.DiskBound || e.TG > 0.1 || e.AnswerS < 3000 {
		t.Fatalf("%+v", e)
	}
}

func TestBatchCostAndExpectedSpeedup(t *testing.T) {
	// hyper's scripts/tn-curve.sh: 2 and 3 tokens cost 1.01x and 1.11x of one.
	hyper := map[int]float64{1: 100, 3: 111, 4: 118}
	cost := BatchCost(hyper)
	if cost(1) != 1 || math.Abs(cost(3)-1.11) > 1e-9 || cost(2) < 1.0 || cost(2) > 1.11 || cost(5) <= cost(4) {
		t.Fatalf("c(1..5) = %v %v %v %v %v", cost(1), cost(2), cost(3), cost(4), cost(5))
	}
	k, gain := BestDraftMax(hyper, DefaultAcceptance, DraftCostFraction("cpu"), 3)
	if k < 2 || gain < 1.4 {
		t.Fatalf("a nearly flat batch cost must favour drafting: k=%d gain %.2f", k, gain)
	}
	// A device where a batch costs linearly in its size gets no drafting.
	linear := map[int]float64{1: 100, 3: 300, 4: 400}
	if k, gain := BestDraftMax(linear, DefaultAcceptance, DraftCostFraction("cpu"), 3); k != 0 || gain != 1 {
		t.Fatalf("linear cost: k=%d gain %.2f", k, gain)
	}
	// Positions with enough trials replace the default acceptance.
	got := AcceptanceFrom([]int{50, 30, 3}, []int{40, 12, 2})
	if got[0] != 0.8 || got[1] != 0.4 || got[2] != DefaultAcceptance[2] {
		t.Fatalf("%v", got)
	}
}

func TestProblemsOfASurvey(t *testing.T) {
	s := &Survey{
		Versions: Versions{CUDABuild: "cuda12"},
		OS:       OSInfo{Name: "windows", Service: false},
		CPU:      CPUInfo{Arch: "amd64", ISA: []string{"avx", "avx2"}, Variant: "haswell", Physical: 8},
		RAM:      RAMInfo{Modules: 1},
		Devices: []Device{
			{Selector: "llamacpp:cuda0", Kind: "cuda", Type: "gpu", Name: "RTX", Vendor: "NVIDIA", Driver: "551.23", CC: 86},
			{Selector: "llamacpp:vulkan0", Kind: "vulkan", Type: "gpu", Name: "Iris Xe", Vendor: "Intel", Unified: true, DriverDate: "2022-01-01", Driver: "30.0.1"},
		},
		Disk:  DiskInfo{Path: "m.gguf", Kind: hwinfo.DiskKind{Type: "hdd"}, SystemDisk: "nvme", Speed: &hwinfo.DiskSpeed{SeqMiBs: 124}},
		Power: PowerInfo{Source: "battery", Saver: true, Name: "balanced"},
		Load:  LoadInfo{CPUPercent: 45, GPUPercent: 3},
		GPUs:  []hwinfo.GPULive{{Index: 0, Name: "RTX", MemTotalMiB: 8192, MemUsedMiB: 3277, TempC: 80, Processes: []hwinfo.GPUProc{{PID: 1, Name: "chrome.exe"}}}},
	}
	got := map[Code]Severity{}
	for _, p := range s.Problems(Constraints{}) {
		got[p.Code] = p.Severity
		if p.Message == "" {
			t.Errorf("%s has no message", p.Code)
		}
	}
	want := []Code{SlowDisk, FasterDisk, CUDANotOwn, OldDriver, VRAMBusy, HotGPU, OnBattery, PowerSaver, BusyMachine, SingleChannel, CUDA13Available}
	for _, c := range want {
		if _, ok := got[c]; !ok {
			t.Errorf("missing %s; got %v", c, got)
		}
	}
	if got[OldDriver] != SevWarn || got[FasterDisk] != SevInfo || got[SlowDisk] != SevWarn {
		t.Errorf("severities: %v", got)
	}
}

// The environments that change what a measurement means, and the old CPU.
func TestEnvironmentProblems(t *testing.T) {
	codes := func(s *Survey) map[Code]bool {
		out := map[Code]bool{}
		for _, p := range s.Problems(Constraints{}) {
			out[p.Code] = true
		}
		return out
	}
	clean := func() *Survey {
		return &Survey{CPU: CPUInfo{Arch: "amd64", Variant: "alderlake", ISA: []string{"avx2"}}, Load: LoadInfo{CPUPercent: 3, GPUPercent: -1}}
	}
	vm := clean()
	vm.OS.VM = true
	if got := codes(vm); !got[VM] || got[WSL] {
		t.Errorf("a virtual machine: %v", got)
	}
	wsl := clean()
	wsl.OS.WSL = true
	if got := codes(wsl); !got[WSL] || got[VM] {
		t.Errorf("WSL: %v", got)
	}
	old := clean()
	old.CPU = CPUInfo{Arch: "amd64", Variant: "sandybridge", ISA: []string{"avx"}}
	if got := codes(old); !got[NoFastKernels] {
		t.Errorf("a CPU without AVX2: %v", got)
	}
	arm := clean()
	arm.CPU = CPUInfo{Arch: "arm64"}
	if got := codes(arm); got[NoFastKernels] {
		t.Errorf("an ARM CPU is not a CPU without AVX2: %v", got)
	}
}

func TestACleanSurveyReportsNothing(t *testing.T) {
	s := &Survey{CPU: CPUInfo{Arch: "amd64", Variant: "alderlake", ISA: []string{"avx2", "avx_vnni"}}, Load: LoadInfo{CPUPercent: 3, GPUPercent: -1}}
	if ps := s.Problems(Constraints{}); len(ps) != 0 {
		t.Fatalf("a clean survey reports %v", Names(ps))
	}
}

func TestNoGPULibraryProblems(t *testing.T) {
	s := &Survey{
		CPU:      CPUInfo{Arch: "amd64"},
		Devices:  []Device{{Selector: "llamacpp:vulkan0", Kind: "vulkan", Type: "gpu", Name: "NVIDIA GeForce RTX 3070 Ti", Vendor: "NVIDIA"}},
		GPUs:     []hwinfo.GPULive{{Index: 0, Name: "NVIDIA GeForce RTX 3070 Ti", ComputeCap: "8.6", MemTotalMiB: 8192}},
		Adapters: []hwinfo.GPUDriver{{Name: "Intel(R) Iris(R) Xe Graphics", Vendor: "Intel", Version: "32.0"}},
		OS:       OSInfo{Service: true},
		Load:     LoadInfo{GPUPercent: -1},
	}
	got := map[Code]bool{}
	for _, p := range s.Problems(Constraints{}) {
		got[p.Code] = true
	}
	if !got[CUDAAddonMissing] || !got[Session0NoIGPU] || got[NoVulkanIGPU] {
		t.Fatalf("%v", got)
	}
	s.OS.Service = false
	got = map[Code]bool{}
	for _, p := range s.Problems(Constraints{}) {
		got[p.Code] = true
	}
	if !got[NoVulkanIGPU] || got[Session0NoIGPU] {
		t.Fatalf("%v", got)
	}
}

func TestProfileKeyDiff(t *testing.T) {
	a := Key{Mode: Optimal, Model: "m", LibsRelease: "b1-n8", Schema: ProfileSchema, Suite: SuiteVersion, Device: "cuda0 RTX 8 GiB", Driver: "581.57", CPU: "i7", Cores: 14, RAMGiB: 31}
	b := a
	if ch := a.Diff(b); len(ch) != 0 {
		t.Fatalf("%v", ch)
	}
	b.Driver, b.LibsRelease = "590.12", "b1-n9"
	ch := a.Diff(b)
	if len(ch) != 2 || !strings.Contains(ch[0].String()+ch[1].String(), "581.57 → 590.12") {
		t.Fatalf("%v", ch)
	}
}

func TestProfileStoreKeepsThePreviousVersion(t *testing.T) {
	st := Store{Dir: filepath.Join(t.TempDir(), "tune")}
	p := &Profile{Schema: ProfileSchema, Mode: Optimal, Key: Key{Model: "gemma-4-e4b-qat"}, Created: time.Now(), Chosen: Chosen{NCtx: 8192}}
	if _, err := st.Load("gemma-4-e4b-qat", Optimal); !os.IsNotExist(err) {
		t.Fatalf("got %v", err)
	}
	if err := st.Save(p); err != nil {
		t.Fatal(err)
	}
	p2 := *p
	p2.Chosen.NCtx = 4608
	if err := st.Save(&p2); err != nil {
		t.Fatal(err)
	}
	got, err := st.Load("gemma-4-e4b-qat", Optimal)
	if err != nil || got.Chosen.NCtx != 4608 {
		t.Fatalf("%+v %v", got, err)
	}
	prev, err := st.LoadPrev("gemma-4-e4b-qat", Optimal)
	if err != nil || prev.Chosen.NCtx != 8192 {
		t.Fatalf("%+v %v", prev, err)
	}
	if p.Hash() == p2.Hash() || p.ID() != "optimal@"+p.Hash() {
		t.Fatalf("hash %s %s", p.Hash(), p2.Hash())
	}
}

// The identity of a profile is what it decides: a repeat of the measurement
// that chooses the same must not make the pool load the model again.
func TestProfileHashIsTheDecisions(t *testing.T) {
	mk := func() *Profile {
		return &Profile{Schema: ProfileSchema, Mode: Optimal, Key: Key{Model: "m"}, Chosen: Chosen{
			Device:   Value{Value: "llamacpp:cuda0", Source: SourceMeasured, Why: "2.8 s", Alternatives: map[string]float64{"llamacpp:cpu": 37}},
			PlanMode: Value{Value: "M3 vram-ple-disk", Source: SourceRule},
			NCtx:     16384, Slots: 1,
			MTP:     Value{Value: Draft{Enabled: true, DraftMax: 3}, Source: SourceMeasured, Alternatives: map[string]float64{"mtp3/0": 142.3, "off": 95}},
			Threads: Value{Value: Team{}, Source: SourcePosture},
		}}
	}
	a, b := mk(), mk()
	b.Chosen.Device.Why, b.Chosen.Device.Alternatives = "2.9 s", map[string]float64{"llamacpp:cpu": 36}
	b.Chosen.MTP.Alternatives, b.Chosen.MTP.Source = map[string]float64{"mtp3/0": 138.9, "off": 94}, SourceMechanism
	if a.Hash() != b.Hash() {
		t.Errorf("the same decisions hash to %s and %s", a.Hash(), b.Hash())
	}
	st := Store{Dir: t.TempDir()}
	if err := st.Save(a); err != nil {
		t.Fatal(err)
	}
	back, err := st.Load("m", Optimal)
	if err != nil || back.Hash() != a.Hash() {
		t.Errorf("a profile read back hashes to %s, not %s (%v)", back.Hash(), a.Hash(), err)
	}
	b.Chosen.MTP.Value = Draft{Enabled: true, DraftMax: 2}
	if a.Hash() == b.Hash() {
		t.Error("another draft length, the same hash")
	}
}

// Drafted and plain decoding must agree, except where the model hesitates
// between two words and the batch kernels decide the toss.
func TestCompareDrafts(t *testing.T) {
	toks := func(ids ...int) []backend.Token {
		var out []backend.Token
		for _, id := range ids {
			out = append(out, backend.Token(id))
		}
		return out
	}
	// The plain run chose token 7 at position 2 with logprob -0.5; alts are what it listed behind it.
	plain := func(alts ...backend.TokenLogprob) []backend.TokenLogprobs {
		lps := make([]backend.TokenLogprobs, 4)
		lps[2] = backend.TokenLogprobs{Token: 7, Logprob: -0.5, Top: append([]backend.TokenLogprob{{Token: 7, Logprob: -0.5}}, alts...)}
		return lps
	}
	same := CompareDrafts(toks(1, 2, 7, 4), toks(1, 2, 7, 4), nil)
	if !same.Match || same.At != -1 || same.NearTie {
		t.Errorf("equal tokens: %+v", same)
	}
	tie := CompareDrafts(toks(1, 2, 9, 4), toks(1, 2, 7, 4), plain(backend.TokenLogprob{Token: 9, Logprob: -0.54}))
	if !tie.Match || !tie.NearTie || tie.At != 2 || tie.Gap < 0.039 || tie.Gap > 0.041 {
		t.Errorf("a near tie: %+v", tie)
	}
	far := CompareDrafts(toks(1, 2, 9, 4), toks(1, 2, 7, 4), plain(backend.TokenLogprob{Token: 9, Logprob: -1.0}))
	if far.Match || far.NearTie || far.At != 2 {
		t.Errorf("a gap of 0.5 nats: %+v", far)
	}
	unseen := CompareDrafts(toks(1, 2, 9, 4), toks(1, 2, 7, 4), plain(backend.TokenLogprob{Token: 8, Logprob: -0.5}))
	if unseen.Match {
		t.Errorf("a token the plain run never listed: %+v", unseen)
	}
	short := CompareDrafts(toks(1, 2), toks(1, 2, 7, 4), nil)
	if short.Match || short.At != 2 {
		t.Errorf("a shorter drafted run: %+v", short)
	}
}
