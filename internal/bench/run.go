package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/memlock"
	"github.com/usunrise88/nanollm/internal/numa"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

// Options describe one benchmark run.
type Options struct {
	Manifest    registry.Manifest
	ModelPath   string
	DrafterPath string
	Backend     backend.Backend
	LibDir      string
	// Relaid serves the layout=nano value of a sweep: the checked relaid copy
	// of the model file (OPT-7). Nil, or an error, refuses those runs.
	Relaid func() (registry.Relaid, error)

	Suite string // standard | dev
	// Full runs every applicable preset; otherwise auto and two
	// representative ones.
	Full bool
	// Presets, when set, are run instead of the suite's choice.
	Presets []string
	Reps    int

	// Sets are the sweep axes of the dev suite (ParseSets); every preset
	// runs once per variant. Empty is the preset as planned.
	Variants []Variant
	// Place is where workers run (NUMA policy, pinned CPUs); a variant's
	// numa key overrides its policy.
	Place Placement
	// Cold evicts the model file from the page cache before each worker, so
	// the load reads the disk. A strict budget does it regardless: pages the
	// cache already holds are charged to whoever read them first, not to the
	// worker's cgroup, and would sit outside the limit.
	Cold bool
	// Advisory plans for the preset's budget but does not enforce it: no
	// cgroup scope and no forced eviction, so a run costs seconds instead of
	// a cold read of the disk. For sweeps of compute settings; the memory
	// numbers of such a run are not the budget's.
	Advisory bool
	// Timeout bounds one worker (all the repetitions of one scenario set); a
	// run that has not finished by then is recorded as failed, too slow. A
	// budget so tight that the weights fault in from the disk on every token
	// would otherwise take hours. Zero is no limit.
	Timeout time.Duration
	// Slow shortens each run (WorkerSpec.Slow), for budgets that stream the
	// weights from a hard disk.
	Slow bool
	// TuneProfile is the id of the autotune profile in force on this machine
	// (docs/OPTIMIZE.md, §3.7), recorded in every run's plan so that results of
	// machines are compared with their tuning. The presets themselves do not
	// depend on it: a public result stays comparable.
	TuneProfile string

	// Exe is the binary to run workers with; empty is this one.
	Exe     string
	Version string
	Commit  string
	Tag     string

	Events func(Event)
}

// Event is a step of a run, for the CLI's progress and the UI's stream.
type Event struct {
	Type      string        `json:"type"` // plan | start | result | estimate | done
	Preset    string        `json:"preset,omitempty"`
	Variant   string        `json:"variant,omitempty"`
	Index     int           `json:"index"`
	Total     int           `json:"total"`
	Run       *benchfmt.Run `json:"run,omitempty"`
	Message   string        `json:"message,omitempty"`
	RemainS   float64       `json:"remain_s,omitempty"`
	ElapsedMS float64       `json:"elapsed_ms,omitempty"`
}

func (o Options) emit(e Event) {
	if o.Events != nil {
		o.Events(e)
	}
}

// Run benchmarks the model under the suite's presets and returns the
// result, unsigned and unsent.
func Run(ctx context.Context, o Options) (*benchfmt.Result, error) {
	if o.Suite == "" {
		o.Suite = SuiteStandard
	}
	if o.Reps <= 0 {
		o.Reps = Reps
		if o.Suite == SuiteDev {
			o.Reps = DevReps
		}
	}
	if len(o.Variants) == 0 {
		o.Variants = []Variant{{}}
	}
	if o.Exe == "" {
		exe, err := os.Executable()
		if err != nil {
			return nil, err
		}
		o.Exe = exe
	}
	started := time.Now()
	devs, err := o.Backend.Devices(ctx)
	if err != nil {
		return nil, err
	}
	var cpu, gpu *backend.Device
	var gpus []benchfmt.GPU
	for i := range devs {
		d := &devs[i]
		switch d.Type {
		case backend.CPU:
			if cpu == nil {
				cpu = d
			}
		case backend.GPU:
			if gpu == nil {
				gpu = d
			}
			gpus = append(gpus, benchfmt.GPU{Name: d.Name, VRAMBytes: d.MemoryTotal, Unified: d.Unified})
		}
	}
	if cpu == nil {
		return nil, errors.New("bench: the backend lists no CPU device")
	}
	model, err := plan.ModelFrom(o.Manifest.ID, o.Manifest.Inventory)
	if err != nil {
		return nil, err
	}
	names := o.Presets
	if len(names) == 0 {
		names = choosePresets(o, cpu, gpu)
	}

	res := &benchfmt.Result{
		Schema:         benchfmt.Schema,
		ConsentVersion: ConsentVersion,
		Suite:          benchfmt.Suite{ID: o.Suite, Version: SuiteVersion, PromptsSHA256: PromptsSHA256(), Quick: !o.Full},
		Client: benchfmt.Client{
			NanoLLM: o.Version, LlamaCpp: benchfmt.LlamaCpp{Tag: o.Tag, Commit: o.Commit},
			OS: runtime.GOOS, Arch: runtime.GOARCH,
		},
		Machine: hwinfo.Collect(ctx, gpus),
		Date:    time.Now().UTC().Format("2006-01-02"),
	}
	res.Machine.Extra = append(res.Machine.Extra, hwinfo.FileDisk(o.ModelPath)...)
	if f, ok := o.Manifest.File(registry.RoleModel); ok {
		res.Model = benchfmt.Model{ID: o.Manifest.ID, Revision: f.Revision, SHA256: f.SHA256}
	}
	if f, ok := o.Manifest.File(registry.RoleDrafter); ok && o.DrafterPath != "" {
		res.Model.DrafterSHA256 = f.SHA256
	}

	seen := map[string]string{} // plan hash → preset that ran it
	var measured time.Duration
	nMeasured := 0
	total := len(names) * len(o.Variants)
	i := -1
	for _, name := range names {
		for _, v := range o.Variants {
			i++
			if err := ctx.Err(); err != nil {
				return res, err
			}
			label := name
			if len(v) > 0 {
				label += " [" + v.String() + "]"
			}
			o.emit(Event{Type: "plan", Preset: name, Variant: v.String(), Index: i, Total: total})
			run, spec, refused := planRun(o, model, name, v, cpu, gpu)
			run.Variant = v.String()
			if refused {
				res.Runs = append(res.Runs, run)
				o.emit(Event{Type: "result", Preset: name, Variant: run.Variant, Index: i, Total: total, Run: &run})
				continue
			}
			if prev, ok := seen[run.PlanHash]; ok {
				run.Status, run.SameAs = benchfmt.StatusSkipped, prev
				run.Reason = "same plan as " + prev
				res.Runs = append(res.Runs, run)
				o.emit(Event{Type: "result", Preset: name, Variant: run.Variant, Index: i, Total: total, Run: &run})
				continue
			}
			seen[run.PlanHash] = label
			o.emit(Event{Type: "start", Preset: name, Variant: run.Variant, Index: i, Total: total})
			t0 := time.Now()
			wr, enforcement, err := o.spawn(ctx, spec, budgetOf(name))
			run.Plan.Enforcement = enforcement
			if err != nil {
				run.Status, run.Reason = benchfmt.StatusFailed, err.Error()
			} else {
				run.Status, run.Metrics, run.Output, run.Reps = benchfmt.StatusOK, wr.Metrics, wr.Output, o.Reps
				if res.Client.GGMLCPUVariant == "" {
					res.Client.GGMLCPUVariant = wr.CPUVariant
				}
			}
			measured += time.Since(t0)
			nMeasured++
			res.Runs = append(res.Runs, run)
			o.emit(Event{Type: "result", Preset: name, Variant: run.Variant, Index: i, Total: total, Run: &run, ElapsedMS: ms(time.Since(t0))})
			if left := total - i - 1; left > 0 {
				o.emit(Event{Type: "estimate", Index: i, Total: total, RemainS: measured.Seconds() / float64(nMeasured) * float64(left)})
			}
		}
	}
	res.DurationS = time.Since(started).Seconds()
	o.emit(Event{Type: "done", Total: total})
	return res, nil
}

// choosePresets is the standard suite's choice: auto (what the machine has
// free) and two representative presets of its kind, or with Full every
// preset the machine can be asked to fit. The dev suite runs them all.
func choosePresets(o Options, cpu, gpu *backend.Device) []string {
	applicable := func(name string) bool {
		p, _ := plan.ParsePreset(name)
		switch {
		case p.Budget.Unified > 0:
			return gpu != nil && gpu.Unified
		case p.Budget.VRAM > 0:
			return gpu != nil && !gpu.Unified
		}
		return true
	}
	var all []string
	for _, n := range plan.Presets {
		if applicable(n) {
			all = append(all, n)
		}
	}
	if o.Full || o.Suite == SuiteDev {
		return all
	}
	dev := *cpu
	if gpu != nil {
		dev = *gpu
	}
	out := []string{}
	if p, err := plan.DerivedPreset(dev); err == nil {
		out = append(out, p.Name)
	}
	reps := []string{"ram-8g", "ram-4g"}
	switch {
	case gpu != nil && gpu.Unified:
		reps = []string{"uma-8g", "uma-4g"}
	case gpu != nil:
		reps = []string{"vram-8g", "vram-4g"}
	}
	for _, r := range reps {
		if applicable(r) {
			out = append(out, r)
		}
	}
	return out
}

func budgetOf(name string) plan.Budget {
	p, _ := plan.ParsePreset(name)
	return p.Budget
}

// planRun plans one preset. A refused preset is a result, not an error.
func planRun(o Options, model plan.Model, name string, v Variant, cpu, gpu *backend.Device) (benchfmt.Run, WorkerSpec, bool) {
	run := benchfmt.Run{Preset: name}
	preset, err := plan.ParsePreset(name)
	if err != nil {
		run.Status, run.Reason = benchfmt.StatusRefused, err.Error()
		return run, WorkerSpec{}, true
	}
	// OPT-7: the relaid copy is another file with other offsets, so it is
	// planned from its own inventory and loaded from its own path.
	modelPath, layout := o.ModelPath, registry.LayoutOriginal
	if v["layout"] == registry.LayoutNano {
		if o.Relaid == nil {
			run.Status, run.Reason = benchfmt.StatusRefused, "layout=nano: no relaid file is available here"
			return run, WorkerSpec{}, true
		}
		rl, err := o.Relaid()
		if err == nil {
			model, err = plan.ModelFrom(o.Manifest.ID, o.Manifest.WithRelaid(rl).Inventory)
		}
		if err != nil {
			run.Status, run.Reason = benchfmt.StatusRefused, "layout=nano: "+err.Error()
			return run, WorkerSpec{}, true
		}
		modelPath, layout = rl.Path, registry.LayoutNano
	}
	req := plan.Request{Model: model, Preset: preset, Proven: plan.ProvenFor(runtime.GOOS), NoMTP: o.DrafterPath == ""}
	onGPU := (preset.Budget.VRAM > 0 || preset.Budget.Unified > 0) && gpu != nil
	if !onGPU && cpu != nil && o.DrafterPath != "" && !v.setsSpeculation() {
		// What a server would run on this CPU by default. A variant that
		// names a speculation setting (a sweep of draft_max, mtp=off, ngram)
		// plans exactly what it says.
		req.DefaultCPUMTP(cpu.Has(backend.FeatureMultiRowQ4_0))
	}
	if onGPU {
		req.FlashAttnSmallBatch = gpu.Has(backend.FeatureFlashAttnSmallBatch)
		req.BackendSampling = gpu.Has(backend.FeatureBackendSampling)
	}
	v.Plan(&req)
	if kind := v["head"]; kind != "" && kind != registry.HeadFull {
		if onGPU || cpu == nil || !cpu.Has(backend.FeatureOutputShortlist) {
			run.Status, run.Reason = benchfmt.StatusRefused, "head="+kind+" needs a CPU and libraries with patch 0009 (device feature output-shortlist)"
			return run, WorkerSpec{}, true
		}
		ids, herr := registry.ShortlistIDs(o.Manifest, kind)
		if herr != nil {
			run.Status, run.Reason = benchfmt.StatusRefused, herr.Error()
			return run, WorkerSpec{}, true
		}
		req.Head, req.HeadIDs = kind, ids
	}
	if onGPU {
		req.Device = &plan.Device{
			Selector: gpu.Selector(), Backend: plan.Kind(gpu.ID),
			Memory: gpu.MemoryTotal, Unified: gpu.Unified,
		}
	}
	p, err := plan.Make(req)
	if err != nil {
		run.Status, run.Reason = benchfmt.StatusRefused, err.Error()
		return run, WorkerSpec{}, true
	}
	if !p.OK() {
		run.Status, run.Reason = benchfmt.StatusRefused, p.Refused
		return run, WorkerSpec{}, true
	}
	spec, err := p.LoadSpec(modelPath, o.DrafterPath)
	if err != nil {
		run.Status, run.Reason = benchfmt.StatusRefused, err.Error()
		return run, WorkerSpec{}, true
	}
	if p.Device == nil {
		spec.Device = cpu.Selector()
	}
	v.Load(&spec)
	place := o.Place
	if n, ok := v["numa"]; ok {
		place.NUMA = n
	}
	place, pinned, err := place.Resolve()
	if err != nil {
		run.Status, run.Reason = benchfmt.StatusRefused, err.Error()
		return run, WorkerSpec{}, true
	}
	if spec.Threads == 0 && pinned > 0 && !spec.FullyOnGPU() {
		// A pinned worker defaults to the physical cores it may run on;
		// this process sees the whole machine.
		spec.Threads = pinned
	}
	if spec.ThreadsBatch == 0 && pinned > 0 && !spec.FullyOnGPU() {
		spec.ThreadsBatch = pinned
	}
	spec = spec.WithDefaultThreads()
	batch, _ := v.int("batch")
	ws := WorkerSpec{Load: spec, Place: place, Slow: o.Slow, Batch: batch, OnlyBatch: v["only"] == "batch", Env: v.Env(), Cold: o.Cold || budgetOf(name).RAM > 0 && !o.Advisory && userScopes()}
	if layout == registry.LayoutNano {
		ws.Layout = layout // the original's hash stays what it was
	}
	for _, lr := range p.LockRanges {
		r := memlock.Range{Off: lr.Offset, Len: lr.Length}
		if lr.Role == registry.RoleDrafter {
			ws.DrafterLocks = append(ws.DrafterLocks, r)
		} else {
			ws.Locks = append(ws.Locks, r)
		}
	}
	for _, rr := range p.RandomRanges {
		ws.Random = append(ws.Random, memlock.Range{Off: rr.Offset, Len: rr.Length})
	}
	run.PlanHash = planHash(ws)
	run.Plan = benchfmt.Plan{
		Mode: string(p.Mode), Device: spec.Device, GPULayers: p.GPULayers, NCtx: p.NCtx, Slots: p.Slots,
		KV: p.KV.TypeK, FA: p.FlashAttention, SWAFull: p.KV.SWAFull, MTP: p.MTP.Enabled, Threads: spec.Threads,
		ThreadsBatch: spec.ThreadsBatch, NBatch: spec.NBatch, NUBatch: spec.NUBatch, KVV: spec.KVTypeV, LoadMode: string(spec.LoadMode), Lazy: string(spec.LazyMode),
		NUMA: place.NUMA, CPUs: place.CPUs, Cold: ws.Cold, Layout: layout,
	}
	run.Plan.Repack = onOff(spec.Repack)
	run.Plan.TuneProfile = o.TuneProfile
	if p.MTP.Enabled {
		run.Plan.DraftMax, run.Plan.DraftPMin = spec.DraftMax, spec.DraftPMin
	}
	return run, ws, false
}

// spawn runs one worker. On Linux a RAM budget is enforced with a transient
// systemd scope when the user manager is there (strict); otherwise the
// budget is only planned for (advisory). A placement is handed to the worker
// in its environment, and the worker places itself before it starts.
// Windows Job Objects arrive with stage 3.
func (o Options) spawn(ctx context.Context, ws WorkerSpec, budget plan.Budget) (WorkerResult, string, error) {
	ws.Backend, ws.LibDir, ws.Reps, ws.Quick = o.Backend.Name(), o.LibDir, o.Reps, !o.Full
	ws.Echo = o.Suite == SuiteDev
	if o.Suite != SuiteDev {
		ws.Batch = 0 // the throughput curve is a development measurement
	}
	enforcement := "advisory"
	strict := runtime.GOOS == "linux" && budget.RAM > 0 && !o.Advisory && userScopes()
	if strict {
		enforcement = "strict"
	}
	ws.Strict = strict
	if o.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, o.Timeout)
		defer cancel()
	}
	in, _ := json.Marshal(ws)
	cmdline := []string{o.Exe, "bench", "worker"}
	if strict {
		limit := budget.RAM + plan.HostOverhead
		cmdline = append([]string{"systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=" + strconv.FormatInt(limit, 10), "-p", "MemorySwapMax=0", "--"}, cmdline...)
	}
	if ws.Cold {
		evictFile(ws.Load.Path)
		evictFile(ws.Load.DrafterPath)
	}
	name, args := cmdline[0], cmdline[1:]
	cmd := exec.CommandContext(ctx, name, args...)
	// The worker places itself (internal/numa) before it reads its spec.
	if len(ws.Env) > 0 || !ws.Place.IsZero() {
		cmd.Env = os.Environ()
		for k, val := range ws.Env {
			cmd.Env = append(cmd.Env, k+"="+val)
		}
		if !ws.Place.IsZero() {
			spec := ws.Place.Spec()
			if ws.Place.NUMA == "distribute" && ws.Load.Threads > 0 {
				// The repack threads must be the decode threads (internal/numa).
				spec += ";threads=" + strconv.Itoa(ws.Load.Threads)
			}
			cmd.Env = append(cmd.Env, numa.EnvPlace+"="+spec)
		}
	}
	cmd.Stdin = bytes.NewReader(in)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	var wr WorkerResult
	if o.Timeout > 0 && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return wr, enforcement, fmt.Errorf("too slow: no result within %s (a budget that leaves the file-backed weights no page cache thrashes the disk)", o.Timeout)
	}
	if jerr := json.Unmarshal(stdout.Bytes(), &wr); jerr != nil {
		tail := strings.TrimSpace(stderr.String())
		if i := strings.LastIndex(tail, "\n"); i >= 0 {
			tail = tail[i+1:]
		}
		if strict && err != nil && strings.Contains(err.Error(), "killed") {
			tail = strings.TrimSpace("killed under the cgroup limit (out of memory?) " + tail)
		}
		if err == nil {
			err = jerr
		}
		return wr, enforcement, fmt.Errorf("worker: %w: %s", err, tail)
	}
	if wr.Error != "" {
		return wr, enforcement, errors.New(wr.Error)
	}
	return wr, enforcement, nil
}

// userScopes reports whether systemd-run can make a scope for this user. It
// is asked once: each probe starts a scope.
func userScopes() bool {
	userScopesOnce.Do(func() {
		if _, err := exec.LookPath("systemd-run"); err != nil {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		userScopesOK = exec.CommandContext(ctx, "systemd-run", "--user", "--scope", "--quiet", "true").Run() == nil
	})
	return userScopesOK
}

var (
	userScopesOnce sync.Once
	userScopesOK   bool
)

// ChoosePresets is the preset list Run would use, for a caller that wants to
// show it before the run starts.
func ChoosePresets(ctx context.Context, o Options) ([]string, error) {
	if len(o.Presets) > 0 {
		return o.Presets, nil
	}
	devs, err := o.Backend.Devices(ctx)
	if err != nil {
		return nil, err
	}
	var cpu, gpu *backend.Device
	for i := range devs {
		switch d := &devs[i]; d.Type {
		case backend.CPU:
			if cpu == nil {
				cpu = d
			}
		case backend.GPU:
			if gpu == nil {
				gpu = d
			}
		}
	}
	if cpu == nil {
		return nil, errors.New("bench: the backend lists no CPU device")
	}
	return choosePresets(o, cpu, gpu), nil
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}
