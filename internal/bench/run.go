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
	"time"

	"github.com/usunrise88/nanollm/benchfmt"
	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/memlock"
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

	Suite string // standard | dev
	// Full runs every applicable preset; otherwise auto and two
	// representative ones.
	Full bool
	// Presets, when set, are run instead of the suite's choice.
	Presets []string
	Reps    int

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
	if f, ok := o.Manifest.File(registry.RoleModel); ok {
		res.Model = benchfmt.Model{ID: o.Manifest.ID, Revision: f.Revision, SHA256: f.SHA256}
	}
	if f, ok := o.Manifest.File(registry.RoleDrafter); ok && o.DrafterPath != "" {
		res.Model.DrafterSHA256 = f.SHA256
	}

	seen := map[string]string{} // plan hash → preset that ran it
	var measured time.Duration
	nMeasured := 0
	for i, name := range names {
		if err := ctx.Err(); err != nil {
			return res, err
		}
		o.emit(Event{Type: "plan", Preset: name, Index: i, Total: len(names)})
		run, spec, refused := planRun(o, model, name, cpu, gpu)
		if refused {
			res.Runs = append(res.Runs, run)
			o.emit(Event{Type: "result", Preset: name, Index: i, Total: len(names), Run: &run})
			continue
		}
		if prev, ok := seen[run.PlanHash]; ok {
			run.Status, run.SameAs = benchfmt.StatusSkipped, prev
			run.Reason = "same plan as " + prev
			res.Runs = append(res.Runs, run)
			o.emit(Event{Type: "result", Preset: name, Index: i, Total: len(names), Run: &run})
			continue
		}
		seen[run.PlanHash] = name
		o.emit(Event{Type: "start", Preset: name, Index: i, Total: len(names)})
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
		o.emit(Event{Type: "result", Preset: name, Index: i, Total: len(names), Run: &run, ElapsedMS: ms(time.Since(t0))})
		if left := len(names) - i - 1; left > 0 {
			o.emit(Event{Type: "estimate", Index: i, Total: len(names), RemainS: measured.Seconds() / float64(nMeasured) * float64(left)})
		}
	}
	res.DurationS = time.Since(started).Seconds()
	o.emit(Event{Type: "done", Total: len(names)})
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
func planRun(o Options, model plan.Model, name string, cpu, gpu *backend.Device) (benchfmt.Run, WorkerSpec, bool) {
	run := benchfmt.Run{Preset: name}
	preset, err := plan.ParsePreset(name)
	if err != nil {
		run.Status, run.Reason = benchfmt.StatusRefused, err.Error()
		return run, WorkerSpec{}, true
	}
	req := plan.Request{Model: model, Preset: preset, Proven: plan.ProvenFor(runtime.GOOS), NoMTP: o.DrafterPath == ""}
	if (preset.Budget.VRAM > 0 || preset.Budget.Unified > 0) && gpu != nil {
		req.Device = &plan.Device{
			Selector: gpu.Selector(), Backend: strings.TrimRight(gpu.ID, "0123456789"),
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
	spec, err := p.LoadSpec(o.ModelPath, o.DrafterPath)
	if err != nil {
		run.Status, run.Reason = benchfmt.StatusRefused, err.Error()
		return run, WorkerSpec{}, true
	}
	if p.Device == nil {
		spec.Device = cpu.Selector()
	}
	ws := WorkerSpec{Load: spec}
	for _, lr := range p.LockRanges {
		r := memlock.Range{Off: lr.Offset, Len: lr.Length}
		if lr.Role == registry.RoleDrafter {
			ws.DrafterLocks = append(ws.DrafterLocks, r)
		} else {
			ws.Locks = append(ws.Locks, r)
		}
	}
	run.PlanHash = planHash(ws)
	run.Plan = benchfmt.Plan{
		Mode: string(p.Mode), Device: spec.Device, GPULayers: p.GPULayers, NCtx: p.NCtx, Slots: p.Slots,
		KV: p.KV.TypeK, FA: p.FlashAttention, SWAFull: p.KV.SWAFull, MTP: p.MTP.Enabled, Threads: spec.Threads,
	}
	return run, ws, false
}

// spawn runs one worker. On Linux a RAM budget is enforced with a transient
// systemd scope when the user manager is there (strict); otherwise the
// budget is only planned for (advisory). Windows Job Objects arrive with
// stage 3.
func (o Options) spawn(ctx context.Context, ws WorkerSpec, budget plan.Budget) (WorkerResult, string, error) {
	ws.Backend, ws.LibDir, ws.Reps, ws.Quick = o.Backend.Name(), o.LibDir, o.Reps, !o.Full
	in, _ := json.Marshal(ws)
	args := []string{"bench", "worker"}
	name := o.Exe
	enforcement := "advisory"
	if runtime.GOOS == "linux" && budget.RAM > 0 && userScopes() {
		limit := budget.RAM + plan.HostOverhead
		args = append([]string{"--user", "--scope", "--quiet", "-p", "MemoryMax=" + strconv.FormatInt(limit, 10), "-p", "MemorySwapMax=0", "--", o.Exe}, args...)
		name, enforcement = "systemd-run", "strict"
	}
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdin = bytes.NewReader(in)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	var wr WorkerResult
	if jerr := json.Unmarshal(stdout.Bytes(), &wr); jerr != nil {
		tail := strings.TrimSpace(stderr.String())
		if i := strings.LastIndex(tail, "\n"); i >= 0 {
			tail = tail[i+1:]
		}
		if err == nil {
			err = jerr
		}
		return wr, enforcement, fmt.Errorf("worker: %v: %s", err, tail)
	}
	if wr.Error != "" {
		return wr, enforcement, errors.New(wr.Error)
	}
	return wr, enforcement, nil
}

// userScopes reports whether systemd-run can make a scope for this user.
func userScopes() bool {
	if _, err := exec.LookPath("systemd-run"); err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return exec.CommandContext(ctx, "systemd-run", "--user", "--scope", "--quiet", "true").Run() == nil
}

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
