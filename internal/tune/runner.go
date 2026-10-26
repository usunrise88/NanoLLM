package tune

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/numa"
	"github.com/usunrise88/nanollm/internal/quietexec"
)

// Runner runs one probe: loads the model as the spec says and measures it.
type Runner interface {
	Probe(ctx context.Context, spec ProbeSpec) (ProbeResult, error)
}

// InProcessRunner probes in the calling process on a backend it already holds.
// The tests use it with a scripted backend; a real run uses SpawnRunner, so
// that every load has a process of its own with its own peak memory.
type InProcessRunner struct {
	Backend backend.Backend
}

func (r InProcessRunner) Probe(ctx context.Context, spec ProbeSpec) (ProbeResult, error) {
	return RunProbe(ctx, r.Backend, spec)
}

// SpawnRunner runs `nanollm optimize worker` as a child process: the spec goes
// in on stdin, the result comes out on stdout, and a crash of the worker is a
// failed probe, not a failed autotuner.
type SpawnRunner struct {
	// Exe is the nanollm binary; empty is the running one.
	Exe string
	// Env is added to the worker's environment.
	Env []string
	// LoadAllowance is how long the load itself may take on top of the spec's
	// budget before the worker is killed; zero is two minutes.
	LoadAllowance time.Duration
}

// WorkerArgs are the arguments that make the binary a probe worker.
var WorkerArgs = []string{"optimize", "worker"}

func (r SpawnRunner) Probe(ctx context.Context, spec ProbeSpec) (ProbeResult, error) {
	var res ProbeResult
	exe := r.Exe
	if exe == "" {
		var err error
		if exe, err = os.Executable(); err != nil {
			return res, err
		}
	}
	allowance := r.LoadAllowance
	if allowance == 0 {
		allowance = 2 * time.Minute
	}
	ctx, cancel := context.WithTimeout(ctx, spec.Budget+allowance)
	defer cancel()
	in, err := json.Marshal(spec)
	if err != nil {
		return res, err
	}
	cmd := quietexec.Command(ctx, exe, WorkerArgs...)
	cmd.Env = append(os.Environ(), r.Env...)
	for k, v := range spec.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.Stdin = bytes.NewReader(in)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	runErr := cmd.Run()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return res, fmt.Errorf("the probe did not finish in %s", spec.Budget+allowance)
	}
	if jerr := json.Unmarshal(stdout.Bytes(), &res); jerr != nil {
		tail := strings.TrimSpace(stderr.String())
		if i := strings.LastIndex(tail, "\n"); i >= 0 {
			tail = tail[i+1:]
		}
		if runErr == nil {
			runErr = jerr
		}
		return res, fmt.Errorf("worker: %w: %s", runErr, tail)
	}
	if res.Error != "" {
		return res, errors.New(res.Error)
	}
	return res, nil
}

// WorkerMain is the body of `nanollm optimize worker`: it reads the spec from
// stdin, probes, and writes the result as JSON to stdout. The caller has
// applied the process placement (numa.EnvPlace) before this is called.
func WorkerMain(ctx context.Context, in *os.File, out *os.File, open func(spec ProbeSpec) (backend.Backend, error)) error {
	var spec ProbeSpec
	if err := json.NewDecoder(in).Decode(&spec); err != nil {
		return fmt.Errorf("optimize worker: read the spec: %w", err)
	}
	b, err := open(spec)
	if err != nil {
		return json.NewEncoder(out).Encode(ProbeResult{Error: err.Error()})
	}
	defer func() { _ = b.Close() }()
	res, err := RunProbe(ctx, b, spec)
	if err != nil {
		res.Error = err.Error()
	}
	return json.NewEncoder(out).Encode(res)
}

// PlaceEnv is the environment entry that places a worker on NUMA nodes
// (internal/numa) for a placement spec like "interleave" or "node:1".
func PlaceEnv(spec string) string { return numa.EnvPlace + "=" + spec }
