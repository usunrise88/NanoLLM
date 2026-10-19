// Command membw measures how fast this machine can read memory, per NUMA
// node and across nodes: the ceiling that CPU decoding runs into, since each
// generated token reads every active weight once (docs/STAGE2.md, OPT-3).
//
//	go run ./tools/membw                  # a table: local and remote reads, one node and all
//	go run ./tools/membw -mib 2048 -sec 3
//
// A token's weights are about 2.5 GiB for the E4B (G2 and G3), so the best
// tokens per second a placement can reach is its bandwidth divided by that;
// the gap to the measured speed is what OPT-6, OPT-10 and OPT-12 chase.
//
// It re-executes itself under each placement (internal/numa), reading a
// buffer of -mib MiB per thread that the thread itself first touched under
// the placement's memory policy, so where the pages land is the policy's
// doing. The kernel is plain Go: a few unrolled 64-bit loads per iteration,
// which streams from DRAM about as fast as a core can pull; it is a floor
// for what AVX2 code reaches, not a ceiling.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/usunrise88/nanollm/internal/hwinfo"
	"github.com/usunrise88/nanollm/internal/numa"
)

func main() {
	mib := flag.Int("mib", 1024, "buffer per thread, in MiB (larger than the caches)")
	sec := flag.Float64("sec", 2, "seconds to read per scenario")
	threads := flag.Int("threads", 0, "threads for a worker run (set by the parent)")
	worker := flag.Bool("worker", false, "one measurement, placed by $NANOLLM_PLACE")
	flag.Parse()

	if *worker {
		if raw := os.Getenv(numa.EnvPlace); raw != "" {
			sp, err := numa.Parse(raw, hwinfo.ParseCPUList)
			if err != nil {
				fatal(err)
			}
			var nodes []int
			for _, n := range hwinfo.NUMATopology() {
				nodes = append(nodes, n.ID)
			}
			if err := numa.Enter(sp, nodes); err != nil {
				fatal(err)
			}
		}
		fmt.Printf("%.3f\n", measure(*threads, *mib, *sec))
		return
	}

	topo := hwinfo.NUMATopology()
	if len(topo) == 0 {
		fmt.Printf("no NUMA topology; one scenario, %d threads\n", runtime.NumCPU())
		fmt.Printf("%-44s %7.1f GB/s\n", "all CPUs, default policy", child("", runtime.NumCPU(), *mib, *sec))
		return
	}
	fmt.Printf("%-44s %10s\n", "scenario", "GB/s")
	for _, n := range topo {
		cores := hwinfo.FirstThreads(n.CPUs)
		own := hwinfo.FormatCPUList(cores)
		fmt.Printf("%-44s %7.1f\n", fmt.Sprintf("node %d, %d cores, local memory", n.ID, len(cores)),
			child(fmt.Sprintf("node:%d;cpus=%s", n.ID, own), len(cores), *mib, *sec))
		if len(topo) > 1 {
			remote := topo[(n.ID+1)%len(topo)].ID
			fmt.Printf("%-44s %7.1f\n", fmt.Sprintf("node %d, %d cores, memory on node %d", n.ID, len(cores), remote),
				child(fmt.Sprintf("node:%d;cpus=%s", remote, own), len(cores), *mib, *sec))
		}
	}
	var all []int
	for _, n := range topo {
		all = append(all, hwinfo.FirstThreads(n.CPUs)...)
	}
	fmt.Printf("%-44s %7.1f\n", fmt.Sprintf("all %d cores, memory first-touched in place", len(all)), child("off;cpus="+hwinfo.FormatCPUList(all), len(all), *mib, *sec))
	if len(topo) > 1 {
		fmt.Printf("%-44s %7.1f\n", fmt.Sprintf("all %d cores, memory interleaved over %d nodes", len(all), len(topo)),
			child("interleave;cpus="+hwinfo.FormatCPUList(all), len(all), *mib, *sec))
	}
	fmt.Printf("\nweights read per token: about 2.5 GiB (E4B); tokens/s ceiling = GB/s / 2.7\n")
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "membw:", err)
	os.Exit(1)
}

// child runs one worker under a placement and returns its GB/s.
func child(place string, threads, mib int, sec float64) float64 {
	exe, err := os.Executable()
	if err != nil {
		fatal(err)
	}
	cmd := exec.CommandContext(context.Background(), exe, "-worker", "-threads", strconv.Itoa(threads), "-mib", strconv.Itoa(mib), "-sec", strconv.FormatFloat(sec, 'f', -1, 64))
	cmd.Env = os.Environ()
	if place != "" {
		cmd.Env = append(cmd.Env, numa.EnvPlace+"="+place)
	}
	out, err := cmd.Output()
	if err != nil {
		fatal(fmt.Errorf("worker (%s): %w", place, err))
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil {
		fatal(fmt.Errorf("worker (%s) printed %q", place, out))
	}
	return v
}

// measure reads a private buffer on each of n threads for sec seconds and
// returns the total GB/s (10^9 bytes).
func measure(n, mib int, sec float64) float64 {
	words := mib << 20 / 8
	bufs := make([][]uint64, n)
	var wg sync.WaitGroup
	for i := range bufs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			runtime.LockOSThread() // first touch from the thread that will read
			b := make([]uint64, words)
			for j := range b {
				b[j] = uint64(j)
			}
			bufs[i] = b
		}()
	}
	wg.Wait()

	start := make(chan struct{})
	var bytes uint64
	var mu sync.Mutex
	var sink uint64
	deadline := time.Duration(sec * float64(time.Second))
	for i := range bufs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			runtime.LockOSThread()
			<-start
			var local, sum uint64
			t0 := time.Now()
			for time.Since(t0) < deadline {
				sum += read(bufs[i])
				local += uint64(len(bufs[i])) * 8
			}
			mu.Lock()
			bytes += local
			sink += sum
			mu.Unlock()
		}()
	}
	t0 := time.Now()
	close(start)
	wg.Wait()
	elapsed := time.Since(t0).Seconds()
	if sink == 42 {
		fmt.Fprintln(os.Stderr) // keep the sum alive
	}
	return float64(bytes) / elapsed / 1e9
}

// read sums b with eight independent accumulators, so loads are not chained
// behind additions.
func read(b []uint64) uint64 {
	var s0, s1, s2, s3, s4, s5, s6, s7 uint64
	p := unsafe.Pointer(unsafe.SliceData(b))
	n := len(b) &^ 7
	for i := 0; i < n; i += 8 {
		q := (*[8]uint64)(unsafe.Add(p, i*8))
		s0 += q[0]
		s1 += q[1]
		s2 += q[2]
		s3 += q[3]
		s4 += q[4]
		s5 += q[5]
		s6 += q[6]
		s7 += q[7]
	}
	return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
}
