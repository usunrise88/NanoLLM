//go:build linux

package numa

import (
	"errors"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// maxNodes is how many node bits a mask carries.
const maxNodes = 1024

// Default reports whether the process runs under the kernel's default memory
// policy: a person who started it under numactl has chosen, and keeps the
// choice.
func Default() bool {
	var mode int32
	_, _, e := unix.Syscall6(unix.SYS_GET_MEMPOLICY, uintptr(unsafe.Pointer(&mode)), 0, 0, 0, 0, 0)
	return e == 0 && mode&0xff == unix.MPOL_DEFAULT
}

// setPolicy applies the spec to the calling thread.
func setPolicy(s Spec, nodes []int) error {
	if len(s.CPUs) > 0 {
		var set unix.CPUSet
		for _, c := range s.CPUs {
			set.Set(c)
		}
		if err := unix.SchedSetaffinity(0, &set); err != nil {
			return err
		}
	}
	var mode uintptr
	switch s.Mode {
	case "interleave":
		mode = unix.MPOL_INTERLEAVE
	case "node":
		mode, nodes = unix.MPOL_BIND, []int{s.Node}
	default:
		return nil
	}
	mask := make([]uint64, maxNodes/64)
	for _, n := range nodes {
		if n < 0 || n >= maxNodes {
			return errors.New("numa: node number out of range")
		}
		mask[n/64] |= 1 << (n % 64)
	}
	_, _, e := unix.Syscall(unix.SYS_SET_MEMPOLICY, mode, uintptr(unsafe.Pointer(&mask[0])), maxNodes)
	if e != 0 {
		return e
	}
	return nil
}

// Enter places this process and re-executes it so that every thread starts
// under the placement. nodes are the memory nodes to interleave over. It
// returns only on failure or when there is nothing to do; on success the
// process image is replaced.
func Enter(s Spec, nodes []int) error {
	if s.IsZero() || os.Getenv(EnvApplied) != "" {
		return nil
	}
	if s.Mode != "off" && s.Mode != "" && s.Mode != "distribute" && !Default() {
		return nil // already under a policy someone chose
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	runtime.LockOSThread() // never unlocked: this thread execs, or the process goes on without a placement
	if err := setPolicy(s, nodes); err != nil {
		runtime.UnlockOSThread()
		return err
	}
	return syscall.Exec(exe, os.Args, execEnv(s, os.Environ()))
}

// execEnv is the environment of the re-executed process: the marker that it
// has been placed and, for "distribute", ggml's strategy and the repack thread
// count unless the person who started it set them.
func execEnv(s Spec, env []string) []string {
	has := func(k string) bool {
		for _, e := range env {
			if strings.HasPrefix(e, k+"=") {
				return true
			}
		}
		return false
	}
	out := append(slices.Clone(env), EnvApplied+"=1")
	if s.Mode == "distribute" {
		if !has(EnvGGML) {
			out = append(out, EnvGGML+"=distribute")
		}
		if s.Threads > 0 && !has(EnvRepackThreads) {
			out = append(out, EnvRepackThreads+"="+strconv.Itoa(s.Threads))
		}
	}
	return out
}
