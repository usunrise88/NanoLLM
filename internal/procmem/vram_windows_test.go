//go:build windows

package procmem

import (
	"os"
	"strconv"
	"testing"
)

func TestGPUProcessInstances(t *testing.T) {
	s := os.Getenv("NANOLLM_TEST_PID")
	if s == "" {
		t.Skip("set NANOLLM_TEST_PID to a process that uses the GPU")
	}
	pid, _ := strconv.Atoi(s)
	for _, name := range []string{"Dedicated Usage", "Shared Usage", "Total Committed"} {
		inst, err := gpuProcessInstances(pid, name)
		t.Logf("%s: %v %v", name, inst, err)
	}
}
