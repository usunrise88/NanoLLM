package procmem

import (
	"errors"
	"os"
	"strconv"
	"testing"
)

func TestMeterSeesAnAllocation(t *testing.T) {
	if Private() <= 0 {
		t.Fatal("no private memory read for this process")
	}
	m := Start()
	before := m.Peak()
	// Touch every page so the memory is committed and resident.
	block := make([]byte, 96<<20)
	for i := 0; i < len(block); i += 4096 {
		block[i] = 1
	}
	peak := m.Stop()
	if peak < before+64<<20 {
		t.Fatalf("peak %d MiB after a 96 MiB allocation, before %d MiB", peak>>20, before>>20)
	}
	if block[0] != 1 {
		t.Fatal("unreachable")
	}
}

// TestProcessVRAMOfThisProcess: a test process holds no GPU memory; the
// counters must answer that, or say they are unavailable.
func TestProcessVRAMOfThisProcess(t *testing.T) {
	v, err := ProcessVRAM(os.Getpid())
	if err != nil && !errors.Is(err, ErrUnavailable) {
		t.Fatal(err)
	}
	if err == nil && v.Dedicated != 0 {
		t.Fatalf("a test process holds %d MiB of VRAM", v.Dedicated>>20)
	}
}

// TestProcessVRAMOfAPID prints what the counters say about the process named
// by NANOLLM_TEST_PID (a GPU user, for example the nanollm server).
func TestProcessVRAMOfAPID(t *testing.T) {
	s := os.Getenv("NANOLLM_TEST_PID")
	if s == "" {
		t.Skip("set NANOLLM_TEST_PID to a process that uses the GPU")
	}
	pid, _ := strconv.Atoi(s)
	v, err := ProcessVRAM(pid)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("pid %d: dedicated %d MiB, shared %d MiB (%s)", pid, v.Dedicated>>20, v.Shared>>20, v.Source)
}
