package posture

import (
	"runtime"
	"testing"
)

func TestApplyAndBack(t *testing.T) {
	before := Current()
	if before.Low {
		t.Skip("the test process already runs below normal priority")
	}
	if err := Apply(true); err != nil {
		t.Fatalf("Apply(true): %v", err)
	}
	if !Current().Low {
		t.Fatalf("not below normal after Apply(true): %+v", Current())
	}
	if runtime.GOOS == "windows" && !Current().Eco {
		t.Logf("EcoQoS is not reported on: %+v (older Windows?)", Current())
	}
	err := Apply(false)
	if runtime.GOOS == "windows" {
		if err != nil || Current().Low {
			t.Fatalf("Apply(false): %v, %+v", err, Current())
		}
		return
	}
	// Elsewhere a priority cannot be taken back, and the caller is told.
	if err != ErrRestart {
		t.Fatalf("Apply(false) after Apply(true): %v, want ErrRestart", err)
	}
	// The test process stays low for the rest of its life: acceptable.
}
