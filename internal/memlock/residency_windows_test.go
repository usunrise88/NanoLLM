//go:build windows

package memlock

import "testing"

// TestLockSurvivesTrim checks VirtualLock does what range-lock needs on
// Windows: after the working set is emptied, the locked view is still in it.
func TestLockSurvivesTrim(t *testing.T) {
	path := tempFile(t, 16<<20)
	pin, err := PinFile(path, []Range{{0, 4 << 20}})
	if err != nil {
		t.Skipf("locking is not permitted here: %v", err)
	}
	defer pin.Close()
	view := pin.viewBytes(0)
	if err := evict(nil, 0, 0); err != nil {
		t.Fatal(err)
	}
	res, err := Resident(view)
	if err != nil {
		t.Fatal(err)
	}
	if res < 4<<20 {
		t.Fatalf("only %d of %d locked bytes in the working set after a trim", res, 4<<20)
	}
}
