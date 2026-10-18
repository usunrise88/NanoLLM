package memlock

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestAlign(t *testing.T) {
	p := PageSize
	got := align([]Range{{p*3 + 10, 5}, {0, 1}, {p, p}, {p*2 - 1, 2}, {p * 100, 10}, {5, 0}}, p*10)
	// Overlapping and adjacent ranges merge; out-of-file and empty ones go.
	want := []Range{{0, p * 4}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	if Total(got) != 4*p {
		t.Fatal("total")
	}
}

func tempFile(t *testing.T, size int64) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "model.bin")
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 1<<20)
	for i := range buf {
		buf[i] = byte(i)
	}
	for written := int64(0); written < size; written += int64(len(buf)) {
		if _, err := f.Write(buf[:min(int64(len(buf)), size-written)]); err != nil {
			t.Fatal(err)
		}
	}
	// Written-back pages, so the kernel can drop them: dirty ones would be
	// kept by MADV_PAGEOUT and make the residency test vacuous.
	if err := f.Sync(); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return p
}

func TestPinAndClose(t *testing.T) {
	path := tempFile(t, 8<<20)
	pin, err := PinFile(path, []Range{{0, 1 << 20}, {4 << 20, 1 << 20}})
	if err != nil {
		t.Skipf("locking is not permitted here: %v", err)
	}
	if pin.Locked() != 2<<20 {
		t.Fatalf("locked %d", pin.Locked())
	}
	if err := pin.Close(); err != nil {
		t.Fatal(err)
	}
	if pin.Locked() != 0 {
		t.Fatal("still locked after close")
	}
}
