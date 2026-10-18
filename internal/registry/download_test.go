package registry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const rev = "0123456789abcdef0123456789abcdef01234567"

// origin serves files under /<repo>/resolve/<rev>/<path> with Range support.
// cut, when set, closes the connection after that many bytes once.
type origin struct {
	files    map[string][]byte
	requests atomic.Int32
	cut      atomic.Int64
	status   int
	gate     chan struct{}
}

func (o *origin) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	o.requests.Add(1)
	if o.gate != nil {
		<-o.gate
	}
	if o.status != 0 {
		w.WriteHeader(o.status)
		return
	}
	body, ok := o.files[r.URL.Path]
	if !ok {
		http.NotFound(w, r)
		return
	}
	if n := o.cut.Swap(0); n > 0 {
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(http.StatusOK)
		w.Write(body[:n])
		if hj, ok := w.(http.Hijacker); ok {
			c, _, _ := hj.Hijack()
			c.Close()
		}
		return
	}
	http.ServeContent(w, r, "f", time.Time{}, bytes.NewReader(body))
}

func fileFor(name string, body []byte) File {
	sum := sha256.Sum256(body)
	return File{Role: RoleModel, Repo: "org/repo", Path: name, Revision: rev, Size: int64(len(body)), SHA256: hex.EncodeToString(sum[:])}
}

func newOrigin(t *testing.T, files map[string][]byte) (*origin, *httptest.Server) {
	o := &origin{files: map[string][]byte{}}
	for name, b := range files {
		o.files["/org/repo/resolve/"+rev+"/"+name] = b
	}
	srv := httptest.NewTLSServer(o)
	t.Cleanup(srv.Close)
	return o, srv
}

func payload(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i * 7)
	}
	return b
}

func TestFetchResumes(t *testing.T) {
	body := payload(3 << 20)
	o, srv := newOrigin(t, map[string][]byte{"m.gguf": body})
	o.cut.Store(1 << 20)
	d := &Downloader{Client: srv.Client(), Primary: srv.URL}
	dest := filepath.Join(t.TempDir(), "m.gguf")
	var last Progress
	if err := d.Fetch(context.Background(), fileFor("m.gguf", body), dest, func(p Progress) { last = p }); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, body) {
		t.Fatal("content differs")
	}
	if o.requests.Load() != 2 || last.Percent != 100 {
		t.Fatalf("requests %d, last progress %+v", o.requests.Load(), last)
	}
	if _, err := os.Stat(dest + ".part"); err == nil {
		t.Fatal("partial left behind")
	}
}

func TestFetchMirrorAndRefusals(t *testing.T) {
	body := payload(1000)
	_, primary := newOrigin(t, nil)
	_, mirror := newOrigin(t, map[string][]byte{"m.gguf": body})
	d := &Downloader{Client: primary.Client(), Primary: primary.URL, Mirrors: []string{mirror.URL}}
	// Both test servers share one CA in httptest, so one client trusts both.
	dest := filepath.Join(t.TempDir(), "m.gguf")
	if err := d.Fetch(context.Background(), fileFor("m.gguf", body), dest, nil); err != nil {
		t.Fatal(err)
	}

	bad := fileFor("m.gguf", body)
	bad.SHA256 = strings.Repeat("0", 64)
	d2 := &Downloader{Client: mirror.Client(), Primary: mirror.URL, Retries: 1}
	err := d2.Fetch(context.Background(), bad, filepath.Join(t.TempDir(), "x"), nil)
	if err == nil || !strings.Contains(err.Error(), "checksum mismatch") {
		t.Fatalf("mismatch: %v", err)
	}

	d3 := &Downloader{Primary: "http://example.com"}
	if err := d3.Fetch(context.Background(), fileFor("m.gguf", body), filepath.Join(t.TempDir(), "x"), nil); err == nil || !strings.Contains(err.Error(), "https") {
		t.Fatalf("plain http must be refused: %v", err)
	}

	unpinned := fileFor("m.gguf", body)
	unpinned.Revision = "main"
	if err := d2.Fetch(context.Background(), unpinned, filepath.Join(t.TempDir(), "x"), nil); err == nil {
		t.Fatal("unpinned file must be refused")
	}
}

func TestFetchNotFoundIsPermanent(t *testing.T) {
	o, srv := newOrigin(t, nil)
	d := &Downloader{Client: srv.Client(), Primary: srv.URL, Retries: 3}
	err := d.Fetch(context.Background(), fileFor("m.gguf", payload(10)), filepath.Join(t.TempDir(), "x"), nil)
	if err == nil || o.requests.Load() != 1 {
		t.Fatalf("404 must not be retried: %v after %d requests", err, o.requests.Load())
	}
}

func manifestFor(files ...File) Manifest {
	return Manifest{ID: "tiny", Format: FormatGGUF, Commercial: CommercialPermitted, Files: files}
}

func TestStorePullStatusRemove(t *testing.T) {
	model, drafter := payload(5000), payload(700)
	_, srv := newOrigin(t, map[string][]byte{"m.gguf": model, "d.gguf": drafter})
	df := fileFor("d.gguf", drafter)
	df.Role = RoleDrafter
	m := manifestFor(fileFor("m.gguf", model), df)
	s := &Store{Root: t.TempDir(), Downloader: &Downloader{Client: srv.Client(), Primary: srv.URL}}

	if st, _ := s.Status(m); st.State != StateAbsent {
		t.Fatalf("fresh store: %s", st.State)
	}
	if err := s.Pull(context.Background(), m, nil); err != nil {
		t.Fatal(err)
	}
	mp, dp, err := s.Ready(m)
	if err != nil || filepath.Base(mp) != "m.gguf" || filepath.Base(dp) != "d.gguf" {
		t.Fatalf("ready: %s %s %v", mp, dp, err)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(mp), ManifestFile)); err != nil {
		t.Fatal("manifest not written beside the weights")
	}

	// Touching a file in place invalidates the stamp until pull rehashes it.
	future := time.Now().Add(time.Hour)
	os.Chtimes(mp, future, future)
	if st, _ := s.Status(m); st.State != StateCorrupt {
		t.Fatalf("touched file: %s", st.State)
	}
	if err := s.Pull(context.Background(), m, nil); err != nil {
		t.Fatal(err)
	}
	if st, _ := s.Status(m); st.State != StateComplete {
		t.Fatalf("after rehash: %s", st.State)
	}

	if err := s.Remove("tiny"); err != nil {
		t.Fatal(err)
	}
	if st, _ := s.Status(m); st.State != StateAbsent {
		t.Fatalf("after remove: %s", st.State)
	}
	if err := s.Remove("../x"); err == nil {
		t.Fatal("path escape must be refused")
	}
}

func TestStoreStrictLicense(t *testing.T) {
	m := manifestFor(fileFor("m.gguf", payload(10)))
	m.Commercial = CommercialUnconfirmed
	s := &Store{Root: t.TempDir(), StrictLicense: true}
	if err := s.Pull(context.Background(), m, nil); err == nil || !strings.Contains(err.Error(), "strict_license") {
		t.Fatalf("strict license: %v", err)
	}
}

func TestStoreSingleFlight(t *testing.T) {
	body := payload(4 << 20)
	o, srv := newOrigin(t, map[string][]byte{"m.gguf": body})
	o.gate = make(chan struct{})
	m := manifestFor(fileFor("m.gguf", body))
	s := &Store{Root: t.TempDir(), Downloader: &Downloader{Client: srv.Client(), Primary: srv.URL}, MaxConcurrent: 4}

	var wg sync.WaitGroup
	errs := make([]error, 3)
	var seen [3]atomic.Bool
	for i := range errs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = s.Pull(context.Background(), m, func(Progress) { seen[i].Store(true) })
		}()
	}
	time.Sleep(100 * time.Millisecond)
	close(o.gate)
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatal(i, err)
		}
	}
	if n := o.requests.Load(); n != 1 {
		t.Fatalf("%d requests for one file", n)
	}
	for i := range seen {
		if !seen[i].Load() {
			t.Fatalf("caller %d saw no progress", i)
		}
	}
}

func TestStorePullCancel(t *testing.T) {
	body := payload(1 << 20)
	o, srv := newOrigin(t, map[string][]byte{"m.gguf": body})
	o.gate = make(chan struct{})
	defer close(o.gate)
	m := manifestFor(fileFor("m.gguf", body))
	s := &Store{Root: t.TempDir(), Downloader: &Downloader{Client: srv.Client(), Primary: srv.URL}}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := s.Pull(ctx, m, nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("cancel: %v", err)
	}
}
