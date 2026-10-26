package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"strings"
	"sync"
	"time"
)

// Progress is one update of a download in flight.
type Progress struct {
	File       string  `json:"file"`
	Downloaded int64   `json:"downloaded"`
	Total      int64   `json:"total"`
	Percent    float64 `json:"percent"`
}

// Downloader fetches pinned files, resuming partial ones, and refuses any
// byte it cannot verify against the catalog's sha256.
type Downloader struct {
	// Client is used as is, except that redirects are held to https.
	Client *http.Client
	// Mirrors are Hugging Face compatible base URLs, tried in order after the
	// primary: <base>/<repo>/resolve/<revision>/<path>.
	Mirrors []string
	// Retries per source. A dropped connection resumes where it stopped.
	Retries int
	// IdleTimeout aborts a transfer that delivers nothing for this long. A
	// timeout on the whole request would cut off a 4 GB download on a slow
	// line that is working fine.
	IdleTimeout time.Duration
	// Primary is the default base URL, https://huggingface.co.
	Primary string

	once sync.Once
	// mu guards Mirrors once SetMirrors may change them under a download.
	mu sync.Mutex
}

// SetMirrors replaces the mirrors from the next download on: nanollm managed
// takes the application's new list without a restart.
func (d *Downloader) SetMirrors(mirrors []string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.Mirrors = append([]string(nil), mirrors...)
}

// DefaultPrimary is where pinned files come from unless a mirror answers.
const DefaultPrimary = "https://huggingface.co"

func (d *Downloader) init() {
	d.once.Do(func() {
		if d.Client == nil {
			d.Client = &http.Client{}
		}
		if d.Retries <= 0 {
			d.Retries = 3
		}
		if d.IdleTimeout <= 0 {
			d.IdleTimeout = 60 * time.Second
		}
		if d.Primary == "" {
			d.Primary = DefaultPrimary
		}
		// Redirects may change host — Hugging Face serves LFS objects from a
		// CDN — but never drop to plaintext. Integrity rests on the sha256,
		// which no redirect can forge; https keeps the transfer private.
		c := *d.Client
		c.CheckRedirect = func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			if req.URL.Scheme != "https" {
				return fmt.Errorf("refusing redirect to %s: downloads must stay on https", req.URL.Scheme)
			}
			return nil
		}
		d.Client = &c
	})
}

// sources lists the URLs a file can come from, primary first.
func (d *Downloader) sources(f File) ([]string, error) {
	if !f.Pinned() {
		return nil, fmt.Errorf("%s/%s is not pinned to a commit, size and sha256", f.Repo, f.Path)
	}
	d.mu.Lock()
	bases := append([]string{d.Primary}, d.Mirrors...)
	d.mu.Unlock()
	var out []string
	for _, base := range bases {
		u, err := url.Parse(base)
		if err != nil || u.Scheme != "https" || u.Host == "" {
			return nil, fmt.Errorf("download source %q must be an https URL", base)
		}
		out = append(out, strings.TrimSuffix(base, "/")+"/"+f.Repo+"/resolve/"+f.Revision+"/"+f.Path)
	}
	return out, nil
}

// Fetch downloads f to dest. A partial dest (dest+".part") is resumed; dest
// appears only once its sha256 matches. onProgress may be nil.
func (d *Downloader) Fetch(ctx context.Context, f File, dest string, onProgress func(Progress)) error {
	d.init()
	urls, err := d.sources(f)
	if err != nil {
		return err
	}
	part := dest + ".part"
	var lastErr error
	for _, u := range urls {
		for attempt := range d.Retries {
			if err := ctx.Err(); err != nil {
				return err
			}
			if attempt > 0 {
				select {
				case <-time.After(time.Duration(1<<attempt) * time.Second):
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			err := d.attempt(ctx, f, u, part, onProgress)
			if err == nil {
				return os.Rename(part, dest)
			}
			lastErr = err
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if isPermanent(err) {
				// This source will not improve on a retry; a mirror might.
				break
			}
		}
	}
	return fmt.Errorf("download %s from any source: %w", path.Base(f.Path), lastErr)
}

func (d *Downloader) attempt(ctx context.Context, f File, rawURL, part string, onProgress func(Progress)) error {
	hasher := sha256.New()
	offset, err := hashExisting(part, hasher)
	if err != nil {
		return err
	}
	if offset > f.Size {
		// More bytes than the pinned file has: the partial is not a prefix of
		// it, whatever it is.
		_ = os.Remove(part)
		hasher.Reset()
		offset = 0
	}
	if offset == f.Size {
		return verify(hasher, f, part)
	}

	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	resp, err := d.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// The server ignored the range: start over rather than append to a
		// prefix that is about to be sent again.
		if offset > 0 {
			hasher.Reset()
			offset = 0
		}
	case http.StatusPartialContent:
	case http.StatusNotFound, http.StatusForbidden, http.StatusGone, http.StatusUnauthorized:
		return permanentf("%s is not available at %s (%s)", path.Base(f.Path), rawURL, resp.Status)
	case http.StatusRequestedRangeNotSatisfiable:
		_ = os.Remove(part)
		return fmt.Errorf("server refused to resume %s; starting over", path.Base(f.Path))
	default:
		return fmt.Errorf("unexpected status %s from %s", resp.Status, rawURL)
	}
	if resp.ContentLength > 0 && offset+resp.ContentLength != f.Size {
		return permanentf("%s: server offers %d bytes, the pinned file has %d",
			path.Base(f.Path), offset+resp.ContentLength, f.Size)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if offset > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	out, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	idle := time.AfterFunc(d.IdleTimeout, func() {
		cancel(fmt.Errorf("no data for %s", d.IdleTimeout))
	})
	defer idle.Stop()

	err = copyProgress(io.MultiWriter(out, hasher), resp.Body, f, offset, idle, d.IdleTimeout, onProgress)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		if cause := context.Cause(ctx); cause != nil && !errors.Is(cause, context.Canceled) {
			return cause
		}
		return err
	}
	return verify(hasher, f, part)
}

func verify(hasher interface{ Sum([]byte) []byte }, f File, part string) error {
	got := hex.EncodeToString(hasher.Sum(nil))
	if got != f.SHA256 {
		// The bytes are worthless; keeping them would make the next attempt
		// resume onto a corrupt prefix.
		_ = os.Remove(part)
		return permanentf("checksum mismatch for %s: got %s, expected %s", path.Base(f.Path), got, f.SHA256)
	}
	return nil
}

// hashExisting feeds a partial download into the hasher, so a resumed file
// still yields the checksum of the whole.
func hashExisting(p string, h io.Writer) (int64, error) {
	f, err := os.Open(p)
	if errors.Is(err, os.ErrNotExist) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return io.Copy(h, f)
}

// copyProgress streams the body, holding it to the pinned size and reporting
// at most four times a second.
func copyProgress(dst io.Writer, src io.Reader, f File, done int64, idle *time.Timer, idleFor time.Duration, onProgress func(Progress)) error {
	buf := make([]byte, 256<<10)
	var last time.Time
	report := func() {
		if onProgress != nil {
			onProgress(Progress{File: path.Base(f.Path), Downloaded: done, Total: f.Size, Percent: percent(done, f.Size)})
		}
	}
	for {
		n, err := src.Read(buf)
		if n > 0 {
			idle.Reset(idleFor)
			if done+int64(n) > f.Size {
				return permanentf("%s is longer than the pinned %d bytes", path.Base(f.Path), f.Size)
			}
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return werr
			}
			done += int64(n)
			if time.Since(last) > 250*time.Millisecond {
				last = time.Now()
				report()
			}
		}
		if err == io.EOF {
			report()
			if done != f.Size {
				return fmt.Errorf("%s: connection closed at %d of %d bytes", path.Base(f.Path), done, f.Size)
			}
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func percent(done, total int64) float64 {
	if total <= 0 {
		return 0
	}
	return float64(done) / float64(total) * 100
}

// permanentError is a failure that retrying the same source cannot fix. The
// next source is still worth trying.
type permanentError struct{ error }

func (e permanentError) Unwrap() error { return e.error }

func permanentf(format string, args ...any) error {
	return permanentError{fmt.Errorf(format, args...)}
}

func isPermanent(err error) bool {
	var p permanentError
	return errors.As(err, &p)
}
