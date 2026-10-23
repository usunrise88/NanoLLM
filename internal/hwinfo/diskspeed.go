package hwinfo

import (
	"context"
	"errors"
	"io"
	"math/rand/v2"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// DiskSpeed is how fast a file reads when the OS cache is out of the way.
type DiskSpeed struct {
	// SeqMiBs is the sequential read rate with several requests in flight, in
	// MiB/s; RandMiBs and RandIOPS are 64 KiB reads at random offsets, one at
	// a time, the access pattern of a model streamed from disk.
	SeqMiBs  float64 `json:"seq_mib_s"`
	RandMiBs float64 `json:"rand_mib_s"`
	RandIOPS float64 `json:"rand_iops"`
	// ReadBytes is how much the probe read, and Direct says whether the reads
	// bypassed the OS cache (they do on every supported OS; a false means the
	// platform refused and the numbers are those of a cached read).
	ReadBytes int64 `json:"read_bytes"`
	Direct    bool  `json:"direct"`
}

// Probe sizes. The sequential probe stops at SeqMaxBytes or SeqMaxTime; the
// random probe runs RandTime.
const (
	diskBlock     = 4 << 20
	diskDepth     = 4
	diskRandBlock = 64 << 10
	SeqMaxBytes   = 1 << 30
	SeqMaxTime    = 2 * time.Second
	RandTime      = time.Second
)

// directFile is a file opened so that reads skip the OS cache, with the buffer
// alignment it needs.
type directFile interface {
	// ReadAt fills p (whose address and length the OS finds aligned) from off.
	ReadAt(p []byte, off int64) (int, error)
	Close() error
}

// ReadDiskSpeed measures the sequential and random read rates of path
// without the OS cache and without administrator rights: the model file
// itself is the test file, so the numbers belong to the disk it lives on. The
// whole probe stays within SeqMaxTime + RandTime.
func ReadDiskSpeed(ctx context.Context, path string) (DiskSpeed, error) {
	var res DiskSpeed
	st, err := os.Stat(path)
	if err != nil {
		return res, err
	}
	size := st.Size()
	if size < 2*diskBlock {
		return res, errors.New("hwinfo: the file is too small to test the disk")
	}
	files := make([]directFile, diskDepth)
	for i := range files {
		f, direct, err := openDirect(path)
		if err != nil {
			for _, g := range files[:i] {
				_ = g.Close()
			}
			return res, err
		}
		files[i], res.Direct = f, direct
	}
	defer func() {
		for _, f := range files {
			_ = f.Close()
		}
	}()

	// Sequential: four readers take consecutive blocks in turn.
	var next, total atomic.Int64
	deadline := time.Now().Add(SeqMaxTime)
	start := time.Now()
	var wg sync.WaitGroup
	for i := range files {
		wg.Add(1)
		go func(f directFile) {
			defer wg.Done()
			buf := alignedBuffer(diskBlock)
			for ctx.Err() == nil && time.Now().Before(deadline) {
				off := (next.Add(1) - 1) * diskBlock
				if off+diskBlock > size || off >= SeqMaxBytes {
					return
				}
				n, err := f.ReadAt(buf, off)
				total.Add(int64(n))
				if err != nil && !errors.Is(err, io.EOF) {
					return
				}
			}
		}(files[i])
	}
	wg.Wait()
	if el := time.Since(start); total.Load() > 0 && el > 0 {
		res.SeqMiBs = float64(total.Load()) / (1 << 20) / el.Seconds()
	}
	res.ReadBytes = total.Load()
	if err := ctx.Err(); err != nil {
		return res, err
	}
	if res.ReadBytes == 0 {
		return res, errors.New("hwinfo: the disk probe read nothing")
	}

	// Random: one 64 KiB read at a time at aligned offsets.
	buf := alignedBuffer(diskRandBlock)
	blocks := (size - diskRandBlock) / diskRandBlock
	var n int64
	var read int64
	t0 := time.Now()
	for time.Since(t0) < RandTime && ctx.Err() == nil {
		off := rand.Int64N(blocks) * diskRandBlock
		got, err := files[0].ReadAt(buf, off)
		if err != nil && !errors.Is(err, io.EOF) {
			break
		}
		n++
		read += int64(got)
	}
	if el := time.Since(t0); n > 0 && el > 0 {
		res.RandIOPS = float64(n) / el.Seconds()
		res.RandMiBs = float64(read) / (1 << 20) / el.Seconds()
	}
	res.ReadBytes += read
	return res, ctx.Err()
}
