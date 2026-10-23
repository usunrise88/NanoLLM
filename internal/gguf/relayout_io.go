package gguf

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"time"
)

// copyChunk is how much of a tensor is read and written at a time: the 4 GB
// of a model never sit in memory.
const copyChunk = 4 << 20

// fixedStart is the size of the header before the metadata: magic, version,
// tensor count and metadata count.
const fixedStart = 4 + 4 + 8 + 8

// encodeHeader is everything before the data area: the source's metadata with
// general.alignment set (added when the source lacks it), the tensor table in
// the new order, and zeros up to DataOffset. r reads the source file.
func (l *Layout) encodeHeader(f *File, r io.ReaderAt) ([]byte, error) {
	if l.Approximate {
		return nil, errors.New("relayout: a plan made from an inventory cannot write a file")
	}
	kv := make([]byte, f.KVEnd-fixedStart)
	if n, err := r.ReadAt(kv, fixedStart); n < len(kv) {
		return nil, fmt.Errorf("relayout: read the metadata: %w", cmpErr(err))
	}
	if off, ok := f.AlignmentOffset(); ok {
		binary.LittleEndian.PutUint32(kv[off-fixedStart:], uint32(l.Alignment))
	}
	var b bytes.Buffer
	b.Grow(int(l.DataOffset))
	le := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { le(uint64(len(s))); b.WriteString(s) }
	nKV := int64(len(f.KV))
	if l.source.addAlign {
		nKV++
	}
	b.WriteString(Magic)
	le(l.source.version)
	le(int64(len(l.Tensors)))
	le(nKV)
	b.Write(kv)
	if l.source.addAlign {
		str(KeyAlignment)
		le(uint32(ValUint32))
		le(uint32(l.Alignment))
	}
	for _, p := range l.Tensors {
		str(p.Name)
		le(uint32(len(p.Shape)))
		for _, d := range p.Shape {
			le(d)
		}
		le(uint32(p.Type))
		le(uint64(p.Offset - l.DataOffset))
	}
	want := l.source.headEnd
	if l.source.addAlign {
		want += alignKVSize
	}
	if int64(b.Len()) != want || int64(b.Len()) > l.DataOffset {
		return nil, fmt.Errorf("relayout: the header is %d bytes, planned %d (data at %d)", b.Len(), want, l.DataOffset)
	}
	b.Write(make([]byte, l.DataOffset-int64(b.Len())))
	return b.Bytes(), nil
}

func cmpErr(err error) error {
	if err == nil || errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}

// writeTo writes the whole relaid file to w: header, then every tensor's bytes
// from src at its new offset, zeros in the padding. onTensor, when set, is
// called before each tensor's bytes.
func (l *Layout) writeTo(ctx context.Context, w io.Writer, src io.ReaderAt, head []byte, onTensor func(Placed)) error {
	if int64(len(head)) != l.DataOffset {
		return fmt.Errorf("relayout: the header is %d bytes, the data starts at %d", len(head), l.DataOffset)
	}
	if _, err := w.Write(head); err != nil {
		return err
	}
	pos := l.DataOffset
	buf := make([]byte, copyChunk)
	for _, p := range l.Tensors {
		if err := writeZeros(w, p.Offset-pos); err != nil {
			return err
		}
		if onTensor != nil {
			onTensor(p)
		}
		for done := int64(0); done < p.Bytes; {
			if err := ctx.Err(); err != nil {
				return err
			}
			n := min(int64(len(buf)), p.Bytes-done)
			if got, err := src.ReadAt(buf[:n], p.From+done); int64(got) < n {
				return fmt.Errorf("read %s: %w", p.Name, cmpErr(err))
			}
			if _, err := w.Write(buf[:n]); err != nil {
				return err
			}
			done += n
		}
		pos = p.Offset + p.Bytes
	}
	return writeZeros(w, l.FileSize-pos)
}

var zeros [1 << 20]byte

func writeZeros(w io.Writer, n int64) error {
	for n > 0 {
		k := min(n, int64(len(zeros)))
		if _, err := w.Write(zeros[:k]); err != nil {
			return err
		}
		n -= k
	}
	return nil
}

// RelayoutOptions are the inputs of Relayout beside the two paths.
type RelayoutOptions struct {
	// Alignment is general.alignment of the result; zero is
	// DefaultRelayoutAlignment.
	Alignment int64
	// SourceSHA256 is the digest of the source when the caller knows it (the
	// catalog's pin, for a file the store verified); empty hashes the source.
	SourceSHA256 string
	// Tool is recorded in the sidecar ("nanollm v0.9").
	Tool string
	// Force replaces a result that is already there.
	Force bool
	// Progress is told how many bytes of the result are written.
	Progress func(done, total int64)
}

// Relayout writes the relaid copy of the model file src to dst and its sidecar
// beside it. The data is streamed in 4 MiB pieces through a temporary file and
// a rename, so dst is either absent or complete; the same source and options
// always give the same bytes.
func Relayout(ctx context.Context, src, dst string, o RelayoutOptions) (Sidecar, error) {
	var sc Sidecar
	f, err := Open(src)
	if err != nil {
		return sc, err
	}
	l, err := PlanFile(f, o.Alignment)
	if err != nil {
		return sc, err
	}
	sf, err := os.Open(src)
	if err != nil {
		return sc, err
	}
	defer sf.Close()
	if st, err := os.Stat(dst); err == nil {
		if sst, _ := sf.Stat(); sst != nil && os.SameFile(st, sst) {
			return sc, errors.New("relayout: the result is the source file")
		}
		if !o.Force {
			return sc, fmt.Errorf("%w: %s", ErrExists, dst)
		}
	}
	head, err := l.encodeHeader(f, sf)
	if err != nil {
		return sc, err
	}
	srcSum := o.SourceSHA256
	if srcSum == "" {
		if srcSum, err = hashReader(ctx, io.NewSectionReader(sf, 0, f.FileSize)); err != nil {
			return sc, err
		}
	}

	tmp := dst + ".part"
	_ = os.Remove(tmp)
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return sc, err
	}
	done := false
	defer func() {
		if !done {
			_ = out.Close()
			_ = os.Remove(tmp)
		}
	}()
	h := sha256.New()
	w := &countWriter{w: io.MultiWriter(out, h), total: l.FileSize, progress: o.Progress}
	if err := l.writeTo(ctx, w, sf, head, nil); err != nil {
		return sc, err
	}
	if w.n != l.FileSize {
		return sc, fmt.Errorf("relayout: wrote %d bytes, planned %d", w.n, l.FileSize)
	}
	if err := out.Sync(); err != nil {
		return sc, err
	}
	if err := out.Close(); err != nil {
		return sc, err
	}
	// A sidecar of the file being replaced must not outlive it.
	_ = os.Remove(SidecarPath(dst))
	if err := os.Rename(tmp, dst); err != nil {
		return sc, err
	}
	done = true
	st, err := os.Stat(dst)
	if err != nil {
		return sc, err
	}
	sc = Sidecar{
		Schema: SidecarSchema, Layout: LayoutNano, LayoutVersion: LayoutVersion,
		Alignment: l.Alignment, Tensors: len(l.Tensors),
		Source:   FileRef{File: filepath.Base(src), Size: f.FileSize, SHA256: srcSum},
		Result:   FileRef{File: filepath.Base(dst), Size: l.FileSize, SHA256: hex.EncodeToString(h.Sum(nil)), MTime: st.ModTime().UTC()},
		Sections: l.Sections, Tool: o.Tool,
	}
	if err := WriteSidecar(SidecarPath(dst), sc); err != nil {
		return sc, err
	}
	return sc, nil
}

type countWriter struct {
	w        io.Writer
	n        int64
	total    int64
	progress func(done, total int64)
	last     time.Time
}

func (c *countWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	if c.progress != nil && (c.n == c.total || time.Since(c.last) > 500*time.Millisecond) {
		c.last = time.Now()
		c.progress(c.n, c.total)
	}
	return n, err
}

func hashReader(ctx context.Context, r io.Reader) (string, error) {
	h := sha256.New()
	buf := make([]byte, copyChunk)
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, err := r.Read(buf)
		h.Write(buf[:n])
		if errors.Is(err, io.EOF) {
			return hex.EncodeToString(h.Sum(nil)), nil
		}
		if err != nil {
			return "", err
		}
	}
}

// VerifyReport is what a full verification covered.
type VerifyReport struct {
	Tensors  int
	Bytes    int64 // tensor bytes compared, without padding
	FileSize int64
	SHA256   string
}

// VerifyRelayout re-reads both files and checks that dst is exactly what
// Relayout writes for src with the alignment its sidecar names: it rebuilds
// the expected file piece by piece from src and compares it with dst, so every
// tensor's bytes, the metadata, the table and the padding are covered, and the
// digest of what it read must be the one in the sidecar. A mismatch names the
// tensor it was found in.
func VerifyRelayout(ctx context.Context, src, dst string, progress func(done, total int64)) (VerifyReport, error) {
	var rep VerifyReport
	sc, err := ReadSidecar(SidecarPath(dst))
	if err != nil {
		return rep, err
	}
	f, err := Open(src)
	if err != nil {
		return rep, err
	}
	if f.FileSize != sc.Source.Size {
		return rep, fmt.Errorf("%s has %d bytes, the sidecar was made from %d", filepath.Base(src), f.FileSize, sc.Source.Size)
	}
	l, err := PlanFile(f, sc.Alignment)
	if err != nil {
		return rep, err
	}
	sf, err := os.Open(src)
	if err != nil {
		return rep, err
	}
	defer sf.Close()
	df, err := os.Open(dst)
	if err != nil {
		return rep, err
	}
	defer df.Close()
	if st, err := df.Stat(); err != nil {
		return rep, err
	} else if st.Size() != l.FileSize {
		return rep, fmt.Errorf("%s has %d bytes, the layout of %s is %d", filepath.Base(dst), st.Size(), filepath.Base(src), l.FileSize)
	}
	head, err := l.encodeHeader(f, sf)
	if err != nil {
		return rep, err
	}
	cw := &compareWriter{r: df, h: sha256.New(), where: "the header", total: l.FileSize, progress: progress}
	err = l.writeTo(ctx, cw, sf, head, func(p Placed) {
		cw.where = p.Name
		rep.Tensors++
		rep.Bytes += p.Bytes
	})
	if err != nil {
		return rep, err
	}
	rep.FileSize = cw.pos
	rep.SHA256 = hex.EncodeToString(cw.h.Sum(nil))
	if rep.SHA256 != sc.Result.SHA256 {
		return rep, fmt.Errorf("the file hashes to %s, the sidecar says %s", rep.SHA256, sc.Result.SHA256)
	}
	// The file must also be one our reader takes as it is.
	g, err := Open(dst)
	if err != nil {
		return rep, err
	}
	if g.DataOffset != l.DataOffset || len(g.Tensors) != len(l.Tensors) || g.Alignment != l.Alignment {
		return rep, fmt.Errorf("the header reads back as %d tensors, data at %d, alignment %d; planned %d, %d, %d",
			len(g.Tensors), g.DataOffset, g.Alignment, len(l.Tensors), l.DataOffset, l.Alignment)
	}
	return rep, nil
}

// compareWriter takes the bytes of the expected file and checks them against
// the file it reads, in step.
type compareWriter struct {
	r        io.Reader
	h        hash.Hash
	buf      []byte
	pos      int64
	where    string
	total    int64
	progress func(done, total int64)
	last     time.Time
}

func (c *compareWriter) Write(p []byte) (int, error) {
	if cap(c.buf) < len(p) {
		c.buf = make([]byte, len(p))
	}
	got := c.buf[:len(p)]
	n, err := io.ReadFull(c.r, got)
	c.h.Write(got[:n])
	if err != nil {
		return 0, fmt.Errorf("the file ends at byte %d, in %s: %w", c.pos+int64(n), c.where, cmpErr(err))
	}
	if !bytes.Equal(got, p) {
		i := 0
		for got[i] == p[i] {
			i++
		}
		return 0, fmt.Errorf("the file differs at byte %d, in %s", c.pos+int64(i), c.where)
	}
	c.pos += int64(n)
	if c.progress != nil && (c.pos == c.total || time.Since(c.last) > 500*time.Millisecond) {
		c.last = time.Now()
		c.progress(c.pos, c.total)
	}
	return n, nil
}
