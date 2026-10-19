package promptcache

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"

	"github.com/usunrise88/nanollm/internal/backend"
)

// A stored state is one file:
//
//	off  size
//	  0     4  magic "NLPC"
//	  4     4  version (1)
//	  8    32  fingerprint of the model and the context shape it was saved under
//	 40     4  number of tokens n
//	 44    32  sha256 of the token ids (little-endian int32, back to back)
//	 76     8  length of the state in bytes
//	 84     4  CRC-32C of the state
//	 88   4·n  the token ids
//	      ...  the state: whatever the backend's SaveSeq returned
//
// The size of the file is therefore known from its header, which is how a
// truncated file is told from a whole one without reading it. The state is
// checked against its CRC only when it is loaded, so that scanning a
// directory of gigabytes reads kilobytes.
const (
	magic      = "NLPC"
	version    = 1
	headerSize = 88
	// Ext is the extension of a stored state.
	Ext = ".nlpc"
	// tmpPrefix and tmpSuffix name a file while it is being written.
	tmpPrefix = ".pc-"
	tmpSuffix = ".tmp"
)

var crcTable = crc32.MakeTable(crc32.Castagnoli)

// Fingerprint identifies everything a saved state depends on. Two stores with
// different fingerprints never read each other's states.
type Fingerprint [32]byte

// NewFingerprint hashes the given fields, each as a length-prefixed string so
// that no two lists of fields give the same bytes. Callers put "name=value"
// strings in: the model file's sha256, the KV cache types, the engine build.
func NewFingerprint(fields ...string) Fingerprint {
	h := sha256.New()
	var n [8]byte
	for _, f := range fields {
		binary.LittleEndian.PutUint64(n[:], uint64(len(f)))
		h.Write(n[:])
		h.Write([]byte(f))
	}
	var fp Fingerprint
	h.Sum(fp[:0])
	return fp
}

// String is the fingerprint in hex.
func (f Fingerprint) String() string { return fmt.Sprintf("%x", f[:]) }

type header struct {
	fp       Fingerprint
	n        int
	tokens   [32]byte
	stateLen uint64
	crc      uint32
}

func (h header) size() int64 { return headerSize + 4*int64(h.n) + int64(h.stateLen) }

func (h header) marshal() []byte {
	b := make([]byte, headerSize)
	copy(b, magic)
	binary.LittleEndian.PutUint32(b[4:], version)
	copy(b[8:], h.fp[:])
	binary.LittleEndian.PutUint32(b[40:], uint32(h.n))
	copy(b[44:], h.tokens[:])
	binary.LittleEndian.PutUint64(b[76:], h.stateLen)
	binary.LittleEndian.PutUint32(b[84:], h.crc)
	return b
}

var (
	errNotOurs = errors.New("not a prompt cache file")
	errVersion = errors.New("another version of the format")
)

func parseHeader(b []byte) (header, error) {
	var h header
	if len(b) < headerSize || string(b[:4]) != magic {
		return h, errNotOurs
	}
	if v := binary.LittleEndian.Uint32(b[4:]); v != version {
		return h, errVersion
	}
	copy(h.fp[:], b[8:40])
	h.n = int(binary.LittleEndian.Uint32(b[40:]))
	copy(h.tokens[:], b[44:76])
	h.stateLen = binary.LittleEndian.Uint64(b[76:])
	h.crc = binary.LittleEndian.Uint32(b[84:])
	return h, nil
}

// tokenBytes writes tokens as little-endian int32 into w.
func tokenBytes(w io.Writer, tokens []backend.Token) error {
	var buf [4096]byte
	for len(tokens) > 0 {
		n := min(len(tokens), len(buf)/4)
		for i, t := range tokens[:n] {
			binary.LittleEndian.PutUint32(buf[4*i:], uint32(t))
		}
		if _, err := w.Write(buf[:4*n]); err != nil {
			return err
		}
		tokens = tokens[n:]
	}
	return nil
}

// tokensSum is the sha256 of the token ids.
func tokensSum(tokens []backend.Token) [32]byte {
	h := sha256.New()
	_ = tokenBytes(h, tokens)
	var sum [32]byte
	h.Sum(sum[:0])
	return sum
}
