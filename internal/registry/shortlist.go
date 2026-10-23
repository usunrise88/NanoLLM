package registry

import (
	"bytes"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"fmt"
)

// Shortlists of the output head (OPT-4, docs/AUTOTUNE.md): the vocabulary ids
// whose logits the head computes while no request needs the whole vocabulary.
// A list is a bitmap built by tools/shortlist.py from a corpus of 11.5 M
// tokens of Russian and English text, source code and documents, and bound to
// the sha256 of the model file it was made for (the vocabulary is the file's).
//
//	"NLSL" | version u32 | vocabulary size u32 | rows u32 | model sha256 (32 B) | bitmap, bit i = id i, low bit first
//
// The rows are a multiple of 8 (the repacked kernels take rows in groups of 8).
//
//go:embed shortlist/*.bin
var shortlistFS embed.FS

// Shortlist kinds a model override can name (models.<id>.head).
const (
	// HeadFull computes the whole head; the default.
	HeadFull = "full"
	// HeadShortlist is the list that keeps the tokens the corpus used and the
	// whole Latin, Cyrillic and common-symbol part of the vocabulary: 65 % of
	// the rows, a loss of 0.002 nat on ru/en text.
	HeadShortlist = "shortlist"
	// HeadShortlistSmall is the tokens the corpus used and nothing else: 37 %
	// of the rows, 0.0085 nat.
	HeadShortlistSmall = "shortlist-small"
)

// Shortlist is a parsed list.
type Shortlist struct {
	// IDs are the vocabulary ids, ascending.
	IDs []int32
	// Vocab is the size of the vocabulary the list was made for.
	Vocab int
	// ModelSHA256 is the file the list belongs to.
	ModelSHA256 string
}

// ShortlistIDs is the list of kind for a catalog entry, checked against the
// sha256 of its model file.
func ShortlistIDs(m Manifest, kind string) ([]int32, error) {
	f, ok := m.File(RoleModel)
	if !ok {
		return nil, fmt.Errorf("registry: %s has no model file", m.ID)
	}
	sl, err := LoadShortlist(m.ID, kind, f.SHA256)
	if err != nil {
		return nil, err
	}
	return sl.IDs, nil
}

// HeadRowBytes is what the head's copy for rows rows costs in memory: a Q4_0
// row of n_embd columns is n_embd/32 blocks of 18 bytes.
func HeadRowBytes(rows, nEmbd int) int64 { return int64(rows) * int64(nEmbd/32) * 18 }

// LoadShortlist returns the list of kind (HeadShortlist, HeadShortlistSmall)
// for a model, or an error saying why there is none: no list for the id, or one
// made for another file.
func LoadShortlist(modelID, kind, modelSHA256 string) (*Shortlist, error) {
	name := "shortlist/" + modelID + ".bin"
	switch kind {
	case HeadShortlist:
	case HeadShortlistSmall:
		name = "shortlist/" + modelID + ".small.bin"
	default:
		return nil, fmt.Errorf("registry: no output shortlist of kind %q", kind)
	}
	raw, err := shortlistFS.ReadFile(name)
	if err != nil {
		return nil, fmt.Errorf("registry: model %s has no output shortlist", modelID)
	}
	sl, err := parseShortlist(raw)
	if err != nil {
		return nil, fmt.Errorf("registry: %s: %w", name, err)
	}
	if modelSHA256 != "" && sl.ModelSHA256 != modelSHA256 {
		return nil, fmt.Errorf("registry: the output shortlist of %s was made for the file %s…, not %s…", modelID, sl.ModelSHA256[:12], modelSHA256[:min(12, len(modelSHA256))])
	}
	return sl, nil
}

func parseShortlist(raw []byte) (*Shortlist, error) {
	const header = 4 + 4*3 + 32
	if len(raw) < header || !bytes.Equal(raw[:4], []byte("NLSL")) {
		return nil, fmt.Errorf("not a shortlist")
	}
	if v := binary.LittleEndian.Uint32(raw[4:]); v != 1 {
		return nil, fmt.Errorf("shortlist version %d", v)
	}
	vocab, rows := int(binary.LittleEndian.Uint32(raw[8:])), int(binary.LittleEndian.Uint32(raw[12:]))
	sl := &Shortlist{Vocab: vocab, ModelSHA256: hex.EncodeToString(raw[16:48])}
	bits := raw[header:]
	if len(bits) != (vocab+7)/8 {
		return nil, fmt.Errorf("the bitmap has %d bytes for a vocabulary of %d", len(bits), vocab)
	}
	for id := 0; id < vocab; id++ {
		if bits[id>>3]>>(id&7)&1 != 0 {
			sl.IDs = append(sl.IDs, int32(id))
		}
	}
	if len(sl.IDs) != rows || rows%8 != 0 {
		return nil, fmt.Errorf("%d ids in the bitmap, %d in the header (a multiple of 8 is needed)", len(sl.IDs), rows)
	}
	return sl, nil
}
