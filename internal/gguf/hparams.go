package gguf

import "fmt"

// HParams are the hyperparameters that size memory: the KV cache (G5) and the
// compute buffers (G7). Names follow the GGUF keys without the architecture
// prefix, so `gemma4.attention.key_length_swa` is KeyLengthSWA.
//
// Per-layer values are expanded to one entry per layer even when the file
// stores a scalar, the way llama.cpp's get_key_or_arr reads them.
type HParams struct {
	Arch                         string `json:"arch"`
	BlockCount                   int    `json:"block_count"`
	ContextLength                int    `json:"context_length"`
	EmbeddingLength              int    `json:"embedding_length"`
	EmbeddingLengthOut           int    `json:"embedding_length_out,omitempty"`
	EmbeddingLengthPerLayerInput int    `json:"embedding_length_per_layer_input"`
	FeedForwardLength            []int  `json:"feed_forward_length"`
	HeadCount                    []int  `json:"head_count"`
	HeadCountKV                  []int  `json:"head_count_kv"`
	KeyLength                    int    `json:"key_length"`
	ValueLength                  int    `json:"value_length"`
	KeyLengthSWA                 int    `json:"key_length_swa"`
	ValueLengthSWA               int    `json:"value_length_swa"`
	SlidingWindow                int    `json:"sliding_window"`
	SlidingWindowPattern         []bool `json:"sliding_window_pattern"`
	SharedKVLayers               int    `json:"shared_kv_layers"`
	NextNPredictLayers           int    `json:"nextn_predict_layers,omitempty"`
	// VocabSize is the length of tokenizer.ggml.tokens.
	VocabSize int `json:"vocab_size"`
	// TiedOutput is set when the file has no output.weight and the LM head
	// reuses token_embd.
	TiedOutput bool `json:"tied_output"`
}

// KVLayers is how many layers hold their own KV cache. The last
// SharedKVLayers layers reuse the cache of an earlier layer of the same kind
// (llama.cpp n_layer_kv_from_start).
func (h HParams) KVLayers() int { return max(h.BlockCount-h.SharedKVLayers, 0) }

// IsSWA reports whether layer il uses sliding-window attention.
func (h HParams) IsSWA(il int) bool {
	return il < len(h.SlidingWindowPattern) && h.SlidingWindowPattern[il]
}

// HeadDimK and HeadDimV are the per-head sizes of layer il: SWA and full
// layers differ in gemma4.
func (h HParams) HeadDimK(il int) int {
	if h.IsSWA(il) {
		return h.KeyLengthSWA
	}
	return h.KeyLength
}

func (h HParams) HeadDimV(il int) int {
	if h.IsSWA(il) {
		return h.ValueLengthSWA
	}
	return h.ValueLength
}

// EmbdKGQA and EmbdVGQA are the K and V row widths of layer il
// (n_embd_k_gqa, n_embd_v_gqa).
func (h HParams) EmbdKGQA(il int) int { return h.HeadDimK(il) * h.at(h.HeadCountKV, il) }
func (h HParams) EmbdVGQA(il int) int { return h.HeadDimV(il) * h.at(h.HeadCountKV, il) }

// EmbdVGQAMax is the widest V row over all layers. Without flash attention V
// is stored transposed, and llama.cpp pads every layer's V to this width.
func (h HParams) EmbdVGQAMax() int {
	m := 0
	for il := range h.BlockCount {
		m = max(m, h.EmbdVGQA(il))
	}
	return m
}

// FF is the FFN width of layer il.
func (h HParams) FF(il int) int { return h.at(h.FeedForwardLength, il) }

// Heads is the query head count of layer il.
func (h HParams) Heads(il int) int { return h.at(h.HeadCount, il) }

func (h HParams) at(xs []int, il int) int {
	if il < len(xs) {
		return xs[il]
	}
	if len(xs) > 0 {
		return xs[len(xs)-1]
	}
	return 0
}

// HParams reads the hyperparameters of the file's architecture.
func (f *File) HParams() (HParams, error) {
	arch, ok := f.String("general.architecture")
	if !ok || arch == "" {
		return HParams{}, fmt.Errorf("%w: no general.architecture", ErrFormat)
	}
	h := HParams{Arch: arch}
	key := func(k string) string { return arch + "." + k }
	need := func(k string) (int, error) {
		n, ok := f.Int(key(k))
		if !ok {
			return 0, fmt.Errorf("%w: missing or non-integer %s", ErrFormat, key(k))
		}
		return int(n), nil
	}
	opt := func(k string, def int) int {
		if n, ok := f.Int(key(k)); ok {
			return int(n)
		}
		return def
	}
	var err error
	if h.BlockCount, err = need("block_count"); err != nil {
		return h, err
	}
	if h.EmbeddingLength, err = need("embedding_length"); err != nil {
		return h, err
	}
	if h.BlockCount <= 0 || h.BlockCount > 4096 || h.EmbeddingLength <= 0 {
		return h, fmt.Errorf("%w: block_count %d, embedding_length %d", ErrFormat, h.BlockCount, h.EmbeddingLength)
	}
	h.ContextLength = opt("context_length", 0)
	h.EmbeddingLengthOut = opt("embedding_length_out", 0)
	h.EmbeddingLengthPerLayerInput = opt("embedding_length_per_layer_input", 0)
	h.SlidingWindow = opt("attention.sliding_window", 0)
	h.SharedKVLayers = opt("attention.shared_kv_layers", 0)
	h.NextNPredictLayers = opt("nextn_predict_layers", 0)
	if h.SharedKVLayers < 0 || h.SharedKVLayers > h.BlockCount {
		return h, fmt.Errorf("%w: shared_kv_layers %d of %d layers", ErrFormat, h.SharedKVLayers, h.BlockCount)
	}

	perLayer := func(k string, def int) ([]int, error) {
		xs, ok := f.Ints(key(k))
		if !ok {
			if def < 0 {
				return nil, fmt.Errorf("%w: missing %s", ErrFormat, key(k))
			}
			xs = []int64{int64(def)}
		}
		if len(xs) != 1 && len(xs) != h.BlockCount {
			return nil, fmt.Errorf("%w: %s has %d entries for %d layers", ErrFormat, key(k), len(xs), h.BlockCount)
		}
		out := make([]int, h.BlockCount)
		for i := range out {
			out[i] = int(xs[min(i, len(xs)-1)])
		}
		return out, nil
	}
	if h.HeadCount, err = perLayer("attention.head_count", -1); err != nil {
		return h, err
	}
	if h.HeadCountKV, err = perLayer("attention.head_count_kv", h.HeadCount[0]); err != nil {
		return h, err
	}
	if h.FeedForwardLength, err = perLayer("feed_forward_length", 0); err != nil {
		return h, err
	}
	if h.HeadCount[0] <= 0 {
		return h, fmt.Errorf("%w: head_count %d", ErrFormat, h.HeadCount[0])
	}
	// llama.cpp's defaults: head size from the embedding, SWA sizes from the
	// full ones.
	h.KeyLength = opt("attention.key_length", h.EmbeddingLength/h.HeadCount[0])
	h.ValueLength = opt("attention.value_length", h.KeyLength)
	h.KeyLengthSWA = opt("attention.key_length_swa", h.KeyLength)
	h.ValueLengthSWA = opt("attention.value_length_swa", h.ValueLength)

	if pat, ok := f.Bools(key("attention.sliding_window_pattern")); ok {
		if len(pat) != h.BlockCount {
			return h, fmt.Errorf("%w: sliding_window_pattern has %d entries for %d layers",
				ErrFormat, len(pat), h.BlockCount)
		}
		h.SlidingWindowPattern = pat
	} else {
		h.SlidingWindowPattern = make([]bool, h.BlockCount)
	}

	if n, ok := f.ArrayLen("tokenizer.ggml.tokens"); ok {
		h.VocabSize = int(n)
	} else if t, ok := f.Tensor("token_embd.weight"); ok && len(t.Shape) == 2 {
		h.VocabSize = int(t.Shape[1])
	}
	_, hasOutput := f.Tensor("output.weight")
	h.TiedOutput = !hasOutput
	return h, nil
}
