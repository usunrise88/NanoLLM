package gguf

import "fmt"

// TinyGemma4 is a gemma4 header with every tensor kind E4B has, at toy sizes:
// six layers in the E4B pattern (five sliding-window, one full), the last two
// sharing KV, per-layer embeddings and a tied LM head. Tests of the inventory,
// `models inspect` and the planner use it where the real file is not at hand.
func TinyGemma4() *Writer {
	const (
		layers = 6
		embd   = 64
		ff     = 128
		ple    = 32
		vocab  = 256
		headsQ = 2
		heads  = 1
		hdFull = 64
		hdSWA  = 32
	)
	pattern := []bool{true, true, true, true, true, false}
	w := NewWriter().
		Set("general.architecture", ArchGemma4).
		Set("general.name", "tiny gemma4").
		Set("gemma4.block_count", uint32(layers)).
		Set("gemma4.context_length", uint32(8192)).
		Set("gemma4.embedding_length", uint32(embd)).
		Set("gemma4.feed_forward_length", uint32(ff)).
		Set("gemma4.attention.head_count", uint32(headsQ)).
		Set("gemma4.attention.head_count_kv", uint32(heads)).
		Set("gemma4.attention.key_length", uint32(hdFull)).
		Set("gemma4.attention.value_length", uint32(hdFull)).
		Set("gemma4.attention.key_length_swa", uint32(hdSWA)).
		Set("gemma4.attention.value_length_swa", uint32(hdSWA)).
		Set("gemma4.attention.sliding_window", uint32(128)).
		Set("gemma4.attention.sliding_window_pattern", pattern).
		Set("gemma4.attention.shared_kv_layers", uint32(2)).
		Set("gemma4.embedding_length_per_layer_input", uint32(ple)).
		Set("tokenizer.ggml.model", "gemma4").
		Set("tokenizer.ggml.tokens", make([]string, vocab))
	w.Tensor("output_norm.weight", TypeF32, []int64{embd}, nil).
		Tensor("per_layer_model_proj.weight", TypeQ4_0, []int64{embd, ple * layers}, nil).
		Tensor("per_layer_proj_norm.weight", TypeF32, []int64{ple}, nil).
		Tensor("per_layer_token_embd.weight", TypeQ4_0, []int64{ple * layers, vocab}, nil).
		Tensor("rope_freqs.weight", TypeF32, []int64{hdFull / 2}, nil).
		Tensor("token_embd.weight", TypeQ4_0, []int64{embd, vocab}, nil)
	for il := range layers {
		hd := hdSWA
		if !pattern[il] {
			hd = hdFull
		}
		t := func(name string, typ Type, shape ...int64) {
			w.Tensor(fmt.Sprintf("blk.%d.%s.weight", il, name), typ, shape, nil)
		}
		t("attn_norm", TypeF32, embd)
		t("attn_q", TypeQ4_0, embd, int64(hd*headsQ))
		t("attn_q_norm", TypeF32, int64(hd))
		if il < layers-2 { // shared-KV layers have no K/V projections
			t("attn_k", TypeQ4_0, embd, int64(hd*heads))
			t("attn_v", TypeQ4_0, embd, int64(hd*heads))
			t("attn_k_norm", TypeF32, int64(hd))
		}
		t("attn_output", TypeQ4_0, int64(hd*headsQ), embd)
		t("post_attention_norm", TypeF32, embd)
		t("ffn_norm", TypeF32, embd)
		t("ffn_gate", TypeQ4_0, embd, ff)
		t("ffn_up", TypeQ4_0, embd, ff)
		t("ffn_down", TypeQ4_0, ff, embd)
		t("post_ffw_norm", TypeF32, embd)
		t("inp_gate", TypeQ4_0, embd, ple)
		t("proj", TypeQ4_0, ple, embd)
		t("post_norm", TypeF32, embd)
		t("layer_output_scale", TypeF32, 1)
	}
	return w
}
