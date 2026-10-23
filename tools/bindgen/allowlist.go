package main

// What NanoLLM binds. A name missing from the headers fails the generator, so
// an upstream rename shows up when the tag is bumped, not at run time.

// headers are read in this order; the C++ staging header last.
var headers = []struct {
	path string
	cxx  bool
}{
	{"ggml/include/ggml.h", false},
	{"ggml/include/ggml-backend.h", false},
	{"include/llama.h", false},
	{"src/llama-ext.h", true},
}

// structs bound by value or read field by field. Each is laid out with
// explicit padding and checked against the C compiler by the layout test.
var allowStructs = []string{
	"llama_model_params",
	"llama_context_params",
	"llama_sampler_chain_params",
	"llama_model_tensor_buft_override",
	"llama_token_data",
	"llama_token_data_array",
	"llama_logit_bias",
	"llama_embd",              // MTP state rows (1.5.M)
	"llama_perf_context_data", // timings for run and bench
	"ggml_backend_dev_caps",
	"ggml_backend_dev_props",
}

var allowEnums = []string{
	"ggml_type",
	"ggml_log_level",
	"ggml_backend_dev_type",
	"llama_vocab_type",
	"llama_rope_scaling_type",
	"llama_pooling_type",
	"llama_attention_type",
	"llama_flash_attn_type",
	"llama_split_mode",
	"llama_load_mode",
	"llama_lazy_mode",
	"llama_context_type",
	"llama_process_type",
}

// allowFuncs are the extern "C" functions, from libggml-base, libggml or
// libllama; the loader looks each up in all three.
var allowFuncs = []string{
	// build identity and logging
	"ggml_commit",
	"llama_print_system_info",
	"llama_log_set",

	// backends and devices
	"ggml_backend_load_all_from_path",
	"ggml_backend_reg_count",
	"ggml_backend_reg_get",
	"ggml_backend_reg_name",
	"ggml_backend_dev_count",
	"ggml_backend_dev_get",
	"ggml_backend_dev_name",
	"ggml_backend_dev_description",
	"ggml_backend_dev_memory",
	"ggml_backend_dev_type",
	"ggml_backend_dev_get_props",
	"ggml_backend_dev_backend_reg",
	"ggml_backend_reg_get_proc_address", // the CPU backend's extra buffer types (repack) are only reachable through it
	"ggml_backend_dev_buffer_type",
	"ggml_backend_cpu_buffer_type",
	"ggml_backend_buft_name",
	"ggml_type_name",

	// lifecycle
	"llama_backend_init",
	"llama_backend_free",
	"llama_model_default_params",
	"llama_context_default_params",
	"llama_sampler_chain_default_params",
	"llama_model_load_from_file",
	"llama_model_free",
	"llama_init_from_model",
	"llama_free",
	"llama_max_devices",
	"llama_max_parallel_sequences",
	"llama_max_tensor_buft_overrides",
	"llama_supports_mmap",
	"llama_supports_mlock",
	"llama_supports_gpu_offload",

	// context and model shape
	"llama_n_ctx",
	"llama_n_ctx_seq",
	"llama_n_batch",
	"llama_n_ubatch",
	"llama_n_seq_max",
	"llama_get_model",
	"llama_get_memory",
	"llama_model_get_vocab",
	"llama_model_n_ctx_train",
	"llama_model_n_embd",
	"llama_model_n_embd_out",
	"llama_model_n_layer",
	"llama_model_n_layer_nextn",
	"llama_model_n_head",
	"llama_model_n_head_kv",
	"llama_model_n_swa",
	"llama_model_has_encoder",
	"llama_model_is_recurrent",
	"llama_model_is_hybrid",
	"llama_model_size",
	"llama_model_n_params",
	"llama_model_desc",
	"llama_model_chat_template",
	"llama_model_meta_count",
	"llama_model_meta_key_by_index",
	"llama_model_meta_val_str_by_index",
	"llama_model_meta_val_str",

	// vocabulary and tokenizer
	"llama_vocab_n_tokens",
	"llama_vocab_type",
	"llama_vocab_is_eog",
	"llama_vocab_is_control",
	"llama_vocab_bos",
	"llama_vocab_eos",
	"llama_vocab_eot",
	"llama_vocab_get_add_bos",
	"llama_vocab_get_suppress_tokens",
	"llama_tokenize",
	"llama_token_to_piece",
	"llama_detokenize",

	// memory (KV cache) and sequence state
	"llama_memory_clear",
	"llama_memory_seq_rm",
	"llama_memory_seq_cp",
	"llama_memory_seq_pos_min",
	"llama_memory_seq_pos_max",
	"llama_state_seq_get_size_ext",
	"llama_state_seq_get_data_ext",
	"llama_state_seq_set_data_ext",

	// batches and evaluation
	"llama_batch_ext_init",
	"llama_batch_ext_free",
	"llama_batch_ext_clear",
	"llama_batch_ext_add_token",
	"llama_batch_ext_add_seq",
	"llama_batch_ext_set_pos",
	"llama_batch_ext_set_output_logits",
	"llama_batch_ext_set_output_embd",
	"llama_batch_ext_set_embd_state", // MTP (1.5.M)
	"llama_process",
	"llama_set_n_threads",
	"llama_synchronize",
	"llama_get_logits_ith",
	"llama_get_embeddings_ith",
	"llama_perf_context",
	"llama_perf_context_reset",

	// sampling
	"llama_sampler_chain_init",
	"llama_sampler_chain_add",
	"llama_sampler_free",
	"llama_sampler_accept",
	"llama_sampler_apply",
	"llama_sampler_reset",
	"llama_sampler_sample",
	"llama_sampler_init_greedy",
	"llama_sampler_init_dist",
	"llama_sampler_init_top_k",
	"llama_sampler_init_top_p",
	"llama_sampler_init_min_p",
	"llama_sampler_init_temp",
	"llama_sampler_init_penalties",
	"llama_sampler_init_logit_bias",
	"llama_sampler_init_grammar",
	"llama_sampler_init_grammar_lazy_patterns",

	// backend sampling: the chain in the device graph, only its result copied
	"llama_set_sampler",
	"llama_get_sampled_token_ith",
	"llama_get_sampled_logits_ith",
	"llama_get_sampled_logits_count_ith",
	"llama_get_sampled_candidates_ith",
	"llama_get_sampled_candidates_count_ith",
}

// allowCxx are functions of the staging header, exported with C++ decorated
// names. They are looked up optionally: a library without them still loads,
// and MTP reports itself unavailable.
var allowCxx = []string{
	"llama_set_embeddings_nextn",
	"llama_get_embeddings_nextn",
	"llama_get_embeddings_nextn_ith",
	"llama_set_nextn_layer_offset",
	"llama_get_ctx_other",
	"llama_model_n_devices",
	"llama_model_get_device",
	"llama_model_set_output_shortlist",
	"llama_set_output_shortlist",
	"llama_model_get_tensor",
}
