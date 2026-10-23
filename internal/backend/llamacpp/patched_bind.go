package llamacpp

import "unsafe"

// The functions of the patches that are not in the released series yet:
// llama_model_set_output_shortlist and llama_set_output_shortlist (patches/next/0009,
// head.go) and llama_model_get_tensor (patches/next/0010, direct.go). They are looked
// up like the staging functions of zz_generated.go: optionally, so a library without
// them still loads and the feature reports itself off.
//
// They are bound here by hand and not by tools/bindgen because the patch is not
// in the released series yet: the generated binding is checked against the
// series in CI. When the patch joins the series, they go into allowCxx of
// tools/bindgen/allowlist.go, the binding is regenerated, and this file goes.
var (
	llamaModelSetOutputShortlist func(model LlamaModel, ids *int32, nIds int32) bool
	llamaSetOutputShortlist      func(ctx LlamaContext, value bool)
	llamaModelGetTensor          func(model LlamaModel, name string) unsafe.Pointer
)

func init() {
	cxxSymbols = append(cxxSymbols,
		cxxSymbol{"llama_model_set_output_shortlist", "_Z32llama_model_set_output_shortlistP11llama_modelPKii", "?llama_model_set_output_shortlist@@YA_NPEAUllama_model@@PEBHH@Z", &llamaModelSetOutputShortlist},
		cxxSymbol{"llama_set_output_shortlist", "_Z26llama_set_output_shortlistP13llama_contextb", "?llama_set_output_shortlist@@YAXPEAUllama_context@@_N@Z", &llamaSetOutputShortlist},
		cxxSymbol{"llama_model_get_tensor", "_Z21llama_model_get_tensorPK11llama_modelPKc", "?llama_model_get_tensor@@YAPEAUggml_tensor@@PEBUllama_model@@PEBD@Z", &llamaModelGetTensor},
	)
}
