package llamacpp

// The two functions of the output shortlist (patches/next/0009, head.go). They
// are looked up like the staging functions of zz_generated.go: optionally, so a
// library without them still loads and the shortlist reports itself off.
//
// They are bound here by hand and not by tools/bindgen because the patch is not
// in the released series yet: the generated binding is checked against the
// series in CI. When the patch joins the series, they go into allowCxx of
// tools/bindgen/allowlist.go, the binding is regenerated, and this file goes.
var (
	llamaModelSetOutputShortlist func(model LlamaModel, ids *int32, nIds int32) bool
	llamaSetOutputShortlist      func(ctx LlamaContext, value bool)
)

func init() {
	cxxSymbols = append(cxxSymbols,
		cxxSymbol{"llama_model_set_output_shortlist", "_Z32llama_model_set_output_shortlistP11llama_modelPKii", "?llama_model_set_output_shortlist@@YA_NPEAUllama_model@@PEBHH@Z", &llamaModelSetOutputShortlist},
		cxxSymbol{"llama_set_output_shortlist", "_Z26llama_set_output_shortlistP13llama_contextb", "?llama_set_output_shortlist@@YAXPEAUllama_context@@_N@Z", &llamaSetOutputShortlist},
	)
}
