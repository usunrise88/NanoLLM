// fa_small.dylib: makes llama-perplexity of the darwin_arm64 archive run with LLAMA_FLASH_ATTN_TYPE_SMALL_BATCH
// (patch 0007), which its libllama knows and its libllama-common cannot parse from "-fa small_batch".
//
//   clang -dynamiclib -o fa_small.dylib fa_small.c -Ldist/darwin_arm64/tools -lllama
//   DYLD_INSERT_LIBRARIES=$PWD/fa_small.dylib llama-perplexity ... -fa on
//
// It interposes llama_init_from_model and sets flash_attn_type (offset 52 of llama_context_params in b11600, as
// internal/backend/llamacpp/zz_generated.go lays it out) to 2. arm64 passes a struct larger than 16 bytes as a
// pointer to the caller's copy, so the pointer is edited in place and passed on unchanged.
#include <stdint.h>
#include <stdio.h>

struct llama_model;
struct llama_context;
struct llama_context *llama_init_from_model(struct llama_model *model, uint8_t *params);

static struct llama_context *fa_small_init(struct llama_model *model, uint8_t *params) {
    int32_t *fa = (int32_t *)(params + 52);
    fprintf(stderr, "fa_small: flash_attn_type %d -> 2 (small_batch)\n", *fa);
    *fa = 2;
    return llama_init_from_model(model, params);
}

__attribute__((used, section("__DATA,__interpose"))) static const struct {
    const void *replacement;
    const void *replacee;
} interposers[] = {
    {(const void *)fa_small_init, (const void *)llama_init_from_model},
};
