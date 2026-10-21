#!/usr/bin/env bash
# Writes the .inc files the harnesses include into $GEN (default .cache/kernel-bench): the old kernels verbatim from
# the pinned upstream file and the kernels of the patched tree in LLAMA_SRC. Sourced by run.sh; run-windows.ps1 runs it.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
SRC=${LLAMA_SRC:-$ROOT/.cache/llama.cpp/src}
GEN=${GEN:-$ROOT/.cache/kernel-bench}
BASE=$(tr -d '[:space:]' < scripts/llama.cpp.base)
FILE=ggml/src/ggml-cpu/arch/x86/repack.cpp

mkdir -p "$GEN"
git -C "$SRC" show "$BASE:$FILE" > "$GEN/orig.cpp"
# the old kernels, verbatim from the upstream file
awk '/^\/\/ add int16_t pairwise and return as 256 bit int vector/{p=1} p{print} p&&/^static inline __m256i mul_sum_i8_pairs_acc_int32x8/{f=1} f&&/^}/{exit}' "$GEN/orig.cpp" > "$GEN/old_helpers.inc"
sed -n '/^\/\/ GEMV for 8x blocks of 32 4-bit quants with a single scale factor per block/,/^\/\/ GEMM for 8x blocks of 32 4-bit quants/p' "$GEN/orig.cpp" | sed '$d' > "$GEN/old_kernel.inc"
sed -n '/^\/\/ GEMM for 8x blocks of 32 4-bit quants with a single scale factor per block/,/^#endif \/\/ defined(__AVX2__) || defined(__AVX512F__)/p' "$GEN/orig.cpp" | sed '$d' > "$GEN/old_gemm.inc"
# the kernels of the patched tree, in a namespace
{ echo 'namespace real {'; sed -n -E '/^#if defined\(GGML_REPACK_Q4_0_GEMV_U8\)/,/^#endif \/\/ AVX2 (without VNNI|Q4_0 u8 kernels)/{p;/^#endif \/\/ AVX2 (without VNNI|Q4_0 u8 kernels)/q}' "$SRC/$FILE"; echo '}'; } > "$GEN/real.inc"
