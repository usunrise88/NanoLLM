#!/usr/bin/env bash
# Micro-benchmarks and bit-exactness checks of the Q4_0 8x8 AVX2 kernels (OPT-16, docs/AUTOTUNE.md).
#
#   scripts/kernel-bench/run.sh gemv [CPU]     # GEMV (decode): old kernel against the candidates, one thread
#   scripts/kernel-bench/run.sh gemm [CPU]     # GEMM (prefill)
#   scripts/kernel-bench/run.sh multi [CPU]    # GEMV with 1 to 7 rows of the activation in one call
#   scripts/kernel-bench/run.sh check          # only the bit-exactness checks (random and extreme data)
#   scripts/kernel-bench/run.sh uops [NAME...] # cycles and FP-pipe uops of single instructions (needs perf)
#
# The "old" kernels are the ones of the pinned upstream commit (scripts/llama.cpp.base); "real" is the code of
# the patched tree in LLAMA_SRC (default .cache/llama.cpp/src, the clone build-libs.sh uses); the other variants
# live in new_kernels.inc / new_gemm.inc and are listed in variants*.inc (a name ending in * is timing-only: it
# is not bit-exact on purpose). Everything is compiled in the container of build-libs.sh with the flags of
# ggml's haswell variant. Timing on a shared host is noise: use scripts/hyper-exclusive.sh for numbers that
# are going to be written down, and CPU to pin the run.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
HERE=scripts/kernel-bench
SRC=${LLAMA_SRC:-$ROOT/.cache/llama.cpp/src}
GEN=$ROOT/.cache/kernel-bench
BASE=$(tr -d '[:space:]' < scripts/llama.cpp.base)
IMAGE=nanollm-build-libs:ubuntu22.04
FILE=ggml/src/ggml-cpu/arch/x86/repack.cpp

mkdir -p "$GEN"
git -C "$SRC" show "$BASE:$FILE" > "$GEN/orig.cpp"
# the old kernels, verbatim from the upstream file
awk '/^\/\/ add int16_t pairwise and return as 256 bit int vector/{p=1} p{print} p&&/^static inline __m256i mul_sum_i8_pairs_acc_int32x8/{f=1} f&&/^}/{exit}' "$GEN/orig.cpp" > "$GEN/old_helpers.inc"
sed -n '/^\/\/ GEMV for 8x blocks of 32 4-bit quants with a single scale factor per block/,/^\/\/ GEMM for 8x blocks of 32 4-bit quants/p' "$GEN/orig.cpp" | sed '$d' > "$GEN/old_kernel.inc"
sed -n '/^\/\/ GEMM for 8x blocks of 32 4-bit quants with a single scale factor per block/,/^#endif \/\/ defined(__AVX2__) || defined(__AVX512F__)/p' "$GEN/orig.cpp" | sed '$d' > "$GEN/old_gemm.inc"
# the kernels of the patched tree, in a namespace
{ echo 'namespace real {'; sed -n '/^#if defined(GGML_REPACK_Q4_0_GEMV_U8)/,/^#endif \/\/ AVX2 without VNNI/{p;/^#endif \/\/ AVX2 without VNNI/q}' "$SRC/$FILE"; echo '}'; } > "$GEN/real.inc"

build() { # build SOURCE OUT [flags]
  local srcf=$1 out=$2
  shift 2
  docker run --rm -u "$(id -u):$(id -g)" -v "$ROOT:/w" -w /w "$IMAGE" \
    g++ -O3 -DNDEBUG -std=c++17 -mavx2 -mfma -mf16c -mbmi2 -I"/w/.cache/kernel-bench" -I"/w/$HERE" "$@" \
    -o "/w/.cache/kernel-bench/$out" "/w/$HERE/$srcf"
}

case ${1:-} in
gemv)  build kb.cpp kb; taskset -c "${2:-3}" "$GEN/kb" ;;
multi) build kb.cpp kb; taskset -c "${2:-3}" "$GEN/kb" multi ;;
gemm)  build kg.cpp kg; taskset -c "${2:-3}" "$GEN/kg" ;;
check) build kb.cpp kb; build kg.cpp kg; "$GEN/kb" check; "$GEN/kg" check ;;
uops)  shift; build uops.cpp uops
       printf "%-12s %8s %8s %8s %8s %8s %8s\n" ins cyc/ins uops/ins p0 p1 p2 p3
       [ $# -gt 0 ] || set -- maddubs maddwd phaddd
       for i in "$@"; do
         out=$(taskset -c "${CPU:-3}" perf stat -x, -e cycles,fpu_pipe_assignment.total0,fpu_pipe_assignment.total1,fpu_pipe_assignment.total2,fpu_pipe_assignment.total3 "$GEN/uops" "$i" 5000000 2>&1 >/dev/null)
         echo "$out" | awk -F, -v n="$i" -v tot=40000000 '/cycles/{c=$1} /total0/{p0=$1} /total1/{p1=$1} /total2/{p2=$1} /total3/{p3=$1} END{printf "%-12s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n", n, c/tot, (p0+p1+p2+p3)/tot, p0/tot, p1/tot, p2/tot, p3/tot}'
       done ;;
*) sed -n 2,11p "$0"; exit 2 ;;
esac
