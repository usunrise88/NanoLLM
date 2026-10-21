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
IMAGE=nanollm-build-libs:ubuntu22.04

GEN=$GEN LLAMA_SRC=$SRC "$ROOT/$HERE/gen.sh"

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
