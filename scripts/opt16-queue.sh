#!/usr/bin/env bash
# The OPT-16 timing queue (docs/AUTOTUNE.md, "OPT-16"): llama-bench A/B of the Q4_0 kernels on both profiles, and the
# one-thread kernel micro-benchmarks. Every step takes scripts/hyper-exclusive.sh on its own, so the lock is let go
# between steps. Raw output: bench/results/hyper/opt16-*.txt, summary: scripts/kernel-bench/summarize.py.
#
#   scripts/opt16-queue.sh            # all steps, about 2 hours of exclusive host in total
#   STEPS="full" scripts/opt16-queue.sh
#
# The library sets (tools directories with llama-bench) it compares, all under .cache/devab:
#   base     dist/linux_amd64/tools of the shipped build, b11600 + 0001 (NANOLLM_BASE_TOOLS to change)
#   F        the first build of the patch: GEMV vphaddd, GEMM 4 rows
#   F2       the build of the patch as it is now (dist/linux_amd64/tools after scripts/build-libs.sh)
#   D2       the same source with the run-time switches of kernel-bench/devlayer.py (Q4K_GEMV, Q4K_GEMM)
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
R=bench/results/hyper
AB=scripts/ab-llama-bench.sh
BASE=${NANOLLM_BASE_TOOLS:-/home/ubuntu/nanollm/dist/linux_amd64/tools}
F=$ROOT/.cache/devab/F
F2=$ROOT/.cache/devab/F2
D=$ROOT/.cache/devab/D2
STEPS=${STEPS:-"full decode prefill micro"}
excl() { scripts/hyper-exclusive.sh -- "$@"; }

for step in $STEPS; do
  echo "step $step $(date -Is)"
  case $step in
  full) # pp512 + tg128, both profiles, 2 interleaved rounds
    excl $AB $R/opt16-kernels-full.txt full 2 weak,strong \
      base=$BASE gemv-hadd-gemm4-hadd=$F final=$F2 devlib-old-kernels=$D@Q4K_GEMV=0,Q4K_GEMM=0 ;;
  decode) # the GEMV variants
    excl $AB $R/opt16-gemv-variants-decode.txt decode 2 weak,strong \
      base=$BASE gemv-hadd=$D@Q4K_GEMV=2 gemv-shuf=$D@Q4K_GEMV=3 gemv-phaddw=$D@Q4K_GEMV=1 \
      gemv-ng2-phaddw=$D@Q4K_GEMV=4 gemv-pf8-phaddw=$D@Q4K_GEMV=5 ;;
  prefill) # the GEMM variants
    excl $AB $R/opt16-gemm-variants-prefill.txt prefill 2 weak,strong \
      base=$BASE gemm-r4-hadd=$D@Q4K_GEMM=2 gemm-r4-phaddw=$D@Q4K_GEMM=3 gemm-r8-hadd=$D@Q4K_GEMM=4 \
      gemm-r8-phaddw=$D@Q4K_GEMM=1 gemm-r12-phaddw=$D@Q4K_GEMM=5 gemm-r16-phaddw=$D@Q4K_GEMM=6 ;;
  micro) # kernel-bench, one thread on cpu 3, 25 interleaved rounds
    export KB_ROUNDS=25
    { echo "# $(date -Is) scripts/kernel-bench/run.sh gemv 3 (KB_ROUNDS=25), commit $(git rev-parse --short HEAD)"
      excl scripts/kernel-bench/run.sh gemv 3; } > $R/opt16-kernel-bench-gemv.txt 2>&1
    { echo "# $(date -Is) scripts/kernel-bench/run.sh gemm 3 (KB_ROUNDS=25), commit $(git rev-parse --short HEAD)"
      excl scripts/kernel-bench/run.sh gemm 3; } > $R/opt16-kernel-bench-gemm.txt 2>&1
    { echo "# $(date -Is) scripts/kernel-bench/run.sh uops (cpu 3)"
      CPU=3 excl scripts/kernel-bench/run.sh uops maddubs maddwd addw and srlw pshufb phaddd phaddw pshufd shufps blendd \
        bcastq bcastd cvtph2ps mulps fma subd xor load xorm maddubsm perm2i permq; } > $R/opt16-uops.txt 2>&1 ;;
  *) echo "unknown step $step" >&2 ;;
  esac
done
echo "done $(date -Is)"
