#!/usr/bin/env bash
# The cost of a batch of n tokens on the CPU, n = 1..8 (docs/OPT-RESEARCH.md,
# OPT-17). Decode with MTP or prompt lookup checks n = 1 + drafts tokens in
# one pass; the repack kernels of ggml run a batch of 2 or 3 rows as separate
# gemv calls and use gemm only from 4 rows. This prints the time per pass for
# every n, on the strong and the weak profile, so the planner can pick the
# draft length from a measured curve and OPT-16 has a number to improve.
# Run it under scripts/hyper-exclusive.sh.
#
#   scripts/tn-curve.sh [TOOLS_DIR]    # default dist/linux_amd64/tools
#
# Output: bench/results/<host>/<date>-tn-curve[-LABEL].txt (LABEL from $TN_LABEL)
set -uo pipefail
cd "$(dirname "$0")/.."
D=${1:-dist/linux_amd64/tools}
MODEL=${MODEL:-$(ls "$HOME"/nanollm-data/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)}
OUT=bench/results/$(hostname -s)/$(date +%F)-tn-curve${TN_LABEL:+-$TN_LABEL}.txt
mkdir -p "$(dirname "$OUT")"
CPUS=$(python3 -I - <<'PY'
cpus = []
for part in open("/sys/devices/system/node/node0/cpulist").read().strip().split(","):
    a, _, b = part.partition("-")
    cpus += range(int(a), int(b or a) + 1)
print(",".join(map(str, cpus[:4])))
PY
)
{
	echo "# tn-curve $(date -Is) $(basename "$MODEL") tools $D"
	# -b/-ub 8: one pass per n. -p n with -n 0 times the prompt pass alone.
	echo "## strong: 16 threads, all nodes"
	nice -n 5 numactl --interleave=all "$D/llama-bench" -m "$MODEL" -t 16 -p 1,2,3,4,5,6,7,8 -n 0 -b 8 -ub 8 -fa 1 -r 5 -o md 2>/dev/null
	echo "## weak: 4 cores of node 0 (cpus $CPUS)"
	nice -n 5 numactl --preferred=0 taskset -c "$CPUS" "$D/llama-bench" -m "$MODEL" -t 4 -p 1,2,3,4,5,6,7,8 -n 0 -b 8 -ub 8 -fa 1 -r 5 -o md 2>/dev/null
} | tee "$OUT"
