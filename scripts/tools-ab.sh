#!/usr/bin/env bash
# A/B of llama-bench builds on both hardware profiles (docs/AUTOTUNE.md):
# the same model, the same threads, the builds interleaved A B C A B C so that
# drift in the host's load hits every build alike. Run it under
# scripts/hyper-exclusive.sh.
#
#   scripts/tools-ab.sh LABEL TOOLS_DIR... [-- ROUNDS]
#
# Each TOOLS_DIR holds a llama-bench (dist/<build>/tools). Output:
# bench/results/<host>/<date>-ab-LABEL.txt, one block per build and round.
set -uo pipefail
cd "$(dirname "$0")/.."
LABEL=$1
shift
DIRS=()
ROUNDS=2
while [[ $# -gt 0 ]]; do
	if [[ $1 == -- ]]; then ROUNDS=$2; break; fi
	DIRS+=("$1")
	shift
done
MODEL=${MODEL:-$(ls "$HOME"/nanollm-data/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)}
OUT=bench/results/$(hostname -s)/$(date +%F)-ab-$LABEL.txt
mkdir -p "$(dirname "$OUT")"
NODE0=$(cut -d, -f1- /sys/devices/system/node/node0/cpulist | tr -d '\n')
WEAK=$(python3 - <<PY
import re
s="$NODE0"
cpus=[]
for part in s.split(','):
    a,_,b=part.partition('-')
    cpus+=range(int(a),int(b or a)+1)
print(','.join(map(str,cpus[:4])))
PY
)
{
	echo "# $LABEL $(date -Is) model $(basename "$MODEL") builds: ${DIRS[*]}"
	for r in $(seq 1 "$ROUNDS"); do
		for d in "${DIRS[@]}"; do
			echo "## round $r strong 16 threads, all nodes: $d"
			nice -n 5 numactl --interleave=all "$d/llama-bench" -m "$MODEL" -t 16 -p 512 -n 128 -fa 1 -r 3 -o md 2>/dev/null
			echo "## round $r weak 4 cores of node 0 (cpus $WEAK): $d"
			nice -n 5 numactl --preferred=0 taskset -c "$WEAK" "$d/llama-bench" -m "$MODEL" -t 4 -p 512 -n 64 -fa 1 -r 3 -o md 2>/dev/null
		done
	done
} | tee "$OUT"
