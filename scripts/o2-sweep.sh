#!/usr/bin/env bash
# Stage 2.6 (O2): batch sizes and KV cache type per profile. n_ubatch 256, 512,
# 1024 against the KV cache type f16 and q8_0, with the depth-N decode scenario
# on (--full), because the KV cache type only matters once the context is
# long. Thread counts were swept in 2.0 (docs/STAGE2.md); the winners become the
# planner's defaults (internal/plan/targets.go, planner.go) only when the gain
# is outside the spread, and the losers stay as options.
#
#   scripts/o2-sweep.sh [LABEL] [CONFIG]
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:-o2}
CONFIG=${2:-$HOME/nanollm-data/base.yaml}
NANOLLM=${NANOLLM:-./dist/nanollm}
REPS=${REPS:-3}

run() { # run NAME ARGS...
	local name=$1
	shift
	echo "=== $(date +%T) $name"
	nice -n 5 "$NANOLLM" bench --config "$CONFIG" --suite dev --full --reps "$REPS" --timeout 25m --label "$name" "$@"
}

run "$LABEL-strong" --profile strong --preset ram-8g --preset ram-4g --set n_ubatch=256,512,1024 --set kv=f16,q8_0
run "$LABEL-weak" --profile weak --preset ram-4g --set n_ubatch=256,512,1024 --set kv=f16,q8_0
echo "=== $(date +%T) done"
