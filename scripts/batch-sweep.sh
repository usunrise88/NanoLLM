#!/usr/bin/env bash
# Stage 2 plan task 5: the throughput of several sequences decoding at once,
# with and without MTP (the CPU part of the O3 gate). The dev suite adds, for the
# batch key, one point per 1, 2, 4… up to N sequences (internal/bench/worker.go,
# batchCurve): the aggregate rate over the wall time, the median rate of one
# sequence, and the same with drafting on. Run it under scripts/hyper-exclusive.sh.
#
#   scripts/hyper-exclusive.sh -- scripts/batch-sweep.sh [LABEL] [CONFIG]
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:-batch}
CONFIG=${2:-$HOME/nanollm-data/base.yaml}
NANOLLM=${NANOLLM:-./dist/nanollm}
REPS=${REPS:-5}

run() { # run NAME ARGS...
	local name=$1
	shift
	echo "=== $(date +%T) $name"
	nice -n 5 "$NANOLLM" bench --config "$CONFIG" --suite dev --reps "$REPS" --timeout 30m --label "$name" "$@"
}

run "$LABEL-strong" --profile strong --preset ram-8g --set batch=8
run "$LABEL-weak" --profile weak --preset ram-8g --set batch=4
echo "=== $(date +%T) done"
