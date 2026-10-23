#!/usr/bin/env bash
# Stage 2 plan task 12 (OPT-4): the speed of the output shortlist. The dev suite on both
# profiles with the whole head, the shortlist (65 % of the rows) and the small one
# (37 %). The libraries are those of the next series (patch 0009), the binary is stamped
# for them: dist/nanollm-next3 with NANOLLM_DEV_SERIES set to the series commit. Run it
# under scripts/hyper-exclusive.sh.
#
#   scripts/hyper-exclusive.sh -- scripts/head-sweep.sh [LABEL]
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:-head}
CONFIG=${CONFIG:-$HOME/nanollm-data/next3.yaml}
NANOLLM=${NANOLLM:-./dist/nanollm-next3}
export NANOLLM_DEV_SERIES=${NANOLLM_DEV_SERIES:-$(git -C .cache/llama.cpp/src rev-parse HEAD)}
REPS=${REPS:-5}

run() { # run NAME ARGS...
	local name=$1
	shift
	echo "=== $(date +%T) $name"
	nice -n 5 "$NANOLLM" bench --config "$CONFIG" --suite dev --reps "$REPS" --timeout 30m --label "$name" "$@"
}

run "$LABEL-strong" --profile strong --preset ram-8g --set head=full,shortlist,shortlist-small
run "$LABEL-weak" --profile weak --preset ram-8g --set head=full,shortlist,shortlist-small
echo "=== $(date +%T) done"
