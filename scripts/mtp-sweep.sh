#!/usr/bin/env bash
# Stage 2.7: MTP on the CPU. draft_max 1..4 against draft_p_min 0, 0.5, 0.75,
# on both profiles (the speed-up is the ratio of the MTP chat rate to the plain
# one in the same worker). The bench prompts decode greedily; temperature 0.7
# is checked separately by `nanollm eval` on the reference configuration.
#
#   scripts/mtp-sweep.sh [LABEL] [CONFIG]
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:-mtp}
CONFIG=${2:-$HOME/nanollm-data/base.yaml}
NANOLLM=${NANOLLM:-./dist/nanollm}
REPS=${REPS:-5}

run() { # run NAME ARGS...
	local name=$1
	shift
	echo "=== $(date +%T) $name"
	nice -n 5 "$NANOLLM" bench --config "$CONFIG" --suite dev --reps "$REPS" --timeout 20m --label "$name" "$@"
}

run "$LABEL-strong" --profile strong --preset ram-8g --set draft_max=1,2,3,4 --set draft_p_min=0,0.5,0.75
run "$LABEL-weak" --profile weak --preset ram-4g --set draft_max=1,2,3,4 --set draft_p_min=0,0.5,0.75
echo "=== $(date +%T) done"
