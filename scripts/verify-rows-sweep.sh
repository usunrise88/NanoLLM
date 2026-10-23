#!/usr/bin/env bash
# Stage 2 plan task 5: how many rows of a step the scheduler may spend on MTP
# drafts and their checks (sched.Loop.SetVerifyRows) so that several sequences
# decoding at once do not lose throughput to drafting. verify_rows 0 is no bound
# (every sequence drafts as far as draft_max); 3 is one sequence with two drafts
# and nothing for a second; 4, 6 and 8 let two to four sequences draft too.
# Run it under scripts/hyper-exclusive.sh.
#
#   scripts/hyper-exclusive.sh -- scripts/verify-rows-sweep.sh [LABEL] [CONFIG]
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:-verify-rows}
CONFIG=${2:-$HOME/nanollm-data/base.yaml}
NANOLLM=${NANOLLM:-./dist/nanollm}
REPS=${REPS:-5}
ROWS=${ROWS:-0,3,4,6,8}

run() { # run NAME ARGS...
	local name=$1
	shift
	echo "=== $(date +%T) $name"
	nice -n 5 "$NANOLLM" bench --config "$CONFIG" --suite dev --reps "$REPS" --timeout 30m --label "$name" "$@"
}

run "$LABEL-strong" --profile strong --preset ram-8g --set only=batch --set batch=4 --set verify_rows="$ROWS"
run "$LABEL-weak" --profile weak --preset ram-8g --set only=batch --set batch=4 --set verify_rows="$ROWS"
echo "=== $(date +%T) done"
