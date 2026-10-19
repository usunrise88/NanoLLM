#!/usr/bin/env bash
# Stage 2.3 sweep: every ram-* preset under a strict cgroup budget on two
# hardware profiles (strong: all nodes, 16 threads; weak: 4 cores of one node,
# cold disk), and the repack on/off comparison for the modes that keep weights
# in a file (M7, M8). Run it with the machine otherwise quiet: a strict budget
# charges the page cache to whoever read the pages first.
#
#   scripts/modes-sweep.sh [LABEL] [CONFIG]
#
# Results go to bench/results/<host>/ (one file per run, never overwritten).
set -euo pipefail

cd "$(dirname "$0")/.."
LABEL=${1:-modes}
CONFIG=${2:-$HOME/nanollm-data/base.yaml}
NANOLLM=${NANOLLM:-./dist/nanollm}
REPS=${REPS:-3}
TIMEOUT=${TIMEOUT:-15m}

# Presets by mode on the E4B: M6 (8g, 6g), M7 (4g, 3g), M8 (2g, 1.5g), M9 (1g).
ALL=(ram-8g ram-6g ram-4g ram-3g ram-2g ram-1.5g ram-1g)
FILE_MODES=(ram-4g ram-3g ram-2g ram-1.5g)

args() { for p in "$@"; do printf -- '--preset %s ' "$p"; done; }

run() { # run NAME ARGS...
	local name=$1
	shift
	echo "=== $(date +%T) $name"
	nice -n 5 "$NANOLLM" bench --config "$CONFIG" --suite dev --reps "$REPS" --timeout "$TIMEOUT" --label "$name" "$@"
}

# shellcheck disable=SC2046
run "$LABEL-strong" --profile strong $(args "${ALL[@]}")
# shellcheck disable=SC2046
run "$LABEL-strong-repack" --profile strong --set repack=on,off $(args "${FILE_MODES[@]}")
# shellcheck disable=SC2046
run "$LABEL-weak" --profile weak $(args "${ALL[@]}")
# shellcheck disable=SC2046
run "$LABEL-weak-repack" --profile weak --set repack=on,off $(args "${FILE_MODES[@]}")
# MADV_RANDOM on the PLE table (the file-backed, randomly read part): on and off.
# shellcheck disable=SC2046
run "$LABEL-advise" --profile strong --set env.NANOLLM_NO_ADVISE_RANDOM=0,1 --preset ram-3g --preset ram-2g
echo "=== $(date +%T) done"
