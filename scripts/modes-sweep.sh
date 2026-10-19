#!/usr/bin/env bash
# Stage 2.3 sweep: every ram-* preset under a strict cgroup budget on two
# hardware profiles (strong: all nodes, 16 threads; weak: 4 cores of one node,
# cold disk), and the repack on/off comparison for the modes that keep weights
# in a file (M7, M8). Run it with the machine otherwise quiet: a strict budget
# charges the page cache to whoever read the pages first.
#
#   scripts/modes-sweep.sh [LABEL] [CONFIG]
#
# M6 and M7 (ram-8g … ram-3g) run in full. M8 and M9 (ram-2g and below) read
# part of the weights from the disk on every token: on a hard disk that is
# seconds per token, so they run with --slow (pp and 12 decode tokens, one
# repetition) and a time limit; a run that does not finish is recorded as too
# slow. Results go to bench/results/<host>/ (one file per run, never
# overwritten).
set -euo pipefail

cd "$(dirname "$0")/.."
LABEL=${1:-modes}
CONFIG=${2:-$HOME/nanollm-data/base.yaml}
NANOLLM=${NANOLLM:-./dist/nanollm}
REPS=${REPS:-3}
TIMEOUT=${TIMEOUT:-15m}
SLOW_TIMEOUT=${SLOW_TIMEOUT:-12m}

RESIDENT=(ram-8g ram-6g ram-4g ram-3g) # M6, M6, M7, M7
STREAMING=(ram-2g ram-1.5g ram-1g)     # M8, M8, M9
M7=(ram-4g ram-3g)
M8=(ram-2g ram-1.5g)

presets() { for p in "$@"; do printf -- '--preset\n%s\n' "$p"; done; }

run() { # run NAME ARGS...
	local name=$1
	shift
	echo "=== $(date +%T) $name"
	nice -n 5 "$NANOLLM" bench --config "$CONFIG" --suite dev --label "$name" "$@"
}
full() { local name=$1; shift; run "$name" --reps "$REPS" --timeout "$TIMEOUT" "$@"; }
slow() { local name=$1; shift; run "$name" --slow --reps 1 --timeout "$SLOW_TIMEOUT" "$@"; }

for profile in strong weak; do
	mapfile -t a < <(presets "${RESIDENT[@]}")
	full "$LABEL-$profile" --profile "$profile" "${a[@]}"
	mapfile -t a < <(presets "${M7[@]}")
	full "$LABEL-$profile-repack" --profile "$profile" --set repack=on,off "${a[@]}"
	mapfile -t a < <(presets "${STREAMING[@]}")
	slow "$LABEL-$profile-streaming" --profile "$profile" "${a[@]}"
	mapfile -t a < <(presets "${M8[@]}")
	slow "$LABEL-$profile-streaming-repack" --profile "$profile" --set repack=on,off "${a[@]}"
done
# MADV_RANDOM on the PLE table (the file-backed, randomly read part): on and off.
full "$LABEL-advise" --profile strong --set env.NANOLLM_ADVISE_RANDOM=0,1 --preset ram-3g
echo "=== $(date +%T) done"
