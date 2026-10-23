#!/usr/bin/env bash
# The measurements of the three optimizations of stage 2 plan tasks 10-12 that a hard disk
# can answer (docs/AUTOTUNE.md: OPT-7 layout, OPT-9 stream methods, OPT-4 head), on the
# libraries of the next series (patches 0009 and 0010) and a binary stamped for them:
#
#   scripts/opt-queue.sh [LABEL]
#   STEPS="layout stream" scripts/opt-queue.sh     # a part of it
#
# Every measurement takes the host lock for itself only (scripts/hyper-exclusive.sh). The
# hard disk of hyper (RAID1, ~110 MB/s) cannot show what an NVMe disk shows: these runs say
# whether the new paths are right and what they cost where the disk is the limit; the same
# queue is meant to be run on the laptop (STEPS, DATA and NANOLLM point it at the machine).
#
#   layout  OPT-7: the model file as it is and relaid (models relayout), M6/M7 loaded cold
#           and M8/M9 streaming
#   stream  OPT-9: stream_method advise, pread and direct on M8/M9
#   head    OPT-4: the head full, shortlist and shortlist-small, both profiles
set -uo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:-opt}
DATA=${DATA:-$HOME/nanollm-data}
NANOLLM=${NANOLLM:-./dist/nanollm-next3}
CONFIG=${CONFIG:-$DATA/next3.yaml}
export NANOLLM_DEV_SERIES=${NANOLLM_DEV_SERIES:-$(git -C .cache/llama.cpp/src rev-parse HEAD)}
STEPS=${STEPS-"layout stream head"}
SLOW_TIMEOUT=${SLOW_TIMEOUT:-14m}

x() { scripts/hyper-exclusive.sh --max-load 6 -- "$@"; }
bench() { # bench NAME ARGS...
	local name=$1
	shift
	echo "=== $(date +%T) $name"
	x nice -n 5 "$NANOLLM" bench --config "$CONFIG" --suite dev --label "$LABEL-$name" "$@" || echo "!!! $name failed (rc=$?)"
}
slow() { local name=$1; shift; bench "$name" --slow --reps 1 --timeout "$SLOW_TIMEOUT" "$@"; }
full() { local name=$1; shift; bench "$name" --reps "${REPS:-3}" --timeout 20m "$@"; }

for step in $STEPS; do
	case $step in
	layout)
		# The relaid copy sits next to the model; writing it is not a measurement.
		"$NANOLLM" models relayout gemma-4-e4b-qat --config "$CONFIG" --verify || echo "!!! relayout failed"
		for profile in strong weak; do
			full "layout-$profile" --profile "$profile" --set layout=original,nano --preset ram-8g --preset ram-4g --preset ram-3g
			slow "layout-$profile-streaming" --profile "$profile" --set layout=original,nano --preset ram-2g --preset ram-1g
		done
		;;
	stream)
		for profile in strong weak; do
			slow "stream-$profile" --profile "$profile" --preset ram-2g --preset ram-1g --set stream_ahead=2 \
				--set stream_method=advise,pread,direct
		done
		;;
	head)
		for profile in strong weak; do
			full "head-$profile" --profile "$profile" --preset ram-8g --set head=full,shortlist,shortlist-small
		done
		;;
	esac
done
echo "=== $(date +%T) done"
