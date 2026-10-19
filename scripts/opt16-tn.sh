#!/usr/bin/env bash
# The cost of a batch of 1..8 tokens (scripts/tn-curve.sh, from the OPT-17 research) for the shipped build and for the
# OPT-16 build, each under its own hyper-exclusive lock.
# Output: bench/results/<host>/<date>-tn-curve-opt16-{base,final}.txt
#
#   scripts/opt16-tn.sh
set -uo pipefail
cd "$(dirname "$0")/.."
BASE=${NANOLLM_BASE_TOOLS:-/home/ubuntu/nanollm/dist/linux_amd64/tools}
FINAL=${NANOLLM_OPT16_TOOLS:-$PWD/.cache/devab/F2}
TN=scripts/tn-curve.sh
[[ -x $TN ]] || TN=/home/ubuntu/nanollm/scripts/tn-curve.sh
TN_LABEL=opt16-base scripts/hyper-exclusive.sh -- "$TN" "$BASE"
TN_LABEL=opt16-final scripts/hyper-exclusive.sh -- "$TN" "$FINAL"
echo "done $(date -Is)"
