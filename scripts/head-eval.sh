#!/usr/bin/env bash
# Stage 2 plan task 12 (OPT-4): the capability gate of the output shortlist. The reference
# configuration (CPU, KV f16, no MTP, the whole head) against the same with models.<id>.head:
# shortlist (and shortlist-small): four suites with thinking off and on, item by item
# (scripts/quality-gate.sh), then the reasoning-direct suite with the shortlist. Quality, not
# speed; it runs on one NUMA node.
#
#   NODE=1 NANOLLM=./dist/nanollm-next3 scripts/head-eval.sh [CONFIG...]   # default: ref-head ref-head-small
set -uo pipefail
cd "$(dirname "$0")/.."
export NANOLLM=${NANOLLM:-./dist/nanollm-next3}
export NANOLLM_DEV_SERIES=${NANOLLM_DEV_SERIES:-$(git -C .cache/llama.cpp/src rev-parse HEAD)}
NODE=${NODE:-1}
export NANOLLM_PLACE="node:$NODE;cpus=$(cat /sys/devices/system/node/node$NODE/cpulist)"
CONFIGS=("$@")
[[ ${#CONFIGS[@]} -gt 0 ]] || CONFIGS=(ref-head ref-head-small)
rc=0
for cfg in "${CONFIGS[@]}"; do
	THINKING=both scripts/quality-gate.sh "$cfg" || rc=1
done
echo "=== $(date +%T) done"
exit $rc
