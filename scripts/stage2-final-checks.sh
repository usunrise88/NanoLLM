#!/usr/bin/env bash
# What stage 2 still needs on the final series (0001 + 0002 + 0003 in
# dist/linux_amd64-next) once the night queue (scripts/night-hdd.sh) is done, each
# step under the benchmark lock:
#   1. the MTP sweep of 2.7 on the new kernels (a batch of 2-3 tokens costs about
#      one token now, docs/AUTOTUNE.md), strong and weak profile;
#   2. the eval of the series, thinking off and on, all four suites, against the
#      reference item by item (the kernels are bit-exact, so the answers must
#      match), without and with prompt lookup;
#   3. the capability gate of repack off (thinking off and on) and of KV cache
#      q8_0 on the long-context suite.
#
#   scripts/stage2-final-checks.sh        # waits for "=== HH:MM:SS done" in the night log
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/go/bin:$HOME/go/bin:$PATH
until grep -q "=== [0-9:]* done" ~/nanollm-data/sweep/night-hdd.log; do sleep 120; done
echo "=== $(date +%T) night queue done; post-night starts"
export NANOLLM_DEV_SERIES=$(tr -d '[:space:]' < .cache/next.commit)
echo "=== $(date +%T) MTP sweep on the next series"
NANOLLM=./dist/nanollm-next scripts/hyper-exclusive.sh -- scripts/mtp-sweep.sh mtp-next ~/nanollm-data/next.yaml
for c in ref-next ref-next-ngram; do
	echo "=== $(date +%T) eval of $c, thinking off and on"
	THINKING=both NANOLLM=./dist/nanollm-next scripts/hyper-exclusive.sh --max-load 99 -- scripts/quality-gate.sh $c
done
echo "=== $(date +%T) gate of repack off, thinking off and on"
THINKING=both scripts/hyper-exclusive.sh --max-load 99 -- scripts/quality-gate.sh ref-norepack
echo "=== $(date +%T) gate of KV q8_0 on the needle suite"
scripts/hyper-exclusive.sh --max-load 99 -- scripts/quality-gate.sh ref-kvq8 long-context
echo "=== $(date +%T) post-night done"
