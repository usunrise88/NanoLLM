#!/usr/bin/env bash
# The capability gate of a candidate configuration against the reference
# (docs/QUALITY-STAGE2.md, 5): the same suites with thinking off, then
# `nanollm eval gate` per suite and capability.
#
#   scripts/quality-gate.sh CONFIG [SUITE...]
#
# CONFIG is a name: ~/nanollm-data/CONFIG.yaml is the candidate, e.g. ref-norepack
# (models.<id>.repack: off) or ref-kvq8 (kv_type: q8_0). The reference reports are
# ~/nanollm-data/eval/ref-SUITE.json (thinking off and on); the gate compares with
# the thinking-off part of them. Reports go to ~/nanollm-data/eval/CONFIG-SUITE.json.
# Quality, not speed: it needs no quiet host, but its load spoils a timing run, so
# start it under scripts/hyper-exclusive.sh --max-load 99 while a queue is running.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=$1
shift
SUITES=("$@")
[[ ${#SUITES[@]} -gt 0 ]] || SUITES=(ru-instruct en-instruct json-tools long-context)
D=${NANOLLM_DATA:-$HOME/nanollm-data}
BIN=${NANOLLM:-./dist/nanollm-stage}
rc=0
for s in "${SUITES[@]}"; do
	ref=$D/eval/ref-$s.json
	cand=$D/eval/$CFG-$s.json
	off=$(mktemp --suffix=.json)
	if [[ ! -f $cand ]]; then
		echo "=== $(date +%T) eval $s with $CFG (thinking off)"
		nice -n 10 "$BIN" eval --config "$D/$CFG.yaml" --suite "$s" --thinking off --out "$cand" 2>&1 | tail -3 || rc=1
	fi
	jq '.results |= map(select(.thinking|not)) | .summary |= with_entries(select(.key|endswith("+thinking")|not)) | del(.by_lang)' "$ref" > "$off"
	echo "=== $(date +%T) gate $s: reference (thinking off) vs $CFG"
	"$BIN" eval gate "$off" "$cand" || rc=1
	python3 -I scripts/eval-diff.py "$off" "$cand" | tail -2
done
exit $rc
