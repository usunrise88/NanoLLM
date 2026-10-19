#!/usr/bin/env bash
# The nightly run on a development host with the real model (docs/STAGE2.md 2.0):
# build, vet, unit tests, the native tests on the model with the long cases on,
# and the data a regression would show in: a short bench, eval, and gate.
#
#   scripts/nightly.sh [--quick]
#
# Run it from cron as the user that owns the model:
#
#   30 2 * * *  /home/ubuntu/nanollm/scripts/nightly.sh
#
# There is no self-hosted CI runner for this: the repository is public and the
# host is shared. Each step runs in a transient cgroup with a memory limit and
# at low priority, so a stuck test cannot take the neighbours' memory or CPU.
# Logs and a summary go to $NANOLLM_NIGHTLY_DIR/<date>/; nothing is pushed.
#
# Environment:
#   NANOLLM_NIGHTLY_DIR   where logs go (default ~/nanollm-data/nightly)
#   NANOLLM_TEST_MODELS   the E4B model directory (default: the one under ~/nanollm-data)
#   NIGHTLY_MEM           memory limit per step (default 16G)
#   NIGHTLY_KEEP          days of logs to keep (default 14)
set -uo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/go/bin:$HOME/go/bin:$PATH"

quick=0
[ "${1:-}" = "--quick" ] && quick=1

root=${NANOLLM_NIGHTLY_DIR:-$HOME/nanollm-data/nightly}
out=$root/$(date +%F)
mkdir -p "$out"
summary=$out/summary.txt
: >"$summary"

models=${NANOLLM_TEST_MODELS:-$(ls -d "$HOME"/nanollm-data/models/gemma-4-e4b-qat/*/ 2>/dev/null | head -1)}
export NANOLLM_TEST_MODELS=${models%/}
export NANOLLM_TEST_LIB_DIR=$PWD/dist/linux_amd64/lib
export NANOLLM_TEST_LONG=1
export NANOLLM_SERVER_DATA_DIR=${NANOLLM_SERVER_DATA_DIR:-$HOME/nanollm-data}
mem=${NIGHTLY_MEM:-16G}
failed=0

# step NAME CMD...  runs CMD in a cgroup scope with a memory limit, at low
# priority, logging to $out/NAME.log, and records the result.
step() {
	local name=$1
	shift
	local start=$SECONDS
	if command -v systemd-run >/dev/null && systemd-run --user --scope --quiet true 2>/dev/null; then
		systemd-run --user --scope --quiet -p MemoryMax="$mem" -p MemorySwapMax=0 -- nice -n 10 "$@" >"$out/$name.log" 2>&1
	else
		nice -n 10 "$@" >"$out/$name.log" 2>&1
	fi
	local rc=$?
	printf '%-14s %s  %ds\n' "$name" "$([ $rc -eq 0 ] && echo ok || echo "FAIL($rc)")" $((SECONDS - start)) | tee -a "$summary"
	[ $rc -eq 0 ] || failed=1
}

echo "nightly $(date -Is) $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty) load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$summary"

step build make build
step vet go vet ./...
step unit go test ./... -count=1
[ -d "$NANOLLM_TEST_MODELS" ] || { echo "no model under NANOLLM_TEST_MODELS: native steps skipped" | tee -a "$summary"; exit $failed; }
step native go test -tags native -count=1 -timeout 60m ./internal/backend/llamacpp/...
[ $quick -eq 1 ] && exit $failed

cfg=$out/nanollm.yaml
cat >"$cfg" <<YAML
server:
  data_dir: $NANOLLM_SERVER_DATA_DIR
backends:
  lib_dir: $NANOLLM_TEST_LIB_DIR
YAML
step bench ./dist/nanollm bench --config "$cfg" --suite dev --advisory --preset ram-8g --reps 3 --label nightly --out "$out/bench.json"
step eval ./dist/nanollm eval --config "$cfg" --suite json-tools --out "$out/eval-json-tools.json"

find "$root" -mindepth 1 -maxdepth 1 -type d -mtime +"${NIGHTLY_KEEP:-14}" -exec rm -rf {} + 2>/dev/null
exit $failed
