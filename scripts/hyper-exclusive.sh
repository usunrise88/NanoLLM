#!/usr/bin/env bash
# Run a timing-sensitive command with the host to ourselves.
#
#   scripts/hyper-exclusive.sh [--max-load N] [--wait SECONDS] -- command args...
#
# hyper is shared: other services, and several of our own jobs (KLD, eval,
# builds, agents), run on it. A throughput number measured while another
# benchmark runs is noise, so every benchmark takes this lock first. The lock
# is an flock on ~/nanollm-data/hyper-bench.lock, so concurrent benchmark
# jobs queue instead of overlapping. Jobs that do not measure time (builds,
# KLD, eval) do not take it.
#
# Before running, the script waits until the 1-minute load average, minus what
# the lock holders themselves could add, drops under --max-load (default 6),
# or gives up after --wait seconds (default 6 h) with status 75.
set -euo pipefail

MAX_LOAD=6
WAIT=21600
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-load) MAX_LOAD=$2; shift 2 ;;
    --wait) WAIT=$2; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[[ $# -gt 0 ]] || { echo "hyper-exclusive: no command" >&2; exit 2; }

LOCK=${NANOLLM_BENCH_LOCK:-$HOME/nanollm-data/hyper-bench.lock}
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -w "$WAIT" 9; then
  echo "hyper-exclusive: could not take $LOCK in ${WAIT}s" >&2
  exit 75
fi
echo "hyper-exclusive: lock taken ($*)" >&2

deadline=$((SECONDS + WAIT))
while :; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN{exit !(l < m)}'; then break; fi
  if ((SECONDS >= deadline)); then
    echo "hyper-exclusive: load $load stayed above $MAX_LOAD" >&2
    exit 75
  fi
  sleep 20
done
echo "hyper-exclusive: load $load, running" >&2
"$@"
