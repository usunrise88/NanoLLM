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
# The script first waits, WITHOUT the lock, until the 1-minute load average
# drops under --max-load (default 6); only then does it queue for the lock, and
# once it has it checks the load again (it may have changed while queued). A
# job that waited for a quiet host while holding the lock would keep every
# other job out, including those that accept any load (--max-load 99). It
# gives up after --wait seconds (default 6 h) with status 75.
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
load() { cut -d' ' -f1 /proc/loadavg; }
quiet() { awk -v l="$(load)" -v m="$MAX_LOAD" 'BEGIN{exit !(l < m)}'; }

deadline=$((SECONDS + WAIT))
while :; do
  until quiet; do
    if ((SECONDS >= deadline)); then
      echo "hyper-exclusive: load $(load) stayed above $MAX_LOAD" >&2
      exit 75
    fi
    sleep 20
  done
  exec 9>"$LOCK"
  left=$((deadline - SECONDS))
  ((left >= 1)) || left=1
  if ! flock -w "$left" 9; then
    echo "hyper-exclusive: could not take $LOCK in ${WAIT}s" >&2
    exit 75
  fi
  quiet && break
  exec 9>&- # the load rose while queued: let the others go and wait again
done
echo "hyper-exclusive: lock taken, load $(load), running ($*)" >&2
"$@"
