#!/usr/bin/env bash
# The soak test of the server on the real model (docs/STAGE2.md 2.8): 24 hours
# of mixed load against a server of 8 threads and 12 GB, in one NUMA node, at
# low priority, on a loopback port.
#
#   scripts/soak.sh [HOURS]            # default 24
#   go run ./tools/loadgen -analyze ~/nanollm-data/soak/<date>/loadgen.jsonl
#
# The server runs in a transient cgroup with MemoryMax=12G and no swap, so a
# leak ends in an OOM kill (a failed soak) instead of in the neighbours' memory.
# Nothing is measured for speed while it runs. The build must be a normal
# one; a -race build is soaked separately (make build-race, then
# NANOLLM_SOAK_BIN=dist/nanollm-race).
set -euo pipefail
cd "$(dirname "$0")/.."

hours=${1:-24}
data=${NANOLLM_SERVER_DATA_DIR:-$HOME/nanollm-data}
out=$data/soak/$(date +%F-%H%M)
mkdir -p "$out"
bin=${NANOLLM_SOAK_BIN:-$PWD/dist/nanollm}
node=${NANOLLM_SOAK_NODE:-1}
port=${NANOLLM_SOAK_PORT:-18090}
cpus=$(python3 - "$node" <<'PY'
import sys
node = sys.argv[1]
text = open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()
print(text)
PY
)

# The queue database lives in the run's own directory, so its growth is the
# run's and not that of every job the data directory has ever seen; the models
# stay where they are.
mkdir -p "$out/data"
webhook_secret=soak-secret

cat >"$out/nanollm.yaml" <<YAML
server:
  addr: 127.0.0.1:$port
  data_dir: $out/data
registry:
  models_dir: $data/models
backends:
  lib_dir: $PWD/dist/linux_amd64/lib
devices:
  default: cpu
  cpu_threads: 8          # the logical CPUs of one node
  numa: node:$node
models:
  default: gemma-4-e4b-qat
  preload: [gemma-4-e4b-qat]
  overrides:
    gemma-4-e4b-qat:
      preset: ram-8g
      slots: 2
queue:
  enabled: true
  max_pending: 50
  concurrency: 2
  webhook_secret: $webhook_secret
  retention_days: 1       # the least there is: the purge runs, nothing is old enough to go in a day
log:
  level: info
  file: $out/server.log
YAML

# A -race build writes each report to $out/race.<pid> and goes on, so a soak
# collects every race it meets instead of stopping at the first.
export GORACE="${GORACE:-log_path=$out/race}"

echo "soak: $hours h, node $node (cpus $cpus), logs in $out"
export NANOLLM_PLACE="node:$node;cpus=$cpus"
systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 -- \
  nice -n 10 "$bin" serve --config "$out/nanollm.yaml" >"$out/serve.out" 2>&1 &
sleep 2
until curl -sf "http://127.0.0.1:$port/readyz" >/dev/null; do
  pgrep -f "serve --config $out/nanollm.yaml" >/dev/null || { echo "soak: the server did not start; see $out/serve.out" >&2; exit 1; }
  sleep 5
done
pid=$(pgrep -f "serve --config $out/nanollm.yaml" | head -1)
echo "soak: server pid $pid is ready"

go run ./tools/loadgen -base "http://127.0.0.1:$port" -workers 4 -duration "${hours}h" \
  -interval 1m -pid "$pid" -out "$out/loadgen.jsonl" -bad-out "$out/bad.jsonl" \
  -hook-listen 127.0.0.1:0 -hook-secret "$webhook_secret" -queue-db "$out/data/nanollm.db" || true
go run ./tools/loadgen -analyze "$out/loadgen.jsonl" | tee "$out/analysis.txt" || true
kill "$pid" 2>/dev/null || true
grep -c "panic\|DATA RACE" "$out/server.log" "$out/serve.out" 2>/dev/null || true
echo "race reports: $(ls "$out"/race.* 2>/dev/null | wc -l)" | tee -a "$out/analysis.txt"
