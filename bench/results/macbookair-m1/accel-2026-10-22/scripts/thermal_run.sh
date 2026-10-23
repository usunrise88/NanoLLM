#!/bin/bash
# Wait for nominal thermal pressure, then run one prefill config many times while logging thermal pressure.
set -u
S=$1; gguf=$2
lvl() { notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $2}'; }
t0=$(date +%s)
while [ "$(lvl)" != "0" ] && [ $(( $(date +%s) - t0 )) -lt 600 ]; do sleep 5; done
echo "cooled after $(( $(date +%s) - t0 )) s, level $(lvl), mem $(sysctl -n kern.memorystatus_level)%"
( start=$(date +%s); while :; do echo "$(( $(date +%s) - start )) $(lvl)"; sleep 2; done ) > $S/thermal.log &
mon=$!
dist/darwin_arm64/tools/llama-bench -m "$gguf" -ngl 99 -fa 0 -ub 256 -b 2048 -p 4096 -n 0 -r 12 -o json > $S/sustained.json 2> $S/sustained.err
kill $mon
echo "bench exit, final level $(lvl)"
