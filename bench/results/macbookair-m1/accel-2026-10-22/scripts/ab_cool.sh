#!/bin/bash
# One prefill config at a time, each started at nominal thermal pressure plus 30 s of rest.
set -u
S=$1; gguf=$2
lvl() { notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $2}'; }
for cfg in "1 512" "0 256" "0 512" "1 256" "1 512"; do
  set -- $cfg
  t0=$(date +%s)
  while [ "$(lvl)" != "0" ] && [ $(( $(date +%s) - t0 )) -lt 420 ]; do sleep 5; done
  sleep 30
  [ "$(sysctl -n kern.memorystatus_level)" -ge 35 ] || { echo "low memory, stop"; exit 3; }
  w=$(( $(date +%s) - t0 ))
  r=$(dist/darwin_arm64/tools/llama-bench -m "$gguf" -ngl 99 -fa $1 -ub $2 -b 2048 -p 4096 -n 0 -r 2 -o json 2>/dev/null | python3 -I -c 'import json,sys; d=json.load(sys.stdin)[0]; print("%.1f" % d["avg_ts"], " ".join("%.0f"%x for x in d["samples_ts"]))')
  echo "fa=$1 ub=$2 waited=${w}s level_after=$(lvl) pp4096=$r"
done
