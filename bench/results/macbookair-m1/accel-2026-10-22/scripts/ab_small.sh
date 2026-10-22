#!/bin/bash
# small_batch flash attention on Metal: prefill at 4096 and decoding at depth 4096, each run started cool.
set -u
gguf=$1
lvl() { notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $2}'; }
cool() { t0=$(date +%s); while [ "$(lvl)" != "0" ] && [ $(( $(date +%s) - t0 )) -lt 420 ]; do sleep 5; done; sleep 30; }
run() {
  cool
  [ "$(sysctl -n kern.memorystatus_level)" -ge 35 ] || { echo "low memory, stop"; exit 3; }
  out=$(dist/darwin_arm64/tools/llama-bench -m "$gguf" -ngl 99 -b 2048 -r 2 -o json "$@" 2>&1)
  echo "$out" | python3 -I -c 'import json,sys
t=sys.stdin.read(); i=t.find("[")
try:
    for d in json.loads(t[i:]): print(" ".join(sys.argv[1:]), "-> %.1f" % d["avg_ts"], " ".join("%.1f"%x for x in d["samples_ts"]))
except Exception as e: print(" ".join(sys.argv[1:]), "FAILED", t[-300:].replace("\n"," | "))' "$@"
}
run -fa small_batch -ub 256 -p 4096 -n 0
run -fa small_batch -ub 512 -p 4096 -n 0
run -fa 1 -ub 256 -p 0 -n 64 -d 4096
run -fa 0 -ub 256 -p 0 -n 64 -d 4096
run -fa small_batch -ub 256 -p 0 -n 64 -d 4096
