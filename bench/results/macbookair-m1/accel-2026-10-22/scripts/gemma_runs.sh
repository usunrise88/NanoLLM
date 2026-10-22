#!/bin/bash
# gemma.json through nanollm serve for each named config dir (.cache/gemma-json/NAME), in the order given; each run
# starts at nominal thermal pressure plus 30 s. Prints totals: wall, TTFT sum, decode time (wall - TTFT), tokens.
set -u
lvl() { notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $2}'; }
cool() { t0=$(date +%s); while [ "$(lvl)" != "0" ] && [ $(( $(date +%s) - t0 )) -lt 420 ]; do sleep 5; done; sleep 30; }
done_list=" "
for name in "$@"; do
  done_list="$done_list$name "; k=$(echo "$done_list" | tr " " "\n" | grep -cx "$name")
  D=.cache/gemma-json/$name; out=$D/steps-$k.log
  cool
  [ "$(sysctl -n kern.memorystatus_level)" -ge 35 ] || { echo "low memory, stop"; exit 3; }
  .cache/bin/nanollm serve -config $D/nanollm.yaml -log-file $D/serve-$k.log > $D/serve-$k.out 2>&1 & srv=$!
  for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8099/healthz >/dev/null 2>&1 && break; sleep 1; done
  sleep 8
  python3 -I bench/results/rusunrisebook/baseline-2026-10-20-gemma-igpu-uma4g/run_steps.py http://127.0.0.1:8099 ~/Downloads/gemma.json $D/answers-$k.json > $out 2>&1 &
  py=$!; min=100
  while kill -0 $py 2>/dev/null; do l=$(sysctl -n kern.memorystatus_level); [ "$l" -lt "$min" ] && min=$l
    if [ "$l" -lt 12 ]; then kill -9 $srv; echo "KILLED at level $l" >> $out; break; fi; sleep 3; done
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  python3 -I - "$out" "$name" "$k" "$min" "$(lvl)" <<'PY'
import re, sys
f, name, k, mn, lv = sys.argv[1:]
wall = ttft = toks = 0.0; acc = []
for line in open(f):
    m = re.search(r"out (\d+), ttft (\d+) ms.*accept ([0-9.]+|None), wall ([0-9.]+) s", line)
    if not m: continue
    toks += int(m[1]); ttft += int(m[2]) / 1000; wall += float(m[4])
    if m[3] != "None": acc.append(float(m[3]))
dec = wall - ttft
print(f"{name} round={k} wall={wall:.1f}s ttft={ttft:.1f}s decode={dec:.1f}s out={toks:.0f} "
      f"decode_tok_s={toks/dec:.2f} accept={(sum(acc)/len(acc) if acc else 0):.2f} min_mem={mn} level_end={lv}", flush=True)
PY
done
