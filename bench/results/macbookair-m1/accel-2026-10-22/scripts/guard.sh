#!/bin/bash
# Runs a command at nominal thermal pressure + 30 s, kills it if available memory drops below 15 %; prints the minimum.
lvl() { notifyutil -g com.apple.system.thermalpressurelevel | awk '{print $2}'; }
t0=$(date +%s); while [ "$(lvl)" != "0" ] && [ $(( $(date +%s) - t0 )) -lt 420 ]; do sleep 5; done; sleep 30
[ "$(sysctl -n kern.memorystatus_level)" -ge 35 ] || { echo "low memory, not started"; exit 3; }
"$@" & pid=$!; min=100
while kill -0 $pid 2>/dev/null; do l=$(sysctl -n kern.memorystatus_level); [ "$l" -lt "$min" ] && min=$l
  if [ "$l" -lt 15 ]; then kill -9 $pid; echo "KILLED: memory level $l"; break; fi; sleep 1; done
wait $pid 2>/dev/null; echo "exit $? min_mem=$min level_end=$(lvl)"
