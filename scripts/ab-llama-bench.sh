#!/usr/bin/env bash
# Interleaved A/B(/C...) llama-bench of several library sets on the same model, raw output appended to a file.
#
#   scripts/hyper-exclusive.sh -- scripts/ab-llama-bench.sh OUTFILE MODE ROUNDS PROFILES NAME=DIR[@VAR=val,VAR=val] ...
#
#   MODE      decode (-p 0 -n 128) | prefill (-p 512 -n 0) | full (-p 512 -n 128); every run is -r 5, -fa 1
#   PROFILES  strong = 16 threads, numactl --interleave=all;  weak = 4 cores of node 0 (cpus 0-3), memory of node 0.
#             "weak,strong" runs both.  Both are the profiles of docs/AUTOTUNE.md.
#   NAME=DIR  a tools directory (llama-bench with its libraries), e.g. dist/linux_amd64/tools; the first one is the
#             baseline. @VAR=val sets environment variables for that run (kernel switches of a dev build).
#
# Each round runs every variant once per profile, in the order given, so slow drift of the host hits all of them.
# Take the lock with scripts/hyper-exclusive.sh: a throughput number measured next to another job is noise.
# The model is the E4B QAT Q4_K_XL of ~/nanollm-data (NANOLLM_BENCH_MODEL to change it).
set -uo pipefail
[[ $# -ge 5 ]] || { sed -n 2,14p "$0"; exit 2; }
OUT=$1; MODE=$2; ROUNDS=$3; PROFILES=$4; shift 4
M=${NANOLLM_BENCH_MODEL:-$(ls "$HOME"/nanollm-data/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)}
case $MODE in
  decode)  ARGS="-p 0 -n 128" ;;
  prefill) ARGS="-p 512 -n 0" ;;
  full)    ARGS="-p 512 -n 128" ;;
  *) echo "ab-llama-bench: unknown mode $MODE" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$OUT")"
{
  echo "# $(date -Is) mode=$MODE rounds=$ROUNDS profiles=$PROFILES variants: $*"
  echo "# host: $(hostname) $(uname -r), load at start: $(cut -d' ' -f1-3 /proc/loadavg), model: $M"
} >> "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for P in ${PROFILES//,/ }; do
    for nd in "$@"; do
      N=${nd%%=*}; REST=${nd#*=}; D=${REST%%@*}; E=""
      [[ $REST != "$D" ]] && E=${REST#*@}
      EV=(); [[ -n $E ]] && IFS=, read -ra EV <<< "$E"
      echo "### variant=$N profile=$P mode=$MODE round=$r env=${E:-none} load=$(cut -d' ' -f1 /proc/loadavg) $(date -Is)" >> "$OUT"
      if [[ $P == strong ]]; then
        env LD_LIBRARY_PATH="$D" "${EV[@]}" numactl --interleave=all "$D/llama-bench" -m "$M" -t 16 -fa 1 $ARGS -r 5 >> "$OUT" 2>&1
      else
        env LD_LIBRARY_PATH="$D" "${EV[@]}" numactl --physcpubind=0-3 --membind=0 "$D/llama-bench" -m "$M" -t 4 -fa 1 $ARGS -r 5 >> "$OUT" 2>&1
      fi
      echo "### end load=$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT"
    done
  done
done
