#!/usr/bin/env bash
# Stage 2 plan task 9: does thinking help on problems that need steps? The other
# suites are saturated (docs/QUALITY-STAGE2.md 4). The generated `reasoning` suite
# (tools/reasoningcases.py: six kinds of problem, ru and en, answers computed) is
# saturated too while the model may write its working (48/48 with thinking off, a mean of
# 400 tokens of working in the answer); `reasoning-direct` asks for the bare answer,
# and that is where a thought can matter. Four runs of its 48 items: thinking off, and
# on with a cap on the thought of 512 and of 2048 tokens and with none. Accuracy, tokens
# and time per run go to ~/nanollm-data/eval/reasoning-direct-*.json.
#
#   NODE=1 scripts/reasoning-eval.sh        # one NUMA node, 8 threads, nice 10
set -uo pipefail
cd "$(dirname "$0")/.."
D=${NANOLLM_DATA:-$HOME/nanollm-data}
BIN=${NANOLLM:-./dist/nanollm}
NODE=${NODE:-1}
CPUS=$(cat /sys/devices/system/node/node$NODE/cpulist)
mkdir -p "$D/eval"
CFG=$D/eval/reasoning.yaml
cat >"$CFG" <<YAML
server:
  data_dir: $D
backends:
  lib_dir: $PWD/dist/linux_amd64/lib
devices:
  default: cpu
  cpu_threads: 8
  numa: node:$NODE
models:
  default: gemma-4-e4b-qat
  overrides:
    gemma-4-e4b-qat:
      preset: ram-8g
      kv_type: f16
generation:
  enable_thinking: false
YAML
export NANOLLM_PLACE="node:$NODE;cpus=$CPUS"

run() { # run NAME ARGS...
	local name=$1
	shift
	local out=$D/eval/reasoning-direct-$name.json
	[[ -f $out ]] && { echo "=== $name: done already"; return; }
	echo "=== $(date +%T) $name"
	nice -n 10 "$BIN" eval --config "$CFG" --suite reasoning-direct --out "$out" "$@" 2>&1 | tail -4
}

run off --thinking off --max-tokens 400
run on-cap512 --thinking on --max-reasoning 512 --max-tokens 3000
run on-cap2048 --thinking on --max-reasoning 2048 --max-tokens 5000
run on-nocap --thinking on --max-tokens 8000
echo "=== $(date +%T) done"
