#!/usr/bin/env bash
# Energy suite of stage 1a (docs/AUTOTUNE.md, "Apple Neural Engine: правила гейта"): idle, one token of the pal4 model
# on the Neural Engine, and the real model decoding on Metal, each through tools/coreml/energy.sh.
#
#   sudo -v                                  # in this terminal, first
#   tools/coreml/energy_suite.sh
#
# Needs the seven body chunks .cache/coreml/c0..c6.mlpackage and the four head parts head0..3.mlpackage
# (tools/coreml/probe_chunk.py, PROBE_SEED=100+i and PROBE_KIND=head PROBE_PART=i), .cache/coreml/predict_bench,
# the pinned llama-bench (scripts/fetch-libs.sh --tools) and the pulled model. Raw samples and outputs go to
# bench/results/macbookair-m1/ane-probe-1a-2026-10-22/energy/. One run after another, never together, with a
# pause to cool down; it refuses to start a load when the system has little memory left.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$root/bench/results/macbookair-m1/ane-probe-1a-2026-10-22/energy
mkdir -p "$out"
cd "$out"

models=""
for i in 0 1 2 3 4 5 6; do models="$models${models:+,}$root/.cache/coreml/c$i.mlpackage"; done
for i in 0 1 2 3; do models="$models,$root/.cache/coreml/head$i.mlpackage"; done
gguf=$(ls "$HOME"/Library/Application\ Support/NanoLLM/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)

[[ ! -e $root/.cache/coreml/busy ]] || { echo "tools/coreml/quality_suite.sh is running: its load would spoil the measurement; wait for it to finish" >&2; exit 4; }
memory_ok() {
	local level
	level=$(sysctl -n kern.memorystatus_level)
	[[ $level -ge 35 ]] || { echo "only $level% of memory is available; close something first" >&2; exit 3; }
}
cool() { echo "cooling down 25 s"; sleep 25; }

memory_ok
echo "== idle, 30 s"
"$root/tools/coreml/energy.sh" idle --skip 0 -- sleep 30
cool

memory_ok
echo "== ANE: 7 body chunks + 4 head parts (pal4), 700 tokens"
"$root/tools/coreml/energy.sh" ane-pal4 --ready "$out/ready" -- "$root/.cache/coreml/predict_bench" "$models" ane 700
cool

memory_ok
echo "== Metal: llama-bench tg128 x6, the real model, all layers on the GPU"
"$root/tools/coreml/energy.sh" metal-tg --skip 8 -- "$root/dist/darwin_arm64/tools/llama-bench" -m "$gguf" -ngl 99 -fa 1 -p 0 -n 128 -r 6 -o json
echo "done: $out"
