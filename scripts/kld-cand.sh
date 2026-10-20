#!/usr/bin/env bash
# One KLD candidate against the reference logits of scripts/kld-noise.sh
# (docs/STAGE2.md 2.4): the same corpus, -c 512, KV cache f16, 16 threads.
#
#   scripts/kld-cand.sh NAME TOOLS_DIR [CHUNKS] [-- llama-perplexity flags...]
#
# CHUNKS (default 64) must not exceed the reference's; fewer chunks read the
# first ones of the file. The report goes to bench/results/<host>/<date>-kld-NAME.json.
# Run it under scripts/hyper-exclusive.sh when the load matters; the numbers do
# not depend on it.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1 TOOLS=$2
shift 2
CHUNKS=64
[[ ${1:-} != "--" && -n ${1:-} ]] && { CHUNKS=$1; shift; }
[[ ${1:-} == "--" ]] && shift
DATA=${NANOLLM_DATA:-$HOME/nanollm-data}
MODEL=${NANOLLM_KLD_MODEL:-$(ls "$DATA"/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)}
BASE=$DATA/kld/base-f16-haswell-t16-c64.bin
OUT=bench/results/$(hostname -s)
mkdir -p "$OUT"
OPT=$(mktemp -d)/optimize
go build -o "$OPT" ./tools/optimize
[[ -f "$OUT/$(date +%F)-kld-$NAME.json" ]] && { echo "kld-cand: $OUT/$(date +%F)-kld-$NAME.json exists"; exit 0; }
systemd-run --user --scope --quiet -p MemoryHigh=16G -p MemoryMax=28G -p MemorySwapMax=0 nice -n 10 \
	"$OPT" kld -tools "$TOOLS" -model "$MODEL" -corpus .cache/kld/corpus.txt -logits "$BASE" -name "$NAME" \
	-out "$OUT/$(date +%F)-kld-$NAME.json" -- -fa on --chunks "$CHUNKS" -t 16 "$@"
