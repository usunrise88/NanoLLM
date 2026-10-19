#!/usr/bin/env bash
# Stage 2.4: the reference logits and the noise pairs behind the KLD threshold
# (docs/STAGE2.md). The reference is CPU haswell, KV cache f16, no MTP, repack
# on, 16 threads, -c 512, upstream's llama-perplexity of the pinned tag. Each
# pair below differs from it in one thing that must not change the answer; the
# KLD of those pairs is the noise, and a threshold is a multiple of it.
#
#   scripts/kld-noise.sh [CHUNKS]      # default 64 chunks of 512 tokens
#
# The reference logits (~0.5 MB per token) go to ~/nanollm-data/kld/ and stay
# out of git; the reports go to bench/results/<host>/. Each run is a transient
# scope that reclaims its own page cache (MemoryHigh), so the big logits file
# does not push the neighbours' cache out.
set -euo pipefail
cd "$(dirname "$0")/.."

CHUNKS=${1:-64}
DATA=${NANOLLM_DATA:-$HOME/nanollm-data}
MODEL=${NANOLLM_KLD_MODEL:-$(ls "$DATA"/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)}
CORPUS=.cache/kld/corpus.txt
UP=dist/linux_amd64-upstream/tools
OWN=dist/linux_amd64-patched/tools
X64=dist/tools-x64
BASE=$DATA/kld/base-f16-haswell-t16-c$CHUNKS.bin
OUT=bench/results/$(hostname -s)
DATE=$(date +%F)
OPT=$(mktemp -d)/optimize
go build -o "$OPT" ./tools/optimize
mkdir -p "$OUT" "$DATA/kld"

scoped() { systemd-run --user --scope --quiet -p MemoryHigh=16G -p MemoryMax=28G -p MemorySwapMax=0 nice -n 10 "$@"; }

if [[ ! -f "$BASE" ]]; then
	echo "=== reference"
	scoped "$OPT" base -tools "$UP" -model "$MODEL" -corpus "$CORPUS" -logits "$BASE" -- -t 16 -fa on --chunks "$CHUNKS"
fi

cand() { # cand NAME TOOLS FLAGS...
	local name=$1 tools=$2
	shift 2
	[[ -f "$OUT/$DATE-kld-$name.json" ]] && return 0
	echo "=== $name"
	scoped "$OPT" kld -tools "$tools" -model "$MODEL" -corpus "$CORPUS" -logits "$BASE" -name "$name" \
		-out "$OUT/$DATE-kld-$name.json" -- -fa on --chunks "$CHUNKS" "$@"
}

# Pairs that must agree: noise.
cand self-t16 "$UP" -t 16
cand threads4 "$UP" -t 4
cand variant-x64 "$X64" -t 16
cand own-build "$OWN" -t 16
# Candidates of the stage: what we want to know the cost of.
cand no-repack "$UP" -t 16 --no-repack
cand kv-q8_0 "$UP" -t 16 -ctk q8_0 -ctv q8_0
echo "=== done"
