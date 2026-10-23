#!/usr/bin/env bash
# Stage 2 plan task 4: the KLD noise and the KV cache q8_0 at a context of 4096
# tokens. scripts/kld-noise.sh did this at 512; the threshold of
# docs/QUALITY-STAGE2.md (2.2) is three noises of that run, and a KV cache of 8
# bits can cost more where the context is long, so the noise is measured again at
# 4096 (self-t16 is a repeat of the reference, threads4 and variant-x64 pairs
# that must agree) and the candidates are held to it. Each chunk scores the second
# half of 4096 tokens, 2048 positions with up to 4095 tokens before them.
#
#   scripts/kld-4096.sh [CHUNKS]      # default 16 chunks of 4096 tokens (the corpus has 14)
#
# The reference logits (~0.5 MB per token) go to ~/nanollm-data/kld/ and stay
# out of git; the reports go to bench/results/<host>/. Each run is a transient
# scope that reclaims its own page cache (MemoryHigh), so the big logits file
# does not push the neighbours' cache out.
set -euo pipefail
cd "$(dirname "$0")/.."

CHUNKS=${1:-16}
CTX=4096
T=${KLD_THREADS:-16}   # the logits do not depend on it (threads4 against 16 is the noise); KLD_PIN=cpulist pins the runs
DATA=${NANOLLM_DATA:-$HOME/nanollm-data}
MODEL=${NANOLLM_KLD_MODEL:-$(ls "$DATA"/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)}
CORPUS=.cache/kld/corpus.txt
UP=dist/linux_amd64-upstream/tools
X64=dist/tools-x64
BASE=$DATA/kld/base-f16-haswell-t16-ctx$CTX-c$CHUNKS.bin
OUT=bench/results/$(hostname -s)
DATE=$(date +%F)
OPT=$(mktemp -d)/optimize
go build -o "$OPT" ./tools/optimize
mkdir -p "$OUT" "$DATA/kld"

scoped() { systemd-run --user --scope --quiet -p MemoryHigh=16G -p MemoryMax=28G -p MemorySwapMax=0 nice -n 10 ${KLD_PIN:+taskset -c "$KLD_PIN"} "$@"; }

if [[ ! -f "$BASE" ]]; then
	echo "=== reference"
	scoped "$OPT" base -tools "$UP" -model "$MODEL" -corpus "$CORPUS" -logits "$BASE" -ctx "$CTX" -- -t "$T" -fa on --chunks "$CHUNKS"
fi

cand() { # cand NAME TOOLS FLAGS...
	local name=$1 tools=$2
	shift 2
	[[ -f "$OUT/$DATE-kld$CTX-$name.json" ]] && return 0
	echo "=== $name"
	scoped "$OPT" kld -tools "$tools" -model "$MODEL" -corpus "$CORPUS" -logits "$BASE" -name "$name" -ctx "$CTX" \
		-out "$OUT/$DATE-kld$CTX-$name.json" -- -fa on --chunks "$CHUNKS" "$@"
}

# The pair that must agree, whose KLD is the noise: another set of kernels (the repeat
# of the reference and another thread count give exactly zero, 2.2 of QUALITY-STAGE2).
cand variant-x64 "$X64" -t "$T"
# Candidates of the stage: what we want to know the cost of. At 4096 tokens a run takes
# an hour or two, so the one the question is about comes first.
cand kv-q8_0 "$UP" -t "$T" -ctk q8_0 -ctv q8_0
cand no-repack "$UP" -t "$T" --no-repack
echo "=== done"
