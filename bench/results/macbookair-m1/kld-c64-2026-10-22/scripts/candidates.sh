#!/bin/bash
# KLD of Metal configurations against the 64-chunk CPU reference, one after another (8 GiB: one model at a time).
#
#   NANOLLM_ROOT=$PWD bench/results/macbookair-m1/kld-c64-2026-10-22/scripts/candidates.sh
#
# The reference (8.6 GB): go run ./tools/optimize base -tools dist/darwin_arm64/tools -model "$M" \
#   -corpus .cache/kld/corpus.txt -logits $BASE -- -fa on --chunks 64 -t 4 -dev none -ngl 0
# fa_small.dylib: clang -dynamiclib -o scripts/fa_small.dylib scripts/fa_small.c -Ldist/darwin_arm64/tools -lllama \
#   -Wl,-rpath,@executable_path (it goes only into llama-perplexity, through the wrapper next to it: in go or in
#   tools/optimize it finds no libllama and aborts).
set -u
cd "${NANOLLM_ROOT:?set NANOLLM_ROOT to the repository}"
M="$HOME/Library/Application Support/NanoLLM/models/gemma-4-e4b-qat/8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
BASE=$HOME/nanollm-data/kld/base-cpu-m1-t4-c64.bin
R=bench/results/macbookair-m1/kld-c64-2026-10-22
LOGS=${LOGS:-.cache/kld}
run() { # name, tools dir, llama-perplexity flags...
  local name=$1 tools=$2; shift 2
  echo "== $name $(date +%T) memory level $(sysctl -n kern.memorystatus_level)"
  go run ./tools/optimize kld -tools "$tools" -model "$M" -corpus .cache/kld/corpus.txt -logits "$BASE" \
    -name "$name" -out "$R/kld-$name.json" -- "$@" > "$LOGS/$name.log" 2>&1
  echo "exit $? $(date +%T)"
  grep -E "fa_small:" "$LOGS/$name.log" | head -1
  grep -E "kld_mean|kld_p99|top1" "$R/kld-$name.json"
}
run metal-c64 dist/darwin_arm64/tools -fa on --chunks 64 -ngl 99
run metal-fa-small-c64 "$R/scripts" -fa on --chunks 64 -ngl 99
run metal-fa-off-c64 dist/darwin_arm64/tools -fa off --chunks 64 -ngl 99
