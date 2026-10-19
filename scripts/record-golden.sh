#!/usr/bin/env bash
# Record the golden files of the machine this runs on (docs/STAGE2.md 2.2):
# greedy output of the pinned tag's llama-server for the E4B, with and without
# the MTP drafter. The model is the one Q4_0 QAT file and the KV cache is f16:
# there are no other variants to record. The ggml CPU variant of the machine
# is in each file's name, so a second machine adds files.
#
#   scripts/fetch-libs.sh --tools          # or scripts/build-libs.sh
#   scripts/record-golden.sh [MODEL_DIR]   # default: the E4B under ~/nanollm-data
#
# Run it on an idle host: llama-server loads the model twice.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/go/bin:$HOME/go/bin:$PATH"

dir=${1:-$(ls -d "$HOME"/nanollm-data/models/gemma-4-e4b-qat/*/ | head -1)}
model=$(ls "$dir"/gemma-4-E4B-it-qat-*.gguf | head -1)
drafter=$(ls "$dir"/mtp-*.gguf | head -1)
tools=${NANOLLM_GOLDEN_TOOLS:-dist/linux_amd64/tools}
out=testdata/golden/$(tr -d '[:space:]' < scripts/llama.cpp.tag)/gemma-4-e4b-qat
mkdir -p "$out"

# The variant is not known until llama-server has started, so record to a
# scratch name and let the tool say what the name should be.
for mode in plain mtp; do
  tmp=$out/.new-$mode.json
  args=(-tools "$tools" -model "$model" -threads 4 -out "$tmp")
  if [[ $mode == mtp ]]; then
    args+=(-drafter "$drafter" -draft-max 3 -draft-p-min 0 -n 64)
  else
    args+=(-n 32)
  fi
  go run ./tools/golden "${args[@]}"
  variant=$(python3 -c "import json; print(json.load(open('$tmp'))['cpu_variant'])")
  name=cpu-
  [[ $mode == mtp ]] && name+=mtp-
  mv "$tmp" "$out/$name$variant.json"
  echo "recorded $out/$name$variant.json"
done
