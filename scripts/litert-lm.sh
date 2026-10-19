#!/usr/bin/env bash
# Stage 2.5: LiteRT-LM (Google's on-device runtime, XNNPACK on the CPU) on the
# same machine and the same prompt sizes as ours, Gemma 4 E4B. The model is
# Google's own .litertlm packaging of E4B (int4), not our GGUF; the weights are
# the same model family and quantization class but not the same file, so this
# compares runtimes, not bit-identical quantizations.
#
#   scripts/litert-lm.sh install            # uv tool install litert-lm (user level)
#   scripts/litert-lm.sh fetch              # the E4B CPU model, 3.7 GB, into ~/nanollm-data/models/litert/
#   scripts/litert-lm.sh bench [THREADS...] # pp512 / tg128, with litert-lm benchmark
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
DIR=${NANOLLM_DATA:-$HOME/nanollm-data}/models/litert
MODEL=$DIR/gemma-4-E4B-it.litertlm
REPO=litert-community/gemma-4-E4B-it-litert-lm

case "${1:-}" in
install) uv tool install litert-lm ;;
fetch)
	mkdir -p "$DIR"
	[[ -f $MODEL ]] || curl -fL --retry 3 -o "$MODEL.part" "https://huggingface.co/$REPO/resolve/main/gemma-4-E4B-it.litertlm" && mv "$MODEL.part" "$MODEL"
	ls -l "$MODEL"
	;;
bench)
	shift
	for t in "${@:-16}"; do
		for spec in false true; do # speculative decoding (MTP) off and on
			echo "=== litert-lm, $t threads, speculative=$spec, pp512 tg128"
			nice -n 5 numactl --interleave=all litert-lm benchmark "$MODEL" -p 512 -d 128 --runs 3 \
				--backend cpu --cpu-thread-count "$t" --speculative-decoding "$spec" 2>&1 | tail -25
		done
	done
	;;
*) sed -n 2,11p "$0"; exit 2 ;;
esac
