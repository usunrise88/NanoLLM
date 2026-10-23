#!/usr/bin/env bash
# KLD of encodings for the Neural Engine, against the CPU reference (docs/AUTOTUNE.md, stage 1b).
#
#   tools/coreml/quality_suite.sh [VARIANT...]      # default: control pal4 pal6 i8c mix
#
# For each variant: a carrier GGUF (tools/coreml/reencode.py) in ~/nanollm-data, llama-perplexity on the CPU against
# the reference logits (tools/optimize kld; same threads and flags as the reference, so what differs is the weights),
# a report in bench/results/macbookair-m1/, the carrier deleted. One after another: a carrier is 6-7 GiB.
# While it runs, .cache/coreml/busy exists and tools/coreml/energy_suite.sh refuses to start (its numbers would
# be spoiled); the reference is ~/nanollm-data/kld/base-cpu-m1-t4-c8.bin (docs/AUTOTUNE.md, "Metal на Apple M1").
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
py=.cache/coreml-venv/bin/python
data=$HOME/nanollm-data
base=$data/kld/base-cpu-m1-t4-c8.bin
out=bench/results/macbookair-m1
src=$(ls "$HOME"/Library/Application\ Support/NanoLLM/models/gemma-4-e4b-qat/*/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf | head -1)
date=$(date +%F)
mkdir -p .cache/bin
[[ -f $base ]] || { echo "no reference logits: $base" >&2; exit 1; }
[[ -x .cache/bin/optimize ]] || go build -o .cache/bin/optimize ./tools/optimize
touch .cache/coreml/busy
trap 'rm -f .cache/coreml/busy' EXIT

flags() { # the encodings of a variant: attn ffn ple head
	case $1 in
	control) echo "--attn ctl --ffn ctl --ple ctl --head tied" ;;
	pal4) echo "--attn pal4 --ffn pal4 --ple i8c --head pal4" ;;
	pal6) echo "--attn pal6 --ffn pal6 --ple i8c --head pal6" ;;
	i8c) echo "--attn i8c --ffn i8c --ple i8c --head i8c" ;;
	mix) echo "--attn i8c --ffn pal6 --ple i8c --head i8c" ;;
	ffn-pal4) echo "--attn ctl --ffn pal4 --ple ctl --head tied" ;;
	attn-pal4) echo "--attn pal4 --ffn ctl --ple ctl --head tied" ;;
	head-pal4) echo "--attn ctl --ffn ctl --ple ctl --head pal4" ;;
	ffn-pal6) echo "--attn ctl --ffn pal6 --ple ctl --head tied" ;;
	head-pal6) echo "--attn ctl --ffn ctl --ple ctl --head pal6" ;;
	*) echo "unknown variant $1" >&2; return 1 ;;
	esac
}

variants=("$@")
[[ ${#variants[@]} -gt 0 ]] || variants=(control pal4 pal6 i8c mix)
for v in "${variants[@]}"; do
	[[ -f $out/$date-kld-carrier-$v.json ]] && { echo "== $v: done already"; continue; }
	f=$(flags "$v")
	carrier=$data/carrier-$v.gguf
	echo "== $v: $f"
	# shellcheck disable=SC2086
	"$py" -I tools/coreml/reencode.py "$src" "$carrier" $f
	.cache/bin/optimize kld -tools dist/darwin_arm64/tools -model "$carrier" -corpus .cache/kld/corpus.txt -logits "$base" \
		-name "carrier-$v" -out "$out/$date-kld-carrier-$v.json" -- -fa on --chunks 8 -t 4 -ngl 0
	rm -f "$carrier"
done
echo "done"
