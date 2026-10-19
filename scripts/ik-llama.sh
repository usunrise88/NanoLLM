#!/usr/bin/env bash
# Stage 2.5: ik_llama.cpp on the same model and prompts as ours (the comparison
# the owner kept in the plan; gemma.cpp moved to stage 3).
#
#   scripts/ik-llama.sh build              # clone + build llama-bench/llama-server in Docker (CPU-heavy: run on a quiet machine)
#   scripts/ik-llama.sh bench MODEL [THREADS...]   # pp512/tg128 of ik_llama.cpp and of our pinned upstream, side by side
#
# The build uses -DGGML_NATIVE=ON in a container on this host, so the binary is
# tuned for the CPU it runs on (znver1 here); upstream's release libraries are
# generic variants chosen at run time, which is part of what is compared.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=${IK_REPO:-https://github.com/ikawrakow/ik_llama.cpp}
SRC=.cache/ik_llama.cpp/src
OUT=dist/ik_llama
IMAGE=nanollm-build-libs:ubuntu22.04

build() {
	if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
		docker build -q -t "$IMAGE" -f scripts/build-libs.Dockerfile scripts >/dev/null
	fi
	mkdir -p "$(dirname "$SRC")" "$OUT"
	[[ -d $SRC/.git ]] || git clone --depth 1 "$REPO" "$SRC"
	git -C "$SRC" rev-parse HEAD | tee "$OUT/COMMIT"
	docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e CCACHE_DISABLE=1 -v "$PWD/$SRC:/src" -v "$PWD/$OUT:/out" "$IMAGE" bash -euo pipefail -c '
		cmake -S /src -B /tmp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
			-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON
		cmake --build /tmp/build -j "$(nproc)" --target llama-bench llama-server
		cp -P /tmp/build/bin/llama-bench /tmp/build/bin/llama-server /out/
		find /tmp/build -name "*.so*" -exec cp -P {} /out/ \;
	'
	ls -l "$OUT"
}

bench() {
	local model=$1
	shift
	local threads=("${@:-16}")
	for t in "${threads[@]}"; do
		echo "=== ik_llama.cpp, $t threads"
		LD_LIBRARY_PATH=$OUT nice -n 5 "$OUT/llama-bench" -m "$model" -t "$t" -p 512 -n 128 -r 3 -fa 1 2>&1 | tail -6
		echo "=== upstream $(cat scripts/llama.cpp.tag), $t threads"
		LD_LIBRARY_PATH=dist/linux_amd64/tools nice -n 5 dist/linux_amd64/tools/llama-bench -m "$model" -t "$t" -p 512 -n 128 -r 3 -fa 1 2>&1 | tail -6
	done
}

case "${1:-}" in
build) build ;;
bench) shift; bench "$@" ;;
*) sed -n 2,9p "$0"; exit 2 ;;
esac
