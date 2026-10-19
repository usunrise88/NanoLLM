#!/usr/bin/env bash
# Fast dev build of one ggml target (default: libggml-cpu-haswell) of a llama.cpp checkout, with the flags of
# scripts/build-libs.sh, in the container of that script. Output in .cache/llama.cpp/build-dev/bin.
#
#   scripts/kernel-bench/devbuild.sh [TARGET...]        # LLAMA_SRC=... to use another checkout
#
# For timing a candidate: copy dist/linux_amd64/tools somewhere, replace its libggml-cpu-haswell.so with the one
# built here (see devlayer.py for the run-time switches) and give that directory to scripts/ab-llama-bench.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
SRC=${LLAMA_SRC:-$ROOT/.cache/llama.cpp/src}
BUILD=$ROOT/.cache/llama.cpp/build-dev
IMAGE=nanollm-build-libs:ubuntu22.04
mkdir -p "$BUILD" "$ROOT/.cache/ccache"
TARGETS=("$@")
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(ggml-cpu-haswell)
docker run --rm --cpus 6 -u "$(id -u):$(id -g)" \
  -v "$SRC:/src" -v "$BUILD:/build" -v "$ROOT/.cache/ccache:/ccache" \
  -e CCACHE_DIR=/ccache -e HOME=/tmp \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/src \
  "$IMAGE" bash -euo pipefail -c "
    if [ ! -f /build/build.ninja ]; then
      cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_INSTALL_RPATH='\$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_VULKAN=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON
    fi
    nice -n 10 cmake --build /build --config Release -j 6 --target ${TARGETS[*]}
  "
ls -la "$BUILD/bin/"libggml-cpu-haswell.so
