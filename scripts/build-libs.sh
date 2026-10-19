#!/usr/bin/env bash
# Build NanoLLM's llama.cpp libraries and tools: the pinned upstream tag plus
# the patch series in patches/llama.cpp (docs/STAGE2.md 2.1).
#
#   scripts/build-libs.sh [--no-vulkan] [--jobs N] [--cpus N] [--src DIR] [--prepare-only] [--no-patches]
#
# Writes dist/linux_amd64/{lib,tools} in the layout scripts/fetch-libs.sh
# --tools produces, so everything downstream (the golden tests, tools/optimize,
# the native tests, make build) takes them without a change. Only linux_amd64
# is built here; the other targets are built by .github/workflows/libs-build.yml
# with the same flags.
#
# The source is the upstream tag in scripts/llama.cpp.base with every
# patches/llama.cpp/*.patch applied as a commit (git am, fixed committer and
# dates), so the commit hash is the same on every machine. That hash is
# scripts/llama.cpp.commit, the one ggml_commit() reports and the binding
# refuses a library without. With no patches it is the upstream commit and the
# build is upstream's: the flags below are those of its ubuntu-vulkan release
# job.
#
# The build runs in a container (scripts/build-libs.Dockerfile) so that it
# does not depend on the host's compiler, and under --cpus so that it leaves
# a shared host usable.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD

VULKAN=1
JOBS=$(nproc)
CPUS=""
SRC=$ROOT/.cache/llama.cpp/src
PREPARE_ONLY=0
PATCHES=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-vulkan) VULKAN=0 ;;
    --jobs) JOBS=$2; shift ;;
    --cpus) CPUS=$2; shift ;;
    --src) SRC=$2; shift ;;
    --prepare-only) PREPARE_ONLY=1 ;;
    --no-patches) PATCHES=0 ;; # upstream's own build, to compare a patched one with
    *) echo "build-libs: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

TAG=$(tr -d '[:space:]' < scripts/llama.cpp.tag)
BASE=$(tr -d '[:space:]' < scripts/llama.cpp.base)
WANT=$(tr -d '[:space:]' < scripts/llama.cpp.commit)

# --- the source: upstream's commit with the patches on top -------------------

if [[ ! -d $SRC/.git ]]; then
  mkdir -p "$(dirname "$SRC")"
  git clone --quiet https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" fetch --quiet origin "refs/tags/$TAG:refs/tags/$TAG" 2>/dev/null || true
git -C "$SRC" checkout --quiet --force --detach "$BASE"
git -C "$SRC" clean --quiet -fdx
if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$BASE" ]]; then
  echo "build-libs: $SRC is not at $BASE" >&2
  exit 1
fi

export GIT_COMMITTER_NAME=NanoLLM GIT_COMMITTER_EMAIL=nanollm@localhost
shopt -s nullglob
[[ $PATCHES -eq 0 ]] && WANT=$BASE
for p in $([[ $PATCHES -eq 1 ]] && echo "$ROOT"/patches/llama.cpp/*.patch); do
  # The committer date follows the patch's author date: no clock in the hash.
  git -C "$SRC" am --quiet --committer-date-is-author-date "$p" \
    || { echo "build-libs: $p does not apply to $BASE" >&2; git -C "$SRC" am --abort || true; exit 1; }
done
GOT=$(git -C "$SRC" rev-parse HEAD)
if [[ "$GOT" != "$WANT" ]]; then
  echo "build-libs: the patched source is $GOT, but scripts/llama.cpp.commit says $WANT." >&2
  echo "build-libs: if the series changed on purpose, write the new hash there." >&2
  exit 1
fi
echo "source: $TAG + $([[ $PATCHES -eq 1 ]] && ls "$ROOT"/patches/llama.cpp/*.patch 2>/dev/null | wc -l | tr -d ' ' || echo 0) patches = ${GOT:0:12}"
[[ $PREPARE_ONLY -eq 1 ]] && exit 0

# --- the build ---------------------------------------------------------------

IMAGE=nanollm-build-libs:ubuntu22.04
[[ $VULKAN -eq 0 ]] && IMAGE=nanollm-build-libs:ubuntu22.04-novulkan
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -q -t "$IMAGE" --build-arg VULKAN=$VULKAN -f scripts/build-libs.Dockerfile scripts >/dev/null
fi

BUILD=$ROOT/.cache/llama.cpp/build-linux_amd64$([[ $PATCHES -eq 0 ]] && echo -upstream)
mkdir -p "$BUILD" "$ROOT/.cache/ccache"
CMAKE_VULKAN=()
[[ $VULKAN -eq 1 ]] && CMAKE_VULKAN=(-DGGML_VULKAN=ON)
DOCKER_CPUS=()
[[ -n $CPUS ]] && DOCKER_CPUS=(--cpus "$CPUS")

# The flags of upstream's ubuntu-vulkan job, with its CMAKE_ARGS (tests and
# examples off, tools and server and the RPC backend on).
docker run --rm "${DOCKER_CPUS[@]}" -u "$(id -u):$(id -g)" \
  -v "$SRC:/src" -v "$BUILD:/build" -v "$ROOT/.cache/ccache:/ccache" \
  -e CCACHE_DIR=/ccache -e HOME=/tmp \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/src \
  "$IMAGE" bash -euo pipefail -c "
    cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_INSTALL_RPATH='\$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
      -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON ${CMAKE_VULKAN[*]} \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DGGML_RPC=ON
    cmake --build /build --config Release -j $JOBS
  "

# --- the layout fetch-libs.sh --tools makes -----------------------------------

OUT=$ROOT/dist/linux_amd64$([[ $PATCHES -eq 0 ]] && echo -upstream)
rm -rf "$OUT/lib" "$OUT/tools"
mkdir -p "$OUT/lib" "$OUT/tools"
find "$BUILD/bin" -maxdepth 1 \( -type f -o -type l \) -exec cp -P {} "$OUT/tools/" \;
find "$BUILD/bin" -maxdepth 1 \( -type f -o -type l \) \( -name 'libllama.*' -o -name 'libggml*' \) -exec cp -P {} "$OUT/lib/" \;
cp "$SRC/LICENSE" "$OUT/tools/" 2>/dev/null || true
echo "${OUT#$ROOT/}: $(ls "$OUT/lib" | wc -l | tr -d ' ') libraries, $(ls "$OUT/tools" | wc -l | tr -d ' ') files in tools"
