#!/usr/bin/env bash
# Fetch the native libraries of one pinned llama.cpp release (SPEC §10, step 1).
#
#   scripts/fetch-libs.sh [--tools] [TAG] [TARGET...]
#
# TAG defaults to the contents of scripts/llama.cpp.tag. TARGET is any of
# linux_amd64 linux_arm64 windows_amd64 windows_arm64 darwin_arm64
# darwin_amd64 darwin_universal, or the CUDA add-ons linux_amd64_cuda and
# windows_amd64_cuda; the default is the machine this runs on.
#
# Each archive is verified against scripts/libs.sha256 before it is opened: an
# archive whose hash is not listed there is refused, not trusted on first use.
# libllama and libggml* are unpacked into dist/<target>/lib/. An add-on target
# gets only ggml-cuda and the CUDA runtime, to unpack over an installation.
#
# --tools also unpacks the whole archive — llama-cli, llama-completion,
# llama-server, llama-tokenize, llama-perplexity and the libraries they link —
# into dist/<target>/tools/. The golden tests, the KL-divergence tooling and
# the MTP reference run take their reference output from the tools of the same
# tag (STAGE1 1.5, 1.10); the release does not ship them.
#
# NanoLLM compiles no C code. If an upstream archive layout changes, this
# script fails on the layout check below rather than shipping a lib/ that is
# missing a backend (SPEC §14, risks).
set -euo pipefail

cd "$(dirname "$0")/.."

MIN_TAG=b9568 # keep in step with llamacpp.MinTag
TOOLS=0
if [[ "${1:-}" == --tools ]]; then
  TOOLS=1
  shift
fi
TAG="${1:-$(tr -d '[:space:]' < scripts/llama.cpp.tag)}"
shift || true

if [[ ! "$TAG" =~ ^b[0-9]+$ ]]; then
  echo "fetch-libs: llama.cpp tag is \"$TAG\"; pin one in scripts/llama.cpp.tag (bNNNN, at least $MIN_TAG)" >&2
  exit 1
fi
if (( ${TAG#b} < ${MIN_TAG#b} )); then
  echo "fetch-libs: $TAG predates Gemma 4 MTP support; the minimum is $MIN_TAG" >&2
  exit 1
fi

# The release assets that carry each target's CPU and GPU backends (SPEC §10,
# release archives), checked against the b11600 release. Upstream renames
# assets from time to time; the sha256 list and the libs workflow catch it.
#
# CUDA 12.x rather than 13.x: the same Ampere-and-newer GPUs, an older minimum
# driver. The Windows runtime archive carries no tag in its name upstream.
assets_for() {
  case "$1" in
    linux_amd64)       echo "llama-$TAG-bin-ubuntu-vulkan-x64.tar.gz" ;;
    linux_arm64)       echo "llama-$TAG-bin-ubuntu-vulkan-arm64.tar.gz" ;;
    windows_amd64)     echo "llama-$TAG-bin-win-vulkan-x64.zip" ;;
    windows_arm64)     echo "llama-$TAG-bin-win-vulkan-arm64.zip" ;;
    darwin_arm64)      echo "llama-$TAG-bin-macos-arm64.tar.gz" ;;
    darwin_amd64)      echo "llama-$TAG-bin-macos-x64.tar.gz" ;;
    linux_amd64_cuda)  echo "llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz cudart-llama-$TAG-bin-ubuntu-cuda-12.8-x64.tar.gz" ;;
    windows_amd64_cuda) echo "llama-$TAG-bin-win-cuda-12.4-x64.zip cudart-llama-bin-win-cuda-12.4-x64.zip" ;;
    *) return 1 ;;
  esac
}

host_target() {
  local os arch
  case "$(uname -s)" in
    Linux) os=linux ;;
    Darwin) os=darwin ;;
    MINGW*|MSYS*|CYGWIN*) os=windows ;;
    *) echo "fetch-libs: unsupported host $(uname -s)" >&2; exit 1 ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64) arch=amd64 ;;
    arm64|aarch64) arch=arm64 ;;
    *) echo "fetch-libs: unsupported architecture $(uname -m)" >&2; exit 1 ;;
  esac
  echo "${os}_${arch}"
}

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

CACHE=.cache/llama.cpp/$TAG
mkdir -p "$CACHE"

# Scratch directories, removed however the script exits.
STAGES=()
trap 'for d in ${STAGES[@]+"${STAGES[@]}"}; do rm -rf "$d"; done' EXIT

# download ASSET: the verified archive in the cache, or exit.
download() {
  local asset=$1 want got
  want=$(awk -v f="$asset" '$2 == f { print $1 }' scripts/libs.sha256)
  if [[ -z "$want" ]]; then
    echo "fetch-libs: $asset is not listed in scripts/libs.sha256; add its hash from the release before fetching" >&2
    exit 1
  fi
  if [[ ! -f "$CACHE/$asset" ]]; then
    echo "fetching $asset"
    curl -fL --retry 3 -o "$CACHE/$asset.part" \
      "https://github.com/ggml-org/llama.cpp/releases/download/$TAG/$asset"
    mv "$CACHE/$asset.part" "$CACHE/$asset"
  fi
  got=$(sha256 "$CACHE/$asset")
  if [[ "$got" != "$want" ]]; then
    echo "fetch-libs: $asset has sha256 $got, expected $want" >&2
    rm -f "$CACHE/$asset"
    exit 1
  fi
}

unpack() {
  case "$1" in
    *.zip)    unzip -q "$1" -d "$2" ;;
    *.tar.gz) tar -xzf "$1" -C "$2" ;;
    *) echo "fetch-libs: unknown archive type $1" >&2; exit 1 ;;
  esac
}

# copy STAGE OUT PATTERN...: files and symlinks whose name matches, flattened
# into OUT. Symlinks are kept as links: libllama.so resolves libggml.so.0 by
# its soname, and the chain must survive the copy.
copy_matching() {
  local stage=$1 out=$2 expr=() p
  shift 2
  for p in "$@"; do expr+=(-o -name "$p"); done
  find "$stage" \( -type f -o -type l \) \( "${expr[@]:1}" \) -exec cp -P {} "$out/" \;
}

fetch() {
  local target=$1 assets asset out stage
  assets=$(assets_for "$target") || { echo "fetch-libs: unknown target $target" >&2; exit 1; }

  stage=$(mktemp -d)
  STAGES+=("$stage")
  for asset in $assets; do
    download "$asset"
    unpack "$CACHE/$asset" "$stage"
  done

  out=dist/$target/lib
  rm -rf "$out"
  mkdir -p "$out"

  if [[ "$target" == *_cuda ]]; then
    # The add-on: the CUDA backend and the runtime it links, nothing that the
    # base archive already installs.
    copy_matching "$stage" "$out" 'libggml-cuda*' 'ggml-cuda*.dll' \
      'libcudart*' 'libcublas*' 'cudart64*.dll' 'cublas64*.dll' 'cublasLt64*.dll'
    if ! compgen -G "$out/*ggml-cuda*" >/dev/null; then
      echo "fetch-libs: no ggml-cuda in $assets; the archive layout changed" >&2
      exit 1
    fi
    echo "$target: $(ls "$out" | wc -l | tr -d ' ') files in $out"
    return
  fi

  # Only the libraries: the archives also carry llama-cli, llama-server and
  # friends, which the release does not ship.
  # libomp.dll is ggml-base's OpenMP runtime in the Windows (clang-cl) builds.
  copy_matching "$stage" "$out" 'libllama.*' 'llama.dll' 'libggml*' 'ggml*.dll' 'libomp*.dll' '*.metal'
  # The CUDA backend ships only in the add-on.
  rm -f "$out"/*ggml-cuda*

  if (( TOOLS )); then
    # The tools resolve their libraries next to themselves ($ORIGIN, the DLL
    # directory, @loader_path), so the archive is unpacked flat as it is.
    local tools=dist/$target/tools
    rm -rf "$tools"
    mkdir -p "$tools"
    find "$stage" \( -type f -o -type l \) -exec cp -P {} "$tools/" \;
    if ! compgen -G "$tools/llama-server*" >/dev/null; then
      echo "fetch-libs: $assets contains no llama-server; the archive layout changed" >&2
      exit 1
    fi
    echo "$target: $(ls "$tools" | wc -l | tr -d ' ') files in $tools"
  fi

  # Layout check: no libllama, or no CPU backend, means the archive changed
  # shape and the binary would start without an engine.
  if ! compgen -G "$out/*llama.*" >/dev/null; then
    echo "fetch-libs: $assets contains no libllama; the archive layout changed" >&2
    exit 1
  fi
  if ! compgen -G "$out/*ggml-cpu*" >/dev/null && ! compgen -G "$out/*ggml.*" >/dev/null; then
    echo "fetch-libs: $assets contains no ggml CPU backend; the archive layout changed" >&2
    exit 1
  fi
  echo "$target: $(ls "$out" | wc -l | tr -d ' ') files in $out"
}

# darwin_universal: the arm64 and x86_64 dylibs merged with lipo (SPEC §10,
# step 2). llvm-lipo does the same job on a Linux runner.
universal() {
  local lipo out f name
  lipo=$(command -v lipo || command -v llvm-lipo || true)
  if [[ -z "$lipo" ]]; then
    echo "fetch-libs: darwin_universal needs lipo or llvm-lipo" >&2
    exit 1
  fi
  fetch darwin_arm64
  fetch darwin_amd64
  out=dist/darwin_universal/lib
  rm -rf "$out"
  mkdir -p "$out"
  for f in dist/darwin_arm64/lib/*; do
    name=$(basename "$f")
    if [[ -L "$f" ]]; then
      cp -P "$f" "$out/$name"
    elif [[ "$name" == *.dylib && -f "dist/darwin_amd64/lib/$name" ]]; then
      "$lipo" -create "$f" "dist/darwin_amd64/lib/$name" -output "$out/$name"
    else
      # Only arm64 has it: ggml-metal, since the x86_64 build has no Metal.
      # On an Intel Mac it fails to load and ggml skips the backend.
      cp "$f" "$out/$name"
    fi
  done
  echo "darwin_universal: $(ls "$out" | wc -l | tr -d ' ') files in $out"
}

targets=("$@")
if (( ${#targets[@]} == 0 )); then
  targets=("$(host_target)")
fi
for t in "${targets[@]}"; do
  if [[ "$t" == darwin_universal ]]; then universal; else fetch "$t"; fi
done
