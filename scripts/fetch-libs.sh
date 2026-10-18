#!/usr/bin/env bash
# Fetch the native libraries of one pinned llama.cpp release (SPEC §10, step 1).
#
#   scripts/fetch-libs.sh [TAG] [TARGET...]
#
# TAG defaults to the contents of scripts/llama.cpp.tag. TARGET is any of
# linux_amd64 linux_arm64 windows_amd64 windows_arm64 darwin_arm64
# darwin_amd64 darwin_universal; the default is the machine this runs on.
#
# Each archive is verified against scripts/libs.sha256 before it is opened: an
# archive whose hash is not listed there is refused, not trusted on first use.
# libllama and libggml* are unpacked into dist/<os>_<arch>/lib/.
#
# NanoLLM compiles no C code. If an upstream archive layout changes, this
# script fails on the layout check below rather than shipping a lib/ that is
# missing a backend (SPEC §14, risks).
set -euo pipefail

cd "$(dirname "$0")/.."

MIN_TAG=b9568 # keep in step with llamacpp.MinTag
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

# The release asset that carries each target's CPU and GPU backends (SPEC §10,
# release archives). Confirm these names against the release page whenever the
# tag is bumped: upstream renames assets from time to time, and the sha256 list
# is what catches it.
asset_for() {
  case "$1" in
    linux_amd64)   echo "llama-$TAG-bin-ubuntu-vulkan-x64.zip" ;;
    linux_arm64)   echo "llama-$TAG-bin-ubuntu-arm64.zip" ;;
    windows_amd64) echo "llama-$TAG-bin-win-vulkan-x64.zip" ;;
    windows_arm64) echo "llama-$TAG-bin-win-cpu-arm64.zip" ;;
    darwin_arm64)  echo "llama-$TAG-bin-macos-arm64.zip" ;;
    darwin_amd64)  echo "llama-$TAG-bin-macos-x64.zip" ;;
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

fetch() {
  local target=$1 asset want got out stage
  asset=$(asset_for "$target") || { echo "fetch-libs: unknown target $target" >&2; exit 1; }

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

  stage=$(mktemp -d)
  STAGES+=("$stage")
  unzip -q "$CACHE/$asset" -d "$stage"

  out=dist/$target/lib
  rm -rf "$out"
  mkdir -p "$out"
  # Only the libraries: the archives also carry llama-cli, llama-server and
  # friends, which the release does not ship.
  find "$stage" -type f \( -name 'libllama.*' -o -name 'llama.dll' \
    -o -name 'libggml*' -o -name 'ggml*.dll' -o -name '*.metal' \) -exec cp {} "$out/" \;

  # Layout check: no libllama, or no CPU backend, means the archive changed
  # shape and the binary would start without an engine.
  if ! compgen -G "$out/*llama.*" >/dev/null; then
    echo "fetch-libs: $asset contains no libllama; the archive layout changed" >&2
    exit 1
  fi
  if ! compgen -G "$out/*ggml-cpu*" >/dev/null && ! compgen -G "$out/*ggml.*" >/dev/null; then
    echo "fetch-libs: $asset contains no ggml CPU backend; the archive layout changed" >&2
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
    if [[ "$name" == *.dylib && -f "dist/darwin_amd64/lib/$name" ]]; then
      "$lipo" -create "$f" "dist/darwin_amd64/lib/$name" -output "$out/$name"
    else
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
