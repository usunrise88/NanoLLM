#!/usr/bin/env bash
# Build release archives (SPEC §10). Prefer `make dist`; CI and the release
# workflow call this directly so that every archive is made one way.
#
#   scripts/dist.sh [TARGET...]
#
# TARGET is one of
#   linux-amd64 linux-arm64 windows-amd64 windows-arm64 darwin-universal
#   linux-amd64-cuda windows-amd64-cuda        (the CUDA add-ons: CUDA 12.x)
#   windows-amd64-cuda13                       (the CUDA 13.4 add-on, Turing and newer, driver 580+)
#   all                                        (every one of the above)
# and defaults to this machine. Environment:
#   VERSION   the release tag stamped into the binary and the file names
#             (default: git describe)
#   UI        both (default): each archive with and without the web UI;
#             1: only with it; 0: only the -noui archives
#   OUT       where the archives go (default dist)
#   FETCH     1 (default): run fetch-libs for each target first; 0: use the
#             dist/<target>/lib/ that is there
#
# NanoLLM compiles no C: every binary is CGO_ENABLED=0 and cross-compiles
# from one Linux runner. Each archive holds the binary, lib/ from the pinned
# llama.cpp release, nanollm.yaml (the annotated example), the service unit
# for its platform, LICENSE, NOTICE and README.md, flat, so that it unpacks
# over an installation. A CUDA add-on holds lib/ with ggml-cuda and the CUDA
# runtime only, and unpacks over the same directory.
#
# The darwin binary is universal: both slices are built and merged with lipo,
# or llvm-lipo on Linux; fetch-libs merges the dylibs the same way. Signing
# and notarization happen afterwards on a Mac (release.yml).
#
# The archive paths are printed on stdout, one per line; progress goes to
# stderr.
set -euo pipefail
# A failure inside $(...) stops the script too (bash 4.4+; macOS's 3.2 lacks it).
shopt -s inherit_errexit 2>/dev/null || true

cd "$(dirname "$0")/.."

VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"
UI="${UI:-both}"
OUT="${OUT:-dist}"
FETCH="${FETCH:-1}"

# The module, not a workspace: a go.work on the build machine must not change
# what the release is built from.
export GOWORK=off CGO_ENABLED=0

say() { printf '==> %s\n' "$*" >&2; }
die() { printf 'dist.sh: %s\n' "$*" >&2; exit 1; }

ALL=(linux-amd64 linux-arm64 windows-amd64 windows-arm64 darwin-universal linux-amd64-cuda windows-amd64-cuda windows-amd64-cuda13)

host_target() {
  local os arch
  case "$(uname -s)" in
    Linux) os=linux ;;
    Darwin) echo darwin-universal; return ;;
    MINGW*|MSYS*|CYGWIN*) os=windows ;;
    *) die "unsupported host $(uname -s)" ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64) arch=amd64 ;;
    arm64|aarch64) arch=arm64 ;;
    *) die "unsupported architecture $(uname -m)" ;;
  esac
  echo "$os-$arch"
}

case "$UI" in
  both) VARIANTS=(ui noui) ;;
  1) VARIANTS=(ui) ;;
  0) VARIANTS=(noui) ;;
  *) die "UI is \"$UI\"; use both, 1 or 0" ;;
esac

# The UI is embedded at compile time from internal/ui/dist, which holds only
# a placeholder page in a fresh checkout. A build that ships the SPA must
# have it staged (make web); one that does not is built with -tags noui.
if [[ " ${VARIANTS[*]} " == *" ui "* && ! -d internal/ui/dist/assets ]]; then
  die "the UI archives need internal/ui/dist/assets: run 'make web' first, or UI=0 for -noui only"
fi

LLAMACPP=github.com/usunrise88/nanollm/internal/backend/llamacpp
LLAMA_TAG=$(tr -d '[:space:]' < scripts/llama.cpp.tag)
LDFLAGS="-X main.version=$VERSION -X $LLAMACPP.PinnedTag=$LLAMA_TAG"
# Without the commit file the binary falls back to the commit its binding was
# generated from, which is what the libraries must match anyway.
if [[ -s scripts/llama.cpp.commit ]]; then
  LDFLAGS+=" -X $LLAMACPP.PinnedCommit=$(tr -d '[:space:]' < scripts/llama.cpp.commit)"
fi
# The autotune profile is keyed on the libraries' release.
if [[ -s scripts/libs.release ]]; then
  LDFLAGS+=" -X $LLAMACPP.LibsRelease=$(tr -d '[:space:]' < scripts/libs.release)"
fi

lipo_tool() {
  local t
  for t in lipo llvm-lipo; do
    if command -v "$t" >/dev/null; then command -v "$t"; return; fi
  done
  # Debian and Ubuntu install it versioned: llvm-lipo-18.
  t=$(compgen -c llvm-lipo- 2>/dev/null | sort -V | tail -1 || true)
  [[ -n "$t" ]] && command -v "$t" && return
  die "darwin-universal needs lipo or llvm-lipo (apt-get install llvm)"
}

# build GOOS GOARCH VARIANT OUTFILE
build() {
  local tags=()
  [[ "$3" == noui ]] && tags=(-tags noui)
  say "go build $1/$2${tags:+ (noui)}"
  GOOS=$1 GOARCH=$2 go build -trimpath ${tags[@]+"${tags[@]}"} -ldflags "$LDFLAGS" -o "$4" ./cmd/nanollm
}

# libs TARGET: dist/<fetch-libs target>/lib, fetched unless FETCH=0.
libs() {
  local lt=${1//-/_}
  if [[ "$FETCH" == 1 ]]; then
    ./scripts/fetch-libs.sh "$LLAMA_TAG" "$lt" >&2
  fi
  [[ -d "dist/$lt/lib" ]] || die "no dist/$lt/lib; run scripts/fetch-libs.sh \"\" $lt (or FETCH=1)"
  echo "dist/$lt/lib"
}

# pack STAGE ARCHIVE: a zip for Windows, a gzipped tar for the rest, with the
# stage's contents at the top level.
pack() {
  local stage=$1 archive=$2 abs
  abs="$(pwd)/$archive"
  rm -f "$abs"
  case "$archive" in
    *.zip)
      if command -v zip >/dev/null; then
        (cd "$stage" && zip -qry "$abs" .)
      elif command -v 7z >/dev/null; then
        (cd "$stage" && 7z a -tzip -bso0 -bsp0 "$abs" ./*)
      elif command -v powershell >/dev/null; then
        powershell -NoProfile -Command "Compress-Archive -Path '$stage/*' -DestinationPath '$abs' -Force"
      else
        die "no zip, 7z or PowerShell to make $archive"
      fi ;;
    *)
      # Owned by root in the archive, so that an install unpacked as root
      # does not hand the files to whatever uid built them.
      local owner=()
      if [[ "$(tar --version 2>/dev/null)" == *GNU* ]]; then
        owner=(--owner=0 --group=0 --numeric-owner --sort=name)
      else
        owner=(--uid 0 --gid 0)
      fi
      tar -C "$stage" "${owner[@]}" -czf "$abs" . ;;
  esac
}

STAGES=()
cleanup() {
  local d
  for d in ${STAGES[@]+"${STAGES[@]}"}; do rm -rf "$d"; done
}
trap cleanup EXIT

# package TARGET VARIANT
package() {
  local target=$1 variant=$2 os arch suffix="" ext=tar.gz exe=nanollm stage lib name
  os=${target%%-*}
  arch=${target#*-}
  [[ "$variant" == noui ]] && suffix=-noui
  [[ "$os" == windows ]] && { ext=zip; exe=nanollm.exe; }

  stage=$(mktemp -d)
  STAGES+=("$stage")

  if [[ "$target" == darwin-universal ]]; then
    local lipo
    lipo=$(lipo_tool)
    build darwin arm64 "$variant" "$stage/nanollm-arm64"
    build darwin amd64 "$variant" "$stage/nanollm-amd64"
    "$lipo" -create "$stage/nanollm-arm64" "$stage/nanollm-amd64" -output "$stage/nanollm"
    rm "$stage/nanollm-arm64" "$stage/nanollm-amd64"
  else
    build "$os" "$arch" "$variant" "$stage/$exe"
  fi
  chmod 0755 "$stage/$exe"

  lib=$LIBDIR
  mkdir -p "$stage/lib"
  cp -P "$lib"/* "$stage/lib/"
  cp configs/nanollm.example.yaml "$stage/nanollm.yaml"
  cp LICENSE NOTICE README.md "$stage/"
  case "$os" in
    linux) cp deploy/nanollm.service "$stage/" ;;
    darwin) cp deploy/nanollm.plist "$stage/" ;;
  esac
  chmod -R u+rwX,go+rX "$stage"

  # A release nobody can run is worse than none: when the archive is for
  # this machine, ask the binary what it is before packing it.
  if [[ "$target" == "$(host_target)" ]]; then
    local out want_ui=true
    [[ "$variant" == noui ]] && want_ui=false
    out=$("$stage/$exe" version)
    grep -Eq "^ui +$want_ui\$" <<<"$out" || die "$target$suffix: version says $(grep '^ui' <<<"$out"), want ui $want_ui"
    grep -q "$VERSION" <<<"$out" || die "$target$suffix: version does not say $VERSION"
  fi

  name="nanollm-$VERSION-$target$suffix.$ext"
  pack "$stage" "$OUT/$name"
  say "$OUT/$name"
  echo "$OUT/$name"
}

# addon TARGET: the CUDA add-on, lib/ only.
addon() {
  local target=$1 base=${1%-cuda*} ext=tar.gz stage
  [[ "$base" == windows-* ]] && ext=zip
  stage=$(mktemp -d)
  STAGES+=("$stage")
  mkdir -p "$stage/lib"
  cp -P "$LIBDIR"/* "$stage/lib/"
  chmod -R u+rwX,go+rX "$stage"
  local name="nanollm-$VERSION-$target.$ext"
  pack "$stage" "$OUT/$name"
  say "$OUT/$name"
  echo "$OUT/$name"
}

targets=("$@")
(( ${#targets[@]} )) || targets=("$(host_target)")
if [[ " ${targets[*]} " == *" all "* ]]; then targets=("${ALL[@]}"); fi

mkdir -p "$OUT"
for t in "${targets[@]}"; do
  [[ " ${ALL[*]} " == *" $t "* ]] || die "unknown target $t; use one of ${ALL[*]} or all"
  LIBDIR=$(libs "$t")
  if [[ "$t" == *-cuda || "$t" == *-cuda13 ]]; then
    addon "$t"
    continue
  fi
  for v in "${VARIANTS[@]}"; do package "$t" "$v"; done
done
