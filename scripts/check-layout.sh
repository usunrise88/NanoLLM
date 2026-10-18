#!/usr/bin/env bash
# Check the struct layout the libllama binding assumes against the C compiler,
# for every release target (STAGE1 1.5, layout test).
#
#   scripts/check-layout.sh LLAMA_CPP_SRC
#
# LLAMA_CPP_SRC is a checkout of the tag in scripts/llama.cpp.tag. tools/bindgen
# writes internal/backend/llamacpp/testdata/layout.c: one _Static_assert per
# size, alignment, field offset and enum value the Go side hard-codes. A
# mismatch on any target fails compilation, so a header change that would
# corrupt memory through purego fails here instead.
#
# The compiler is zig cc, which carries libc headers for every target, so one
# Linux job covers all six; `pip install ziglang` provides it as
# `python3 -m ziglang`. Each target is compiled to a throwaway object (zig's
# -fsyntax-only trips over its own cache); nothing is linked or run.
set -euo pipefail

cd "$(dirname "$0")/.."

SRC="${1:?usage: scripts/check-layout.sh LLAMA_CPP_SRC}"
FILE=internal/backend/llamacpp/testdata/layout.c

if command -v zig >/dev/null; then
  ZIG=(zig)
elif python3 -m ziglang version >/dev/null 2>&1; then
  ZIG=(python3 -m ziglang)
else
  echo "check-layout: needs zig (pip install ziglang)" >&2
  exit 1
fi

# The Windows targets use the MinGW headers zig ships; the structs involved
# (fixed-width integers, pointers, size_t, bool, enums) lay out the same under
# the MSVC ABI that the release DLLs are built with.
TARGETS=(
  x86_64-linux-gnu
  aarch64-linux-gnu
  x86_64-windows-gnu
  aarch64-windows-gnu
  x86_64-macos
  aarch64-macos
)

OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

fail=0
for t in "${TARGETS[@]}"; do
  if "${ZIG[@]}" cc -target "$t" -std=c11 -c -o "$OUT/$t.o" \
      -I "$SRC/include" -I "$SRC/ggml/include" "$FILE"; then
    echo "layout ok: $t"
  else
    echo "layout MISMATCH: $t" >&2
    fail=1
  fi
done
exit $fail
