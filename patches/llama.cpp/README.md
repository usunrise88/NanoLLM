# Patches to llama.cpp

NanoLLM builds its own libraries from the pinned upstream tag plus the patches
in this directory (`scripts/build-libs.sh`, docs/STAGE2.md 2.1). Each patch is
a `git format-patch` file, `NNNN-short-name.patch`, applied in order with
`git am`. The author and date inside a patch are what the build commit
carries, so the hash of the patched tree is the same on every machine.

Rules:

- A patch changes one thing and says in its message what it measured: speed
  before and after, and the KL divergence against the unpatched build when it
  can change the numbers (docs/STAGE2.md 2.4).
- After any change to the series, write the new commit hash (`git rev-parse
  HEAD` of the patched checkout) into `scripts/llama.cpp.commit`.
  `scripts/llama.cpp.base` stays the upstream commit.
- A patch that fixes something for everyone is offered upstream, once the
  owner agrees.

The series, in order:

- 0001 gemma4: drop the rows nobody needs after the last KV-owning layer (OPT-11, libllama).
- 0002 x86: Q4_0 8x8 GEMV/GEMM kernels for AVX2 without VNNI, unsigned nibbles
  (OPT-16, CPU backend; bit-exact with the existing kernels, prefill x1.8-1.9,
  decode x1.15-1.4 on a Zen 1).
- 0003 ggml-cpu: repack a tensor on several threads while the model loads, with
  NUMA first-touch (OPT-14, CPU backend; `GGML_REPACK_THREADS`).
- 0004 x86: prefetch the Q4_0 weight blocks 8 ahead in the multi-row GEMV; the CPU backend lists
  `REPACK_Q4_0_ROWS` (the CPU MTP default keys on it).
- 0005 x86: the Q4_0 8x8 kernels of 0002 on CPUs with AVX-VNNI and no AVX512-VNNI (the alderlake
  variant: Alder Lake and later), with `vpdpbusd` instead of `vpmaddubsw` (OPT-16-VNNI; bit-exact
  with upstream's lut kernels, prefill x1.4 and MTP on the CPU x1.9 on an i7-12700H; the decode of
  one token is at the memory ceiling and does not move). Windows build:
  `scripts/build-libs-windows.ps1`; kernels: `scripts/kernel-bench/run-windows.ps1`.

0002 to 0005 change the CPU backend, so the targets marked `cpu` in
`libs-build.yml` ship the CPU variants too (see there and `scripts/fetch-libs.sh`).
Without patches the build is upstream's.

## Patches under test: `patches/next`

A patch that is written but not yet shipped lives in `patches/next/`, so the
released series above (the one `scripts/llama.cpp.commit`, CI and the release
describe) does not move under a running tree. `scripts/build-libs.sh --next`
applies the series and then `patches/next/*.patch`, writes `dist/linux_amd64-next`,
and prints the commit of the combined tree. Try it with a binary stamped for
that commit and `NANOLLM_DEV_SERIES=<commit>`:

    go build -ldflags "-X github.com/usunrise88/nanollm/internal/backend/llamacpp.PinnedCommit=<commit>" ./cmd/nanollm

When the owner agrees to a libs release, the patches move from `next/` into
this directory, the new commit goes into `scripts/llama.cpp.commit`, the
bindings are regenerated (`make bindgen`), and `libs-build.yml` publishes the
new `b11600-nN`.
