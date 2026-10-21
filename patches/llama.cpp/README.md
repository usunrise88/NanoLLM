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
- 0006 vulkan: on Intel GPUs that share memory with the host and run the Windows driver (Iris Xe), the large mat-vec
  workgroup for every row count and 4 rows per workgroup in the batch kernels (host-side choices in `ggml-vulkan.cpp`;
  decode +12%, verification of 2 to 3 speculative drafts +14 to +21%, prefill unchanged). The Windows build of the libs
  ships our `ggml-vulkan.dll` for it; other vendors and drivers keep upstream's choices.
- 0007 llama: `LLAMA_FLASH_ATTN_TYPE_SMALL_BATCH`, flash attention on batches of up to 8 tokens and on sliding-window
  layers, the matmul attention for longer batches on the full-attention layers (libllama, plus `small_batch` for
  common's and llama-bench's `-fa`; needs an unquantized KV cache, which keeps the flash-attention layout). Prefill on an
  Iris Xe +23% at 4096 tokens of context, +42% at 8192, KLD below that of `-fa 0`; gemma.json on the iGPU 95 -> 89-90 s.
  NanoLLM asks for it on GPUs listing `fa-small-batch` (Intel iGPUs under Vulkan on Windows).
- 0008 cuda: kernels and CUDA graphs for Gemma 4 E4B decoding, with a switch (`GGML_CUDA_NANO_<NAME>`) for each change
  (`ggml/src/ggml-cuda/nano.cuh`): a CUDA graph per batch shape, the block of the Q4_0 mat-vec per column count, a lighter
  rms_norm, the q8_1 of an activation kept for the next mat-vec (and written by the norm), gelu*mul in one launch. On an RTX
  3070 Ti Laptop the kernel time of a token goes from 9.62 to 8.45 ms (-12 %), of a draft's verification from 9.93 to
  9.07 ms; a token verified in a batch gets the same bits as decoded alone (`tests/test-nano-batch-invariance.cpp`).
  It touches only `ggml-cuda` and the tests, so it works beside the `ggml-base` of any series; `libs-build.yml` builds it
  for two toolkits (CUDA 12.4 and 13.4) as `nanollm-libs-windows_amd64_cuda` and `..._cuda13`, and
  `scripts/build-libs-windows.ps1 -Cuda` builds it locally. Results: docs/AUTOTUNE.md, "Своя ggml-cuda.dll".

0002 to 0005 change the CPU backend, so the targets marked `cpu` in
`libs-build.yml` ship the CPU variants too (see there and `scripts/fetch-libs.sh`), 0006 changes the
Vulkan backend, which the `vulkan` targets ship (windows_amd64 for now), and 0008 changes only the CUDA backend, which the
`cuda` targets ship (windows_amd64_cuda and _cuda13; Linux keeps upstream's for now).
Without patches the build is upstream's.

## Patches under test: `patches/next`

A patch that is written but not yet shipped lives in `patches/next/`, so the
released series above (the one `scripts/llama.cpp.commit`, CI and the release
describe) does not move under a running tree. `scripts/build-libs.sh --next`
applies the series and then `patches/next/*.patch`, writes `dist/linux_amd64-next`,
and prints the commit of the combined tree. Try it with a binary stamped for
that commit and `NANOLLM_DEV_SERIES=<commit>`:

    go build -ldflags "-X github.com/usunrise88/nanollm/internal/backend/llamacpp.PinnedCommit=<commit>" ./cmd/nanollm

`next/` is empty now.

When the owner agrees to a libs release, the patches move from `next/` into
this directory, the new commit goes into `scripts/llama.cpp.commit`, the
bindings are regenerated (`make bindgen`), and `libs-build.yml` publishes the
new `b11600-nN`.
