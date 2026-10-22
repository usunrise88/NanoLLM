# Our own ggml-cuda.dll on the RTX 3070 Ti Laptop (Gemma 4 E4B, vram-4g)

Work on the CUDA kernels of Gemma 4 E4B decoding, 2026-10-21. The plan is `docs/AUTOTUNE.md` ("Единый механизм
настройки", the kernel-variant rule); the patches are `patches/llama.cpp/0008…` and later ("cuda:" subjects).
Every change has a switch `GGML_CUDA_NANO_<NAME>` (`ggml/src/ggml-cuda/nano.cuh`), so both sides of an A/B are one
DLL; `bench --set env.GGML_CUDA_NANO_X=0` / `1` sets it for a worker.

Method: one run per variant from a cool GPU (`run_dev_sweep.ps1 -Once`, ≤ 66 °C; the laptop GPU throttles within a
minute otherwise), the dev suite (pp, tg, chat, MTP chat, echo). Single runs, no statistics: they show a direction,
not a decimal. Kernel times come from Nsight Systems (`scripts/cuda-profile.ps1`), which is exact per launch.

## Build and parity

`scripts/build-libs-windows.ps1 -Cuda -Tests -Tools` builds ggml-cuda.dll (CUDA 13.4, MSVC host compiler, sm_86 only
for the dev card, ~6 min for a change in common.cuh) and `test-backend-ops`, `llama-bench` into
`.cache/llama.cpp/out-windows_amd64-cuda`. The dev lib directory is `C:\Projects\runtimes\libs\dev-cuda` (n7 libs
without Vulkan, our ggml-cuda.dll, the CUDA 13 runtime); `cuda-dev.yaml` points at it.

Without any patch (CUDA 13.4, sm_86, MSVC) against upstream's CUDA 12.4 DLL, `llama-bench` pp512 / tg128, A B B A:

| | pp512 | tg128 |
| --- | --- | --- |
| upstream 12.4 | 6002 / 5937 | 94.4 / 95.6 |
| ours 13.4, no patch | 5981 / 6060 | 93.2 / 94.6 |

Parity (±1 %). `test-backend-ops` on CUDA0 for MUL_MAT, RMS_NORM, GLU, FLASH_ATTN_EXT, UNARY, ROPE, SET_ROWS without
patches: 7104 of 7104 pass.

## Where the time goes (Nsight Systems, the whole dev suite, no patches)

`raw/nsys-base.txt` (+ `raw/nsys-base_cuda_gpu_kern_sum.csv`). 4953 ms of kernel time:

| kernels | share |
| --- | --- |
| MMVQ Q4_0, one token (gate+up fused with GEGLU 28 %, the other matrices 28 %, small-K 2 %) | 58 % |
| MMVQ Q4_0, a verify of four tokens (gate, up and the rest unfused) | 13.5 % |
| RMS norms (post norms with the residual 5.4 %, input norms 3.1 %, q/k with rope 1.6 %) | ~11 % |
| quantize_q8_1 (151 145 launches, 1.4 us each) | 4.4 % |
| flash attention (D256 and D512, 1–4 tokens) + fixups | ~4 % |
| the rest (mul, set_rows, get_rows, gelu, argmax of the sampler, top-k) | ~5 % |

Per matrix (from the earlier tg64 trace, `rtx-2026-10-21/raw/nsys-tg64.sqlite`, median per launch):

| matrix | per token | launch | bytes | GB/s |
| --- | --- | --- | --- | --- |
| gate+up fused (10240 rows, K 2560, two weights) | 42 | 84.1 us | 29.4 MB | 350 |
| down (K 10240) and o (K 2048/4096), 2560 rows | 84 | 35.7 us | 3.7–14.7 MB | ~350–380 |
| output head (262144 rows, K 2560) | 1 | 883 us | 377 MB | **427** |
| q 2048 rows | 35 | 8.5 us | 2.95 MB | 347 |
| k, v 512 rows | 40 | 3.5 us | 0.74 MB | 211 |
| ple gate 256 rows / ple proj 2560 rows (K 256) | 42+42 | 2.5 / 3.3 us | 0.37 MB | 111–148 |

The big matrices run at 78–95 % of the 448 GB/s peak, the small ones are launch-bound; a kernel is ~25 per layer,
~1100 per token, and each costs at least ~1.5 us even when it does nothing. So the levers are fewer kernels (fuse), and
the K loop of the matrices whose K is not a multiple of the block (K = 2560 is 80 blocks of Q4_0, the 4-warp block
covers 64 per trip: the second trip is one warp's).

## Result (patch 0008, everything in one DLL)

Kernel time per evaluation from Nsight Systems of `llama-bench` (`nsys_mmvq.py total`; a token is one launch of the
262144-row head), all switches off against the defaults:

| | off | defaults |
| --- | --- | --- |
| one token | 9.621 ms, 1195 kernels (mmvq 7.37, norm 1.05, fa 0.47, quantize 0.43, other 0.31) | **8.451 ms**, 1062 kernels (mmvq 6.69, norm 0.73, fa 0.46, quantize 0.25, other 0.32) |
| four tokens (`-p 4`) | 9.925 ms, 1268 kernels | **9.070 ms**, 1111 kernels |

The dev suite, one run each from a cool GPU (`ablation.ps1 -Name fin`; `raw/dev-fin-*`, in `../rtx-2026-10-21/raw`). Single
runs: the direction, not the decimal (±3-5 %).

| setting | tg | chat | MTP chat (acc.) | echo |
| --- | --- | --- | --- | --- |
| everything off | 93.0 | 93.5 | 156.6 (0.44) | 265.0 |
| **defaults** | **106.5** | **107.2** | **174.2 (0.46)** | **285.7** |
| lean norm off | 99.5 | 99.9 | 157.7 | 279.6 |
| MMVQ block shapes off | 101.1 | 100.8 | 149.9 | 280.4 |
| q8_1 cache and norm q8_1 off | 101.8 | 102.0 | 167.2 | 272.5 |
| graph per shape off | 104.8 | 103.9 | 167.3 | 278.2 |
| gelu*mul off | 106.7 | 106.8 | 172.7 | 288.3 |
| vector FA on | 106.0 | 106.0 | 172.2 | 287.7 |

### The block of the Q4_0 mat-vec

`mmvq_sweep.py` (test-backend-ops perf) over the E4B matrices said one warp per row is faster for K = 2560, but the small
matrices are served from the 4 MB L2 there (592 GB/s on a 448 GB/s card): only the big ones (gate/up, down, the head) are
valid. The real decode (nsys of llama-bench tg64, `run_nsys_shapes.sh`) shows where it matters: the **fused** gate/up (two
weights per row) is the one kernel below the card's ceiling.

| block (warps, rows) | fused gate/up, one token | per token | gate or up, four tokens |
| --- | --- | --- | --- |
| upstream: 4 warps, 1 row (2 rows for 2-4 tokens) | 84.4 us | 3.67 ms | 41.3 us |
| 1 warp, 1 row | 71.7 | 3.10 | |
| 1 warp, 2 rows | 71.2 | 3.05 | 38.4 |
| 1 warp, 4 rows | 75.4 | 3.25 | 38.1 |
| 2 warps, 1 row | 71.6 | 3.07 | |
| 3 warps, 1 row | 71.5 | 3.09 | |
| 4 warps, 2-4 rows | 80-81 | 3.5 | |

The other matrices (down, q, o, the head: 427 GB/s) do not gain; a shape that is faster for gate/up only changes their
bits. The table in `mmvq_nano_shape` therefore applies one warp per row to K = 2560, 8192-65536 rows, for 1-4 columns.

### Four tokens at once give the bits one token gives

`TestNativeMTPEquivalence` (MTP decoding equals plain decoding token for token) failed with the first inexact variants at
top-1/top-2 margins of 0.1-0.34, far above the rounding of two correct kernels. Cause: the order of a row's partial sums
differs when the number of warps differs between one and four columns, and the lean norm treated a 3-D input (the
per-layer norms: a batch is a channel there) with upstream's kernel for four tokens and its own for one. The model's int8
activations turn any such rounding into a flipped value. After the fix (one warp count for all columns, channels in the
lean kernel, the Q4_0 dot rounded op by op: `a*b - c*d` was fused differently in the fused gate/up and in the two mat-vecs
of a verification, 37 % of the rows) the equality holds on 13 prompts and `test-nano-batch-invariance` finds every matrix
bit for bit the same with 1-4 columns, the gate and up with the GEGLU too.

### Quality

`kld_cuda.ps1` (llama-perplexity, 30 chunks of 2048 = 61 thousand tokens, reference: upstream's kernels on the same card):

| | mean KLD | max | p99 | same top-1 |
| --- | --- | --- | --- | --- |
| `-fa off` (another correct algorithm: the noise) | 0.001424 +- 0.000037 | 0.73 | 0.01233 | 98.28 % |
| defaults | 0.001623 +- 0.000243 | 7.36 | 0.01179 | 98.32 % |

The defaults differ from the noise in one position (max 7.36), where both norm changes together flip a near-tie of the model
(each alone: max 0.47 and 0.11); without it the mean is 0.0015. The mat-vec shapes are not in this run (a batch above 8 tokens
goes to MMQ); the bit-for-bit test and test-backend-ops cover them.

### MTP draft length with the final kernels

The dev suite, one run each (`raw/dev-mtp3-*`; the default measured twice, 173.6 and 167.0: the noise is ~4 %). The output
is the same token for token in every row (sha 52793f74), which is the batch invariance at work.

| draft_max, p_min | MTP chat (acceptance) | echo |
| --- | --- | --- |
| 2, 0 | 167.3 (0.54) | 233.3 |
| **3, 0 (the default)** | **173.6 / 167.0 (0.46)** | **288.2 / 277.8** |
| 4, 0 | 151.7 (0.37) | 310.0 |
| 4, 0.3 | 163.7 (0.49) | 302.7 |
| 4, 0.5 | 159.5 (0.60) | 304.0 |
| 6, 0.5 | 151.7 (0.52) | 329.7 |

The graph per shape removes the old cost of a minimum probability (it was 130 against 147 tok/s on a single graph), so
4 / 0.3 is now a fair alternative: -3 % on chat for +9 % on repetitive text. The planner keeps 3 / 0.

### What did not work

- The gate/up GEGLU fusion for 2-4 tokens (`FUSE_BATCH`): the kernel takes 156 us for four tokens (89 us with one warp) against
  87 us for the two mat-vecs and the GEGLU launch it replaces. Off.
- The vector flash attention kernel for one token on Ampere (`FA_VEC`): no change on cc 8.6, at any depth (below). Off.
- `gelu*mul` in one launch: no measurable change (42 launches of 1.2 us); kept, it is exact.

## The add-ons of release b11600-n8, from CI, on the RTX

The two archives the libs-build workflow published (`nanollm-libs-windows_amd64_cuda.tar.gz`, CUDA 12.4, 655 MB, and
`..._cuda13.tar.gz`, CUDA 13.4, 561 MB; sha256 as in `scripts/libs.sha256`, checked after the download) unpacked over a lib
directory of the same series: `llama.dll`, `ggml.dll`, `ggml-base.dll` built from the clean tree c96d19c52 by
`build-libs-windows.ps1 -Variants none -Core`, and the CPU variants and libomp of n7 (patch 0008 does not touch them).
`nanollm devices` lists `backend-sampling cuda-nano` for the card with both. Their ggml-cuda.dll is the full release list
of architectures (75, 80, 86 real, 89 real, 90, and for 12.4 the PTX of 50, 61, 70): 585 MB for 12.4, 151 MB for 13.4,
against the 32 MB of the sm_86-only dev build.

The dev suite, one run each from a cool GPU (`run_dev_sweep.ps1 -Once -Reps 2`; `../rtx-2026-10-21/raw/dev-n8cu12-*`,
`dev-n8cu13-*`); single runs, the direction not the decimal:

| ggml-cuda.dll | pp | tg | chat | MTP chat (acc.) | echo | output sha |
| --- | --- | --- | --- | --- | --- | --- |
| dev build, 13.4, sm_86 only (`dev-fin-2`) | 3973 | 106.5 | 107.2 | 174.2 (0.46) | 285.7 | 52793f74 |
| release add-on, 13.4 | 4651 | 104.3 | 104.5 | 169.2 (0.46) | 282.4 | 52793f74 |
| release add-on, 12.4 | 4207 | 102.4 | 102.3 | 169.1 (0.47) | 273.7 | 8271256b |

The 13.4 add-on gives the output of the dev build bit for bit and the speed within the noise. The 12.4 one is 2-4 % slower
and writes other tokens in the greedy chat run (another compiler, other rounding in the places the patch leaves to it, as
upstream's own 12.4 and 13.4 builds differ). What matters is that the property the patch exists for holds in it too:
`NANOLLM_TEST_DEVICE=cuda0 go test -tags native` with each add-on, `TestNativeBatchInvariance` (largest difference 0 for
batches of 2, 3 and 4 on both prompts), `TestNativeMTPEquivalence` (MTP == plain decoding on 13 prompts, 12.4: 310 of 847
drafts accepted, 85 -> 150 tok/s; 13.4: 309 of 860, 85 -> 144), `TestNativeCudaNano` and `TestNativeBackendSampling` pass.

## Flash attention kernel for one token, by depth (plan item 1.7)

Upstream picks the MMA kernel for D=256 on Ampere and the vector kernel from Ada on; `GGML_CUDA_NANO_FA_VEC=1` takes the vector
kernel for one token on cc 8.6 too. `fa_depth.ps1`: `llama-bench -fa 1 -p 0 -n 64 -d 0,4096,8192 -r 2`, A B B A, each from a
cool GPU (<= 66 C), with the ggml-cuda.dll of the b11600-n8 CUDA 13.4 add-on (`raw/fa-depth-*.csv`), tg in tokens/s:

| depth | FA_VEC=0 (MMA) | FA_VEC=1 (vector) |
| --- | --- | --- |
| 0 | 104.1, 104.7 | 104.2, 104.9 |
| 4096 | 97.9, 98.0 | 97.4, 97.2 |
| 8192 | 95.9, 95.4 | 96.1, 96.4 |

No difference at any depth (< 1 %). nsys of the same model at depth 4096 (`raw/nsys-fa-vec.txt`) shows the switch changes the
kernel: 100 launches of `flash_attn_ext_f16<256,256,2,4>` (10.05 us) + 100 `stream_k_fixup` (2.6 us) become 100 of
`flash_attn_ext_vec<256,1>` (11.35 us) + 65 `combine_results` (2.0 us): 1.26 ms against 1.26 ms. The D=512 layers have no vector
kernel (the vector instances stop at 256) and 2-4 tokens go to the MMA kernel on Ampere for an unquantized cache, so there is
nothing to choose there. All flash attention is 0.46 ms of the 8.45 ms of a token: whatever the choice, it can move the token
by 5 % at most, and by measurement it moves it by nothing. The rule for cc 8.6 stays upstream's (`FA_VEC` 0).
