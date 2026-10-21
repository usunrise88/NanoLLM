# Throughput phase on the i7-12700H laptop: stage 0 measurements and the first results (2026-10-21)

Machine: i7-12700H (6 P-cores with SMT + 8 E-cores, AVX2 + AVX-VNNI), 2x16 GiB DDR4-3200, Iris Xe (96 EU), two
Kingston NVMe drives, Windows 11, AC, power plan Performance. NanoLLM dev binary built from the working tree
(`dist/windows_amd64/nanollm-dev.exe`), llama.cpp b11600 `aad9f95e3` (the n4 series) unless said otherwise.
One run per configuration (see `feedback-single-run-enough`); where a decision rests on a few percent the runs are
alternated and repeated. Raw files: `raw/` (bench result files and logs), `gemma/` (gemma.json runs), the drivers
`run_stage*.ps1`, `run_gemma.ps1`, `summarize.py`.

**Caveat on the first runs.** Everything measured before 03:25 (the first CPU sweeps, the Vulkan env sweep) ran
while the toolchain installer's leftovers (Defender scans of the new files) and my own tooling were using the
machine: pp numbers of those files are 10-25% low and noisy. The CPU thread sweep was redone on a quiet machine
(`driver-c.log`); the Vulkan env sweep is GPU-bound and its conclusion (nothing helps) does not depend on it.

## 1. What the hardware allows

| | |
|---|---|
| Memory read ceiling (`tools/membw`, AVX2 loads, 256 MiB per thread) | 6 P-cores one thread each **37.3 GB/s**, 2 threads 31.9, 12 threads 37.6, all 20 CPUs 38.2, 8 E-cores 33.9 (51.2 GB/s is the paper peak) |
| Bytes read per decoded token | 2.7 GB (2.1 GB of layers + the 360 MiB tied head) |
| Decode ceiling | 37.3 / 2.7 = **13.8 tok/s**; measured CPU decode 13.4-13.8 (97%) |
| iGPU decode (llama-bench, `GGML_VK_PERF_LOGGER`) | 10.2-10.5 tok/s; the Q4_0 mat-vec kernels reach 29-34 GB/s per shape, the whole token is 95 ms of GPU time |
| NVMe C: (Kingston NV3, PCIe 4) | overlapped reads, queue depth 4, 4 MiB: **4.2 GB/s**; 4 unbuffered streams 2.7-3.3; `PrefetchVirtualMemory` 2.5-2.7; mmap page faults: 1 thread 1.1, 16 threads 3.9 |
| NVMe D: (Kingston SNVS) | 2.0-2.3 overlapped / unbuffered, mmap 1.1-2.0, prefetch 1.6 |

Consequences: decode is at the memory ceiling on the CPU and within 80-90% of it on the iGPU, so decode only grows
with fewer bytes per token (speculation); prefill is compute-bound and is where kernels and threads matter. A
streamed model (2.6 GB per token) cannot exceed ~1.5 tok/s from this disk.

## 2. CPU (preset ram-6g, mode M6)

gemma.json (5 requests, 16.3k prompt tokens, 849 generated), whole run:

| Configuration | Wall, s | Prefill, tok/s | Decode, tok/s |
|---|---|---|---|
| shipped libs, 6 decode / 6 batch threads (the old default) | 213 | 109-113 | 10.7-12.9 |
| shipped libs, 6 / 20 threads (new default on hybrid CPUs) | 183 | 139-145 | 10.5-12.7 |
| + patch 0005 (AVX-VNNI kernels), no MTP | 155 | 182-193 | 10.6-13.0 |
| + MTP, draft_max 2, p_min 0 | 130 | 188-195 | 14.2-28.1 |
| + MTP, draft_max 2, p_min 0.3 / 0.5 / 0.75 | 124 / 126 / 130 | | |
| + MTP, draft_max 3, p_min 0 | 132 | | |
| **planner defaults of the working tree** (6 / 20 threads, MTP 2 / 0.5, VNNI libs; two runs) | **127 / 125** | 188-198 | 14-26 |
| (reference) iGPU uma-4g MTP-2 on the same day | **101** | 393-449 | 10.0-18.8 |

- `threads_batch` (dev suite, pp512, quiet machine, decode threads 6): 6 -> 125-128 tok/s, 14 -> 138, 16 -> 152,
  18 -> 161, 20 -> 155-172; decode unchanged at 13.4-13.8. Default now: all logical CPUs for the batch team on
  hybrid CPUs (`hwinfo.DefaultBatchThreads`, `backend.LoadSpec.WithDefaultThreads`); an explicit `cpu_threads` keeps
  the batch team equal to it. With the VNNI kernels: 12 -> 170 s, 16 -> 155 s, 20 -> 153 s on gemma.json.
- OpenMP environment: `KMP_BLOCKTIME=0` decode 8.2, `OMP_PROC_BIND=spread` 5.5, `close` 10.7, `OMP_WAIT_POLICY`
  and `KMP_BLOCKTIME=200/infinite` no change: the defaults stay.
- Repack off: pp 40-58, tg 6.9-7.6 (half of the repacked decode): repack is essential. M7 (PLE read from the file)
  equals M6 in speed (12.7 vs 12.6) and frees 1.5 GiB of private memory.
- `swa_full` on/off: no measurable difference at depth 0 or 4096 on the CPU (the d4096 numbers are within the
  noise, 8-10 tok/s); off saves 0.6 GiB of RSS at 16k context but limits the prompt cache to the last 512 tokens, so
  the planner's choice stays.
- Patch 0005 (`patches/next/0005-...patch`): the Q4_0 8x8 kernels of 0002 for AVX2 builds without AVX512-VNNI
  (the alderlake variant was left on upstream's lut kernels), with `vpdpbusd` and the "-8" folded into the
  accumulator seed. Bit-exact (kernel-bench check; gemma.json texts identical to the shipped library letter by
  letter). `scripts/kernel-bench/run-windows.ps1` on one P-core thread: GEMM 3.08 -> 1.53 ns per row-block
  (2.0x), GEMV 5.4 -> 2.6 ns per block, 3 rows in one call 16.1 -> 5.3 ns. Dev suite ram-6g (6/20 threads): shipped
  and rebuilt-unchanged 165-172 tok/s, VNNI 235-238; chat with MTP 12.6 -> 22.7 (default 2 / 0.75) / 26.1 (2 / 0)
  / 27.0 (3 / 0) tok/s. The library now lists `REPACK_Q4_0_ROWS`, so the CPU MTP default applies; its p_min
  changed from 0.75 to 0.5 (`plan.CPUMTPDraftPMin`).
- Where prefill time goes (a debug per-op timer in the CPU backend, not kept): Q4_0 mat-mul 87% of a prefill graph,
  flash attention 6%, everything else under 3% each; decode graphs 94% mat-mul. With 6 threads the VNNI GEMM gives
  218 tok/s, with 20 threads 237: the package is power-limited, so more threads add little.

- Overhead outside the matrices (plan item 0.8): the NanoLLM dev bench decodes at 13.4-14.0 tok/s at depth 0 against
  13.0-13.7 for `llama-bench` on the same library, i.e. the Go side (logits copy, sampler, purego) costs at most 3%;
  no OPT-12 work is warranted. Decode graphs spend 94% of their time in the Q4_0 mat-mul.

## 3. iGPU (Vulkan, uma-4g)

- Vulkan environment knobs (`GGML_VK_FORCE_MMVQ`, `DISABLE_MMVQ`, `DISABLE_F16`, `PREFER_HOST_MEMORY`,
  `MAX_NODES_PER_SUBMIT`, `DISABLE_FUSION`, `DISABLE_INTEGER_DOT_PRODUCT`): none beats the default (pp 434-446,
  tg 9.5-9.9); `DISABLE_F16` costs 5x of the prefill, `DISABLE_INTEGER_DOT_PRODUCT` 3x.
- `n_ubatch` 256 / 512 / 1024 / 2048: pp 428 / 428 / 409 / 393: 512 stays.
- MTP draft_max 1-4 x p_min 0 / 0.5: chat 13.7-17.7 tok/s, flat around draft_max 2-3; n-gram (2, 3, 4): 10.7-10.9 on
  chat, no use. The defaults stay.
- Memory: the process's private bytes **include** the Vulkan buffers on this UMA device (private 3.65 GiB = 3.2 GiB GPU
  shared + 0.45 GiB of the process), so the plan of uma-4g (3.45 GB) matches the "private bytes" definition of the
  6 GB limit; the host overhead is about 0.25 GiB larger than `plan.HostOverhead`, not worth a change.
- Hybrid "prefill on the iGPU, decode on the CPU" (llama-bench, `-ngl 0 -dev Vulkan0 --repack 0`): pp 360 (CPU 121,
  Vulkan only 295 on the same libllama), but decode 4.1 tok/s because the CPU weights must stay in plain Q4_0 for
  the offload (repacked weights are not offloaded): the one-copy hybrid is a net loss. Two copies (Vulkan buffer +
  repacked CPU weights) would need ~6.4 GB of private memory, over the 6 GB limit, and would give ~87 s on gemma.json
  against 101 s for the iGPU alone only with the CPU MTP of section 2: parked for budgets of 8 GB and more.
- Mat-mul efficiency of the Vulkan prefill kernels: 2.9 TFLOPS-equivalent (int8 MMQ, medium tile 64 x 64, 128
  threads, subgroups of 8; the large tile is off because the device offers 32 KiB of shared memory) of ~8.6 TOPS
  peak (96 EU x 32 MAC x 1.4 GHz). Tile experiment: a dev build of `ggml-vulkan.dll` (Windows, 17-66 s per build with
  `scripts/build-libs-windows.ps1 -Vulkan`) with the warptile of the Q4_0 int8 MMQ pipelines overridable by an
  environment variable (`vk-dev-tiles.patch`, not shipped), llama-bench pp512 / pp2048 on the patched libllama,
  `raw/vk-grid.txt`: default 474-476 / 450; subgroup 16 or 32, 64-thread blocks, TM x TN 4x4: 472-477 / 447-452;
  TM x TN 4x2: 386; blocks of 256 or 512 threads with tiles 128 x 64 / 64 x 128 / 128 x 128: 402 / 426 / 325-330;
  BK 64: 472. The default tile is at the optimum of what the host-side parameters allow; more would need a new GLSL
  kernel for the prefill, which is not worth the risk (the same device spends 83% of the prefill in these
  kernels, so even +10% of them is +8% of the prefill, ~3 s of the 101 s gemma.json run).
- Decode on the iGPU, second look (after the user's objection that there must be room): a plain-sum Vulkan shader
  (`vkbw/`) reads at most **34 GB/s** on this GPU whatever the workgroup size, chunk or load width, so the decode
  ceiling is 12.6 tok/s, not the CPU's 37 GB/s. The profile (`GGML_VK_PERF_LOGGER`) showed the Q4_0 mat-vec kernels at
  28.6-34 GB/s, the worst on the shapes with more than 8192 rows (FFN gate and up, the 262144-row head), which
  ggml-vulkan sends to the 16-lane variant. Patch 0006 (`patches/llama.cpp`): the 64-lane variant for every row count
  and, in the batch kernels (2-8 columns, q8_1: the verification of MTP drafts), the 64-lane variant with 4 rows per
  workgroup instead of the pinned 16-lane one with 2 (`vk_dmmv.ps1`, `vk_mmvq.ps1`, `vk_ab.ps1`; interleaved, 3 rounds,
  upstream -> patched, tok/s): tg32 10.38-10.53 -> 11.74-11.78 (+12%), pp2 20.1-20.9 -> 23.0-23.4, pp3 29.8-30.2 ->
  34.2-34.7 (+14%), pp4 36.7-36.9 -> 44.0-45.3 (+21%), pp6 46.4-47.1 -> 60.8-62.6 (+31%); pp512 unchanged. Rows per
  workgroup 1 and 2 are worse (pp3 25 and 31), 8 and 16 worse than 4. End to end (gemma.json, uma-4g, interleaved):
  MTP-2 100.7 / 100.9 s upstream against 93.8 / 99.1 s patched, MTP-3 109.7 against 98.8 s; the time spent decoding
  drops 5-11%, the 40 s of prefill do not move. One surprise on the way: the first run with MTP showed no gain at all
  because the verification of 3 tokens goes through the q8_1 kernels, which the first change did not touch (the profile of
  a round: verification 107 ms of GPU time, two drafts 2 ms each).
- Attention on the iGPU, the third look (afternoon of 2026-10-21; a stray `find` from an earlier session held one core
  until 14:00, so the absolute numbers before then are a little low, the interleaved comparisons stand). Decode per
  token at an empty context (profile): mat-vec 78 ms at 33-34 GB/s, flash attention 3.6 ms, norms 4.6 ms, the rest
  ~1 ms; the batch kernels after 0006 make the verification of 3 tokens cost 87 ms against 85 for one. Prefill, though,
  falls with the depth of the context, and only with flash attention: llama-bench pp512 at 0 / 2048 / 4096 / 8192 tokens
  of context 475 / 386 / 331 / 252 with it, 462 / - / 409 / 355 with `-fa 0` (`vk_depth` runs in `raw/vk-depth*.txt`).
  The scalar flash-attention shader is the slow path on this GPU, and none of its host-side parameters helps
  (`vk-dev-fa.patch`, `vk_fa.ps1`, `raw/vk-fa-pp1.txt`: block rows 16 / 32, block columns 64, workgroup 256, D split 4
  / 16, row split 1, subgroups 16 / 32, shared-memory staging: 140-310 tok/s at 4096 against 302-324 for the
  default). Patch 0007 (released in b11600-n7): `LLAMA_FLASH_ATTN_TYPE_SMALL_BATCH`, flash attention for batches
  of up to 8 tokens and on the sliding-window layers, the matmul attention for longer batches on the full-attention
  layers, the KV cache in the flash-attention layout. llama-bench pp512 (`raw/lb-0007*.txt`, `raw/vk-attnmm.txt`):
  +9-14% at 1024-2048, +23% at 4096, +42% at 8192, even at an empty context, decode unchanged; the matmul attention on
  the sliding-window layers too is worse than flash attention (the full-size SWA cache is computed in full). KLD on
  Vulkan against flash attention everywhere (`kld_fa.ps1`): 0.00113 (99.9% 0.030), below `-fa 0` (0.00132) and the
  CPU backend (0.00238).
- gemma.json with flash attention on and off (`run_famatrix.ps1`, the n6 libraries with the 0007 llama.dll, two
  interleaved rounds; the first CPU run with flash attention ran on a hot machine and is left out):

  | | wall, s | prefill, s (tok/s) | decode, s (tokens, tok/s) | peak private, GB |
  |---|---|---|---|---|
  | CPU ram-6g, flash attention on | 125.5 | 81.0 (195) | 44.3 (835, 18.9) | 6.29 |
  | CPU ram-6g, flash attention off | 131.4-133.7 | 86.2-87.9 (185-189) | 45.1-45.6 (980, 21.5-21.8) | 6.07 |
  | iGPU uma-4g MTP-2, flash attention on | 94.7-95.7 | 39.7-40.4 (391-398) | 54.9-55.2 (845, 15.3-15.4) | 3.97 |
  | iGPU uma-4g MTP-2, flash attention off | 160.1-163.3 | 40.7-42.4 (373-388) | 119.3-120.8 (984, 8.1-8.2) | 4.32-4.35 |
  | iGPU uma-4g MTP-2, small (0007) | **89.0-89.1** | **35.5-35.8 (441-446)** | 53.1-53.4 (830, 15.6) | 4.08 |

  The answers differ between the attention paths (another float order flips near ties), so the token counts differ:
  compare the rates. On the iGPU flash attention must stay for decoding: without it the 35 sliding-window layers attend
  over the whole full-size SWA cache at every token. On the CPU flash attention costs the decode 13% (18.9 against
  21.8 tok/s with the same MTP) and saves 6% of the prefill. The cause (profiled below): the CPU's flash attention for
  1-3 tokens walks K and V once per query head (no sharing across the 4 heads of a KV head), converting every row from
  f16 and accumulating V in f16; the masked cells of the sliding-window layers are skipped one by one, cheaply.
- That kernel change, tried and not shipped (`cpu-fa-gqa.patch`, `cpu_fa_gqa.ps1`, `cpu_fa_prof.ps1`, `kld_cpu_fa.ps1`,
  `run_cpufa*.ps1`). A per-op timer in the CPU backend (llama-bench, 6 threads, 4096 tokens of context) put flash
  attention at 12.5 ms per decoded token against 7.5 ms for the matmul attention, at 14 against 8 ms for a 3-token
  verification; the K/V reads are ~3 ms, so the kernel is compute-bound (each query head converts every K and V row
  from f16 and accumulates V in f16). The candidate walks the cache once per KV head for all the query heads and
  tokens of a batch of up to 8, in tiles of 32 positions (scores, one softmax update per tile, V accumulated by a small
  GEMM in f32). llama-bench against the n6 kernel: tg32 at 8192 +16-17%, pp3 (the verification shape) at 8192
  +39-47%, at 4096 0-15%. KLD on decode shapes (llama-perplexity -ub 1 / -ub 3, against the n6 kernel): within the
  noise of the project's rule only with Q rounded to f16 as the row path does (mean 0.0011, p99 0.0083, top-1 98.9%;
  with f32 Q the p99 was 3.5x the noise). gemma.json (prompts up to 4.6k tokens) did not move: 133.7 against 133.8 s
  (both CPU libraries built locally, order A B B A). And it breaks `TestNativePromptCacheRoundTrip`: the tiles and the
  per-thread chunks follow the cache cells, so a prefix restored into other cells rounds differently and a near-tie
  of the greedy continuation flips (the n6 kernel keeps it). To ship it, the tiles would have to be cut over the
  unmasked positions, not over the cells. A side finding: the CPU backend built locally (VS clang 22) prefills 4-5%
  slower than the CI build of the same source (clang 20.1.8): compare libraries from the same toolchain.
- CPU and iGPU reading memory at once (`run_bwmix.ps1`, `raw/bwmix.txt`): membw with 4 / 6 / 8 / 14 threads gets
  11.5 / 18.2 / 20.5 / 25.8 GB/s while the iGPU's shader drops from 33-34 to 22 / 20 / 17 / 13; the sum is 34-39 GB/s,
  what the CPU reaches alone. The memory controller is the limit, so splitting a decode between the CPU and the iGPU
  cannot raise its bandwidth.

## 4. Disk (streaming modes)

See section 1 for the raw speeds. The mmap path of the file-backed modes reads at 1.1 GB/s from one faulting
thread, 2.3 times below `PrefetchVirtualMemory` and 3.8 times below overlapped reads with a queue depth of 4.

The layer streamer (OPT-8) now exists on Windows (`internal/memlock/stream_windows.go`: `PrefetchVirtualMemory` ahead,
`VirtualUnlock` behind; the view of the model is found by walking the address space for `MEM_MAPPED` regions named
like the file; tests in `stream_windows_test.go`). What this machine can and cannot show:

- Cold first pass (`run_stream.ps1`: the model is copied with `robocopy /J`, so none of its pages is cached; a fresh
  process runs `bench --slow`, preset ram-1g / ram-2g): `stream_ahead` 0 / 2 / 4 give pp 72.8 / 69.7 / 70.4 tok/s
  (ram-1g) and 137 / 136 / 132 (ram-2g), tg 7.5-7.9 and 9.5-9.9: no difference outside the 10-20% run-to-run noise,
  because the prefill of M9 (plain Q4_0, 58-70 tok/s) computes for 7-8 s and the whole file is read from the NVMe in 4 s
  in parallel.
- `stream_evict=on` (trim behind) halves the decode when the cache is plentiful: tg 3.5 vs 7.7 (ram-1g), 6.1 vs
  8-9.6 (ram-2g), because every token faults the trimmed pages back into the working set. As on Linux it is for
  machines whose RAM is really short, and stays off by default.
- The steady state of a streamed decode cannot be measured honestly here: with 32 GiB the standby list holds the 4 GiB
  model after the first token, so a "disk" decode is a RAM decode (7.5-9.9 tok/s, the unrepacked CPU speed). Pushing
  the cache out needs an administrator (`NtSetSystemInformation(MemoryPurgeStandbyList)`) or a machine with little RAM;
  `tools/memhog` takes memory away but the commit limit of this machine (64 GB, 38 GB used) lets it leave only
  ~9 GB available. The next step for the streaming modes on Windows is OPT-9 (overlapped unbuffered reads at queue depth
  4 into a ring: 4.2 GB/s against 2.5 for prefetch and 1.1 for faults), measured on such a machine.

## 5. Final runs: the best setting of each backend

gemma.json once per backend, a minute apart, on a quiet machine (`run_final.ps1`, `gemma_table.py`; the n6 libraries
with the 0007 llama.dll, which is what b11600-n7 ships). Input = prompt tokens (of them from the prompt cache), time =
the whole request, peaks from the `mem.json` sampler of the run.

CPU, preset ram-6g, planner defaults (6 / 20 threads, MTP 2 / p_min 0.5, flash attention on):

| # | Input, tokens (cached) | Output, tokens | Time, s | Prefill, tok/s | Decode, tok/s | MTP acceptance |
|---|---|---|---|---|---|---|
| 1 | 4365 (0) | 149 | 31.5 | 199 | 15.6 | 0.63 |
| 2 | 4625 (132) | 107 | 31.8 | 194 | 12.3 | 0.44 |
| 3 | 4572 (132) | 124 | 31.2 | 192 | 15.2 | 0.59 |
| 4 | 2010 (132) | 111 | 15.6 | 201 | 17.7 | 0.65 |
| 5 | 726 (86) | 344 | 18.3 | 194 | 22.9 | 0.74 |
| all | 15816 new | 835 | **128.4** | 196 | 17.6 | |

Peak working set 6.02 GB, peak private bytes 6.29 GB, no GPU memory.

iGPU, preset uma-4g, MTP 2, flash attention `small` (the planner's choice on this device):

| # | Input, tokens (cached) | Output, tokens | Time, s | Prefill, tok/s | Decode, tok/s | MTP acceptance |
|---|---|---|---|---|---|---|
| 1 | 4365 (0) | 124 | 19.1 | 448 | 13.2 | 0.48 |
| 2 | 4625 (132) | 107 | 20.4 | 435 | 10.6 | 0.31 |
| 3 | 4572 (132) | 131 | 20.3 | 435 | 13.0 | 0.44 |
| 4 | 2010 (132) | 124 | 12.6 | 451 | 14.6 | 0.42 |
| 5 | 726 (86) | 344 | 17.9 | 455 | 20.9 | 0.69 |
| all | 15816 new | 830 | **90.1** | 441 | 15.3 | |

Peak working set 6.57 GB, peak private bytes 4.08 GB (the Vulkan buffers included), GPU shared 3.55 GB; the rest of the
working set is file-backed pages of the mapped model, which the 6 GB limit does not count.

Where the phase leaves the two backends: the CPU 213 -> 128 s (threads for the batch, patch 0005, MTP on the CPU),
the iGPU 101 -> 90 s (patches 0006 and 0007). The iGPU wins by its prefill (2.2x); the CPU decodes faster (17.6
against 15.3 tok/s, with higher MTP acceptance). Decode on both is at the memory controller's ceiling (CPU 37 GB/s,
shader 34 GB/s, both together no more); what is left is prefill on the CPU (compute) and the acceptance of the drafts.
