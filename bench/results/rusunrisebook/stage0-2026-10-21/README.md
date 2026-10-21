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
  kernel, which is not worth the risk for a few percent (the same device spends 83% of the prefill in these
  kernels, so even +10% of them is +8% of the prefill, ~3 s of the 101 s gemma.json run).
- Decode on the iGPU: the Q4_0 mat-vec kernels read 29-34 GB/s (gate/up 29, down 34, head 28.6); at the best shape
  all of them would give +6-8% of the token. Not worth a kernel either.

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
