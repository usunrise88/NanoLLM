# `nanollm optimize`, stage 0: the measurements the design waited for (2026-10-23)

The questions of docs/OPTIMIZE.md, section 6, answered on the laptop (i7-12700H, 32 GiB, RTX 3070 Ti Laptop 8 GiB with
the desktop on it, Iris Xe, NVMe 3295 MiB/s). Gemma 4 E4B QAT with its MTP drafter; a build of the working tree
(`nanollm-opt.exe`, llama.cpp b11600) over the release v0.1.0-rc.1 libraries with the CUDA 13 add-on (`D:\n8rc`); the
CPU, the iGPU and the RTX never ran at the same time, the GPU started cool (<= 70 C), one pass per measurement.
`run-stage0.sh` drives the bench runs; the `calibrate-*.json` files are `nanollm optimize calibrate --json`.

Memory is what the plan promised the owner: the private bytes of the process (`PeakPagefileUsage`, the working-set
counters sampled every 100 ms in the worker) plus, on a discrete GPU, the process's dedicated video memory from the PDH
counter `\GPU Process Memory(pid_*)\Dedicated Usage`. The page cache and the commit charge are not counted (the commit
of a CUDA process includes its video memory under WDDM).

## 1. M3 against M2 on CUDA, and whether optimal fits 4 GiB (`m2-m3-cuda.json`, `cuda-vram8.json`)

vram-4g, 8k context, 3 repetitions, same card, same cool start:

| Mode | pp, tok/s | tg, tok/s | MTP, tok/s | Private | VRAM dedicated | VRAM shared (PDH) | Total |
|---|---|---|---|---|---|---|---|
| M2 vram-ple-host | 5484 | 95.3 | 168.0 | 2.54 GiB | 3.34 GiB | 2.04 GiB | 5.88 GiB |
| M3 vram-ple-disk | 5645 | 100.8 | 166.8 | 0.68 GiB | 3.34 GiB | 0.17 GiB | **4.02 GiB** |

- **M3 is as fast as M2** (+3 % pp, +6 % tg, MTP within 1 %), so under a memory cap the planner takes M3: it holds
  the per-layer embeddings in the file (read through the 0.7 GiB it keeps) and not pinned in host memory. Decision:
  rule (a) of the plan is on; M2 is skipped when a cap is set. Without a cap nothing changes.
- **The optimal mode fits.** M3 at 8k context holds 4.02 GiB, under the limit of 4 GiB + 200 MiB. The plan said
  3449 MiB of video memory, the process holds 3424 MiB (-0.7 %): `GPUOverheadCUDA` stays 320 MiB.
- **Host memory of the driver.** A process with a GPU context holds ~448 MiB of private memory beyond the plan (the
  driver, cudart, cuBLAS, the graphs): 695 MiB measured against the 223 MiB the plan has for the host on CUDA in M3
  (the same excess in M2), and on the Iris Xe through Vulkan, M3 on unified memory, 3.65 GiB measured against the plan's
  3.21 GiB (+445 MiB, section 3). `plan.GPUHostMemory` adds 448 MiB to the combined check for a CUDA or Vulkan device;
  on the Iris it moves the optimal mode's plan from 32k to 16k context (4.01 GB planned against the 4.40 GB cap).
- **vram-8g does not fit next to the desktop** (it holds 3 GiB): pp 267, tg 65, MTP 33 tok/s, 0.68 GiB of shared
  memory against the 0.27 GiB of host memory its plan has. That spill is what the memory check must catch (below).

## 2. PDH counters and the spill rule

- The counters work for an ordinary user and for the process itself (no administrator rights). They are read at the
  checkpoints (after the load, after the prefill, at the end of decode), not continuously.
- M2's Shared Usage is its pinned host memory (2.04 GiB; the plan has 2.08 GiB of host memory); M3's is 0.17 GiB (the
  plan 0.22 GiB), M1 at vram-8g's is 0.68 GiB (the plan 0.27 GiB). **Rule: `vram_spill` when Shared exceeds the
  plan's host memory (without the driver's 448 MiB, which is not shared memory) by more than 256 MiB** (not
  "Shared > 0", which would flag M2 and every CUDA process). The three runs above: M2 -47 MiB, M3 -47 MiB, vram-8g
  +424 MiB, the one whose prefill fell to 267 tok/s.
- Not measured: PDH from LocalSystem (the installed service). It needs the service, which needs administrator rights
  to install; the fallbacks (free-memory difference of the device, then the backend's buffers plus overhead) are in
  the code and the source of the figure is written in the profile.

## 3. The plan against the private bytes (`cpu-ram4.json`, `iris-uma4.json`, `silent-ram-*.json`)

| Run | Planned | Measured |
|---|---|---|
| CPU ram-4g (anonymous memory), M7 | 3.28 GiB of RAM | 3.30 GiB private (+0.6 %) |
| CPU ram-3g, M7, 2 threads | 2.94 GiB of RAM | 2.96 GiB private (+0.7 %) |
| RTX vram-4g, M3 | 223 MiB host + 3449 MiB video | 695 MiB private (+472 MiB) + 3424 MiB video (-0.7 %) |
| Iris uma-4g, M3 (unified memory) | 3.21 GiB | 3.65 GiB private (+445 MiB; the shared 3.31 GiB is the same memory: not added) |

The CPU plan is within 1 % of the measurement; a GPU context adds ~450 MiB of host memory that the plan did not have
(`plan.GPUHostMemory`). The 200 MiB tolerance of the optimal mode is spent half on the planner (it plans to
4 GiB + 100 MiB) and half on the error of the plan.

## 4. The quiet mode (`silent-ram-*.json`, `calibrate-cpu-t2*.json`)

Two threads, ram-4g down to ram-1g:

| Budget | Mode | pp, tok/s | tg, tok/s (empty context) | Private |
|---|---|---|---|---|
| ram-4g | ram-ple-disk, 8k | 92.4 | 11.0 | 3.30 GiB |
| ram-3g | ram-ple-disk, 4k | 90.3 | 11.3 | 2.96 GiB |
| ram-2g | ram-hybrid | 56.8 | 4.6 | 1.71 GiB |
| ram-1.5g | ram-hybrid | 34.1 | 3.7 | 1.21 GiB |
| ram-1g | mmap | 23.2 | 2.7 | 0.60 GiB |

- On this NVMe every rung of the ladder is above 1 tok/s, so **the quiet mode here takes ram-1g: 0.6 GiB of private
  memory at 2.7 tok/s.** The plan of ram-1g (M9) reads 2.08 GiB of the file for every token; the disk formula
  (`DiskTokS = disk MiB/s / cold MiB per token`) gives 3295 / 2131 = 1.5 tok/s and the measurement says 2.7: the
  formula is a lower bound (part of the cold bytes is in the page cache). On a 130 MB/s disk it says 0.06 tok/s, so a
  slow disk keeps the quiet mode at ram-3g/ram-4g (no streaming).
- **Two threads decode as fast as six** (11.25 against 11.1 tok/s at an empty context; the decode waits for memory,
  `calibrate-cpu-t2.json`, `calibrate-cpu-t0.json`); prefill follows the threads (86 against 202 tok/s).
- **EcoQoS and the lower priority cost nothing on an idle machine** (`calibrate-cpu-t2-low.json`, below normal
  priority + EcoQoS, 2 threads): pp512 96.7 against 86.0, tg 11.6 against 11.25, tg after 4096 tokens 8.2 against 8.5
  (noise). It yields to other programs under load, which is what the mode is for.
- Decode after the 4096-token prompt is 8.2-8.5 tok/s at 2 threads at ram-8g: the 1 tok/s threshold is not near for
  a resident model. The streaming rungs are measured at depth by the probe, with the disk formula as its forecast.

## 5. The reference prompt and the depth factors (`calibrate-*.json`)

A prompt of ~4000 tokens of our own text, the question before the text (every prefix is a valid prompt), the hash
frozen in `refprompt.txt`'s test. pp is the average over the whole prompt, tg the rate after it:

| Device | pp512 | pp over 4096 | depth_pp | tg at 0 | tg after 4096 | depth_tg | MTP gain at depth |
|---|---|---|---|---|---|---|---|
| CPU, 6 threads (default) | 202 | 174 | 0.86 | 11.1 | 9.6 | 0.87 | 1.91 (mtp3/0) |
| CPU, 7 threads | 158 | 148 | 0.94 | 13.4 | 10.3 | 0.77 | 1.89 |
| CPU, 2 threads | 86 | 79 | 0.92 | 11.25 | 8.5 | 0.76 | 1.48 (mtp2/0.5) |
| Iris Xe | 449 | 416 | 0.92 | 11.3 | 9.1 | 0.80 | 1.89 (mtp3/0) |
| RTX 3070 Ti | 5693 | 6351 | 1.12 | 100 | 96.5 | 0.96 | 1.87 (mtp3/0) |

- The factor that carries a probe at 512 tokens to the whole prompt is within 9 % across thread counts of one CPU and
  differs by up to 30 % between device classes (the RTX gets faster per token with a longer batch), so the baselines
  carry them per class and the probe measures the full prompt when its forecast is <= 25 s.
- **The error of carrying a 512-token probe over to the whole request** (section 7 of the spec asks for <= 10 % on T, the
  time of the typical answer), with the baseline's factors of the 6-thread run (0.86 / 0.87) applied to the other thread
  counts: 7 threads predict T = 55.8 s against 56.8 s measured with the full prompt (-1.6 %), 2 threads 86.0 s against
  87.0 s (-1.1 %), although the pp and tg parts are off by 7-15 % in opposite directions; the 6-thread, Iris and RTX runs
  use their own factors and are exact by construction.
- The 7th thread (the first on an E-core) costs the 12700H 22 % of prefill (158 against 202); its decode reads 13.4
  against 11.1 tok/s, a +21 % that the 2-thread run (11.25) says is the noise of one pass. "Half of the physical
  cores" is 7 here and the optimal mode keeps it as the rule; a thread-count probe on a hybrid CPU is worth its load
  and is part of the powerful mode's candidates.

## 6. A CUDA kernel-shape switch after the graphs are captured

`TestNativeCudaNanoAtRuntime` (native, `NANOLLM_TEST_DEVICE=cuda0`): `MMVQ_NWARPS` set on a model that has already run
(its CUDA graphs captured) changes nothing; a model loaded with it set does. **Kernel-shape candidates therefore cost a
load each** and are outside the v1 candidate list (the owner's bound of two loads per run); the test fails if the DLL
ever starts to drop its graphs on a switch, and the probe can then try shapes on one load.

## 7. The cost of a batch against the full sweep of draft lengths

Batch times for 1, 3 and 4 tokens (ms) and the expected speed-up `E(k)` with the default acceptance against the
variant that decoded fastest after the prompt:

| Device | c(1), c(3), c(4) | E(1), E(2), E(3) | Mechanism's k | Sweep best at depth | The mechanism's pick against the best |
|---|---|---|---|---|---|
| CPU, 6 threads | 95, 108, 125 | 1.54, 1.73, 1.61 | 2 | mtp3/0 18.3 | mtp2/0 17.9: 98 % |
| CPU, 7 threads | 99, 121, 137 | 1.47, 1.60, 1.52 | 2 | mtp3/0 19.5 | mtp2/0 17.5: 90 % |
| CPU, 2 threads | 124, 167, 206 | 1.40, 1.47, 1.29 | 2 | mtp2/0.5 12.6 | mtp2/0 12.5: 99 % |
| Iris Xe | 112, 137, 143 | 1.48, 1.61, 1.64 | 3 | mtp3/0 17.2 | the same |
| RTX | 9.9, 11.5, 12.0 | 1.56, 1.78, 1.82 | 3 | mtp3/0 180.8 | the same |

The batch cost predicts the best draft length exactly on both GPUs, within 2 % on two CPU runs and within 10 % on the
third; a CPU without the fast kernels takes the mechanism's pick and one confirming measurement, a CPU or GPU with a
matrix cell measures the variants. p_min 0 against 0.5 moves the result by -4 % to +9 % at depth, in either direction
(one pass each: noise).

## 8. Findings outside the plan

- `CUDA_VISIBLE_DEVICES=-1` makes the device listing hang with the release libraries; the "CUDA add-on missing" check
  (`cuda_addon_missing`) is tested with a library directory without the add-on (`C:\Projects\runtimes\libs\n8-nocuda`),
  and `GGML_VK_VISIBLE_DEVICES=1` shows only the Iris to Vulkan.
- Switch 0006 of the llama.cpp patches (an environment switch to turn a patch off) needs a libraries release; it does
  not block v1 and is listed in docs/OPTIMIZE.md, section 9.

## Decisions

1. Rule (a) of the plan: under a memory cap every full-offload mode that fits is grown and the one with the largest
   context wins; M2 is skipped under a cap (M3 is as fast and holds less host memory).
2. Memory = private bytes (+ dedicated VRAM on a discrete GPU); `vram_spill` = PDH Shared above the planned host memory
   by more than 256 MiB; a CUDA process adds 448 MiB of host memory to the plan.
3. The probes of one load (threads, draft length, batch cost) are kept; kernel shapes are not in v1.
4. The baselines of the matrix are these calibrations (`estimated` is off for the 12700H, Iris Xe and RTX 3070 Ti;
   the EPYC's stay estimates from its 2026-10-19/20 runs).
