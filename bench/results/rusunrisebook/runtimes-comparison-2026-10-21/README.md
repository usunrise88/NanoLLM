# NanoLLM baseline against other runtimes: Gemma 4 E4B on the i7-12700H and Iris Xe

Taken 2026-10-21 on the same machine and driver as `../baseline-2026-10-20-gemma-igpu-uma4g/`
(i7-12700H, 32 GiB DDR4-3200 dual channel, Iris Xe, Intel driver 32.0.101.7092, AC power, plan
Performance). The NVIDIA GPU was not used. One run per configuration unless a row says otherwise: the aim
was to see how good the baseline is, not to measure a few percent.

The workload is the five requests of `gemma.json` (prompts of 4.4k, 4.6k, 4.6k, 2.0k and 0.7k tokens,
temperature 0, one after another), and for a one-tool comparison `llama-bench` (pp512, tg128) on the same GGUF.
Every runtime writes its own text, so the totals depend on how long each answer turned out; the rates per
token are the fairer figure.

## Verdict

- **iGPU: our Vulkan is the best of what could be run.** Same tool, same GGUF: Vulkan pp512 294 / tg128 10.3,
  OpenVINO 290 / 7.7, SYCL 150 / 9.3 tok/s. On the real workload SYCL inside NanoLLM is 1.6x slower than Vulkan
  once its kernels are compiled (the first request took 258 s). OpenVINO loads in NanoLLM but fails on the first
  request (see below).
- **CPU: our patched backend is the best of what could be run, except LiteRT-LM on prompt processing.**
  Real workload, ram-6g: 225-230 s against 312-341 s of ik_llama.cpp. With the same `llama-bench` the patches give
  prefill x1.6-1.7 (115 against 69 tok/s on 6 P-cores, 123 against 74 on 14 threads) and leave decode unchanged
  (13.4 tok/s). LiteRT-LM prefills faster on the CPU (157-182 tok/s at a 1024-token prompt against our 115-123 at
  512), and decodes slower (8-9 against 13.4).
- **Decode on this machine is faster on the CPU than on the iGPU** (13.4 against 10.3 tok/s in `llama-bench`),
  while prefill is 2.5x faster on the iGPU (294 against 115-123). With about 2.6 GB of weights read per token,
  the CPU reaches roughly 35 GB/s and the iGPU 27 GB/s of the 51 GB/s peak of the memory; neither gets to the
  "about 19.6 tok/s" ceiling worked out earlier, the CPU comes closest.
- **Not testable one to one: ORT GenAI** (no E4B model it can load that is 4-bit for CPU or DirectML; see below).

## Real workload: gemma.json, five requests, one run

| Runtime | Device | Total wall, s | Out tokens | Prefill, tok/s | Decode, tok/s | Peak working set / private |
|---|---|---|---|---|---|---|
| NanoLLM, Vulkan, MTP-2 (the baseline, mean of 2) | iGPU | **106.2** | 849 | 375-430 | 9.3-17.2 | 6.0 / 3.7 GiB (GPU 3.2 GiB inside) |
| NanoLLM, Vulkan, MTP-3 (baseline, mean of 2) | iGPU | 111.8 | 964 | 383-433 | 9.0-18.8 | 6.0 / 3.7 GiB |
| NanoLLM, SYCL, MTP-2 | iGPU | 407.5 (129.7 after the first request, against 82.3 on Vulkan) | 808 | 17 (compiling), 156-208 | 6.3-15.1 | 7.0 / 4.6 GiB |
| NanoLLM, OpenVINO (GPU.0) | iGPU | fails: HTTP 500 on the first request | - | - | - | 12.2 / 8.5 GiB |
| NanoLLM, ram-6g, MTP-2 | CPU | 230.3 | 974 | 101-108 | 9.4-16.9 | 5.6 / 5.8 GiB |
| NanoLLM, ram-6g, no MTP | CPU | 225.3 | 825 | 102-110 | 9.7-12.5 | 5.4 / 5.6 GiB |
| ik_llama.cpp, no MTP | CPU, 6 P-cores | 340.9 | 1005 | 49-81 | 8.3-8.9 | 5.4 / 2.7 GiB |
| ik_llama.cpp, MTP-2 | CPU, 6 P-cores | 312.1 | 848 | 62-76 | 8.9-12.1 | 8.5 / 5.5 GiB |
| ik_llama.cpp, MTP-3 | CPU, 6 P-cores | 361.4 (the first request 122 s to the first token, a cold start) | 848 | 36-81 | 9.4-16.3 | 8.5 / 5.5 GiB |

`comparison-table.txt` has every request. The NanoLLM CPU runs used the shipped defaults of that day: 6 threads (hwinfo counts only the 6 P-cores on a
hybrid CPU; the earlier text said 14), no pinning. The `llama-bench` rows with 14 threads below are a separate check
of the thread count, not the NanoLLM default. Correction of 2026-10-21: the 14-thread `llama-bench` prefill is +7%,
and the stage 0 sweep (`../stage0-2026-10-21/README.md`) found 20 batch threads +35% on the quiet machine.

## One tool: llama-bench b11600, pp512 and tg128, Gemma 4 E4B Q4_0 GGUF

| Backend libraries | pp512, tok/s | tg128, tok/s | Runs |
|---|---|---|---|
| Vulkan, iGPU (upstream asset, which is also ours) | 293.7 | 10.25 | 1 |
| OpenVINO, iGPU | 290.0 | 7.65 | 1 |
| SYCL, iGPU | 149.6 | 9.27 | 1 |
| CPU upstream, 6 P-cores pinned | 69.0 (the first run 60.4, a cold start) | 13.4 | 3 |
| CPU with our patches, 6 P-cores pinned | 114.7 | 13.4 | 3 |
| CPU upstream, 14 threads | 73.8 | 13.6 | 3 |
| CPU with our patches, 14 threads | 123.0 | 13.7 | 3 |

Which of the patches 0002-0004 gives the prefill gain on this CPU (0002 is documented as meant for AVX2 without
VNNI) was not isolated. A first single run had put the patched build at 7.7 tok/s decode on 6 P-cores; three
repetitions showed it was noise.

## Other runtimes

- **ik_llama.cpp** (master 565f08d, built by me with MinGW gcc 15.2 and `-march=native`, VNNI in the binary: 8541
  `vpdpbusd`). Gemma 4 works on the CPU; its Vulkan backend is not supported for an iGPU (their README and a
  `wontfix` issue), so only the CPU was tested. Its own `llama-bench` gives 86.8 / 8.8 on 6 P-cores. Adding E-cores
  made it worse (14 threads: 6.2 tok/s decode); `-rtr`, no flash attention and no mmap changed nothing. The build is
  a caveat: decode of 8.8 against 13.4 for upstream llama.cpp on the same cores may be the toolchain (libgomp on
  Windows hybrid cores), not the project. No MSVC or clang was available.
- **LiteRT-LM 0.18.1** (`pip install litert-lm`, model `gemma-4-E4B-it.litertlm`, 3,659,530,240 B). Its own
  `benchmark`, 1024 tokens of prefill and 256 of decode: CPU on 6 P-cores 182 / 8.1 tok/s, CPU with default
  threads 157 / 9.2, CPU with MTP 178 / 4.5 (slower). Its GPU backend is Dawn on Direct3D 12 and picked the RTX
  3070 Ti (55.7 tok/s decode, 590 prefill); hiding the NVIDIA Vulkan driver does not change that and there is no
  adapter setting, so no iGPU figure exists. Those GPU figures are not comparable and were dropped.
- **ORT GenAI 0.18.0**: its model builder does not handle E4B. The only GenAI-format E4B is an fp16 export of 17.5 GB
  (4x our Q4 and three times the 6 GB limit); the int4 variants are CUDA-only; the DirectML package (0.14.1) is older
  than the per-layer-embedding support (0.15). Not tested.
- **llama.cpp SYCL and OpenVINO backends inside NanoLLM.** The release b11600 carries both as
  `ggml-sycl.dll` and `ggml-openvino.dll` (zip assets of 149 and 99 MB). Dropped next to our patched libraries
  (a copy of `dist/windows_amd64/lib`) they load without changing NanoLLM and show up as `llamacpp:sycl0` and
  `llamacpp:openvino0..2`. OpenVINO needs `GGML_OPENVINO_DEVICE=GPU.0` (it does not follow the device name). In
  NanoLLM its first request fails with an OpenVINO `Reshape` error ("Cannot infer '-1' dimension with zero-size
  output dimension"); upstream `llama-bench` runs with it. The cause was not found; a guess is patch 0001 (it drops
  rows after the last KV-owning layer, which can give a zero-size tensor), not checked.

## Caveats

- One run per configuration, apart from the baseline (two) and the CPU `llama-bench` (three).
- Outliers that were kept out of the readings but are in the raw files: the SYCL first request (kernel compile,
  258 s), the ik_llama.cpp MTP-3 first request (122 s), the cold first `llama-bench` run (60.4 tok/s).
- Models and quantisation differ between runtimes (the same GGUF for NanoLLM, ik_llama.cpp and `llama-bench`;
  the `.litertlm` file for LiteRT-LM), and the texts differ, so the number of generated tokens differs.
- The first ik_llama.cpp attempt left the model's reasoning on and wrote 750-1240 tokens per answer; it was redone
  with `--chat-template-kwargs '{"enable_thinking":false}'` and only that run is kept.

## Files

- `comparison-table.txt`: every request of every `gemma.json` run, the rates and the similarity to the reference.
- `raw/`: server logs, answers, memory samples, `llama-bench` and LiteRT-LM outputs.
- `scripts/`: the scripts that made them. The libraries of the SYCL and OpenVINO runs were built by copying
  `dist/windows_amd64/lib` and adding the backend DLLs and their runtimes from the b11600 release zips.
