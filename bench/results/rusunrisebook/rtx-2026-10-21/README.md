# RTX 3070 Ti Laptop: a short check of Vulkan and CUDA (2026-10-21)

The standard suite (quick choice: the derived preset plus vram-8g and vram-4g, 3 repetitions) on the laptop's
NVIDIA RTX 3070 Ti Laptop GPU (8 GiB GDDR6, driver 581.57), not submitted. Gemma 4 E4B QAT (UD-Q4_K_XL) with its
MTP drafter; NanoLLM b11600-n7-1-ga02a725 (`dist/windows_amd64/nanollm.exe`), libraries of the n7 series.

- Vulkan: `dist/windows_amd64/lib` (our n7 libraries, our `ggml-vulkan.dll`), `GGML_VK_VISIBLE_DEVICES=0` so that the
  suite's "first GPU" is the RTX and not the Iris Xe.
- CUDA: the n7 libraries without `ggml-vulkan.dll` plus the `windows_amd64_cuda` add-on (`scripts/fetch-libs.sh b11600
  windows_amd64_cuda`: upstream's `ggml-cuda.dll` built with CUDA 12.4, cudart and cuBLAS), in
  `C:\Projects\runtimes\libs\n7-cuda`.

The desktop runs on this GPU: 1.0-2.4 GiB of its memory were taken before each run (nvidia-smi), so the derived
preset is 7 GiB through Vulkan and 6.5 GiB through CUDA, and vram-8g oversubscribes the card.

Driver: `run_rtx.ps1`; configurations `vulkan.yaml`, `cuda.yaml`; results `standard-vulkan.json`, `standard-cuda.json`,
`cuda-vram-8g-split.json` (vram-8g through CUDA again, with the fix below).

| Backend | Preset | Mode, context x slots | pp, tok/s | tg, tok/s | chat, tok/s | MTP, tok/s (acceptance) | Peak VRAM | Peak RSS |
|---|---|---|---|---|---|---|---|---|
| Vulkan | vram-7g (derived) | vram-full, 16k x 2 | 5023 | 88.2 | 87.1 | 107.7 (0.43) | 6.72 GB | 2.51 GB |
| Vulkan | vram-8g | vram-full, 32k x 4 | 3646 | 90.0 | 87.7 | 105.9 (0.43) | 7.15 GB | 2.75 GB |
| Vulkan | vram-4g | vram-ple-host, 8k x 1 | 4891 | 93.4 | 93.3 | 105.3 (0.43) | 3.31 GB | 2.54 GB |
| CUDA | vram-6.5g (derived) | vram-full, 16k x 2 | 5916 | 89.1 | 89.6 | 147.3 (0.47) | 5.41 GB | 0.71 GB |
| CUDA | vram-8g | vram-full, 32k x 4 | failed: the drafter did not load (below) | | | | | |
| CUDA | vram-8g, with the fix | vram-full, 32k x 4 | 425 | 67.1 | 66.5 | 54.3 (0.47) | 7.13 GB | 1.39 GB |
| CUDA | vram-4g | vram-ple-host, 8k x 1 | 5435 | 88.8 | 88.9 | 139.2 (0.47) | 3.32 GB | 2.70 GB |

pp has a large spread over the 3 repetitions (+-1100-2600 tok/s: the first repetition is cold); tg is within +-1-2.

- **Decode is the same on both** (88-93 tok/s); **CUDA prefills 10-20% faster** and
  **drafts better: MTP 139-147 tok/s against 105-108** (acceptance 0.47 against 0.43).
- **Output.** Vulkan on the RTX gives the same greedy 128 tokens as the CPU and the Iris Xe (sha256 97e62ed1...); CUDA
  gives another (8271256b...). The two answers agree for about 50 tokens and then take a different near-tie word
  (both read well): the numerics of CUDA's kernels, not an error.
- **vram-8g does not fit next to the desktop.** Through Vulkan the driver moves part of it to shared memory (pp
  3646 instead of ~5000); through CUDA (sysmem fallback) pp falls to 425 and tg to 67. The derived preset (what the
  server would choose) is the one to use on this machine.
- **Bug found and fixed: "invalid vector subscript" on a full CUDA device.** Without `tensor_split` llama.cpp splits
  the layers by the free memory each device reports (`src/llama-model.cpp` 1618-1660). Once vram-8g's KV cache
  (2.3 GB) had filled the card, CUDA under WDDM reported 0 bytes free with the total known (the only input that leads there); the split points became
  0/0 = NaN, `upper_bound` ran past the end and `devices.at(1)` threw, so the drafter, loaded after the target's
  context, failed. NanoLLM now passes an explicit split (the whole model on the planned device,
  `internal/backend/llamacpp/model.go`, `modelParams`), the same split llama.cpp takes for one device with memory
  to report.

## CUDA at vram-4g: where the time goes and what can move (study, 2026-10-21 evening)

Base: CUDA vram-4g (mode vram-ple-host: the per-layer embeddings and the input token_embd in pinned host memory,
the rest on the GPU, 3.3 GB VRAM, 8k context, 1 slot), standard suite pp 5435 / tg 88.8 / MTP 139 tok/s.
llama-bench on the same n7 libraries with upstream's `ggml-cuda.dll` (`C:\Projects\runtimes\bench-cuda-n7`):
pp512 6022, tg128 94.4. Raw files in `raw/`; drivers `run_dev_sweep.ps1` (waits for the GPU to cool before each
variant, runs the list forward and backward).

- **The laptop GPU throttles within a minute of decode.** It reaches its 130 W power cap at 90 % utilisation, then
  85-87 C, and the core clock falls from 1785 to ~1425 MHz (memory stays at 7001 MHz); throttle reasons 0x4 (power)
  and then 0x20 (thermal). Back-to-back variants measure the cooling: a thread sweep 1-2-4-6 read 88 -> 73 tok/s, the
  same pair in the other order 6 then 1 read 88.8 then 81.7. Every comparison here starts on a cool GPU (<= 66 C;
  the desktop keeps it at 63-65 C idle).
- **Decode is far from the memory ceiling.** 2.66 GB of weights on the GPU at 448 GB/s is ~6 ms per token; measured
  10.6 ms in llama-bench with the GPU busy 90 % of the time at its power cap: the kernels (about a thousand per token
  for Gemma 4 E4B: 368 Q4_0 matmuls through MMVQ plus the F32 element-wise ops), not launches. CUDA graphs are on and
  matter: disabling them costs 19 % (94 -> 79). A per-kernel profile needs Nsight Systems (not installed).
- **MTP** (`raw/dev-mtp-*.json`, each value twice): chat draft_max 2 / 3 / 4 / 6 = 148 / 140-146 / 137 / 108-113 tok/s
  (acceptance 0.56 / 0.47 / 0.36 / 0.26); the echo text the other way round (207 / 216-223 / 264-268 / 274-279).
  p_min 0.5 loses on CUDA (116-120 at draft_max 4-6) although it wins on the CPU: with p_min the verify batch
  changes size from step to step, and the CUDA backend keeps one graph per context, so each new shape runs eagerly
  and is captured again (ggml-cuda.cu 2726-2769). The planner's default (3, p_min 0) is right for chat.
- **Host-side sampling costs 8-11 % of MTP and 2-4 % of plain decode.** Every output row (262144 logits, 1 MiB) is
  copied to the host and scanned there: the drafter's top-10 in Go costs 0.23 ms per draft token
  (`BenchmarkTopOfTopK`), the target's chain in llama.cpp builds a 262144-entry candidate array per sampled position.
  b11600 has backend sampling (`llama_set_sampler`, greedy/top-k/top-p/min-p/dist in the graph, several outputs per
  sequence), which leaves only the token (and the top-10 for the drafter) to copy.
- **Small or no gain:** n_ubatch 1024 pp2048 +3.6 % (6405 against 6182; more compute buffer), `GGML_CUDA_GRAPH_OPT=1`
  +0.8 % (noise), flash attention off -7 % pp / -3 % tg, CPU threads for the GPU decode within noise once the heat is
  controlled (llama-bench 1-8 threads 90.5-94.6 in a heating sweep).

### Where the decode time goes on the GPU (Nsight Systems, 2026-10-21)

`raw/nsys-tg64.nsys-rep` (+ `raw/nsys-tg64_cuda_gpu_kern_sum.csv`): llama-bench tg64, CUDA graphs traced per node
(CUDA Toolkit 13.4 / Nsight Systems 2026.3 installed with `scripts/setup-windows-cuda.ps1`). 10.1 ms of kernel time
per token out of 10.6 ms: the GPU is busy, the launches are not the cost.

| Kernels | ms per token | share |
| --- | --- | --- |
| MMVQ Q4_0, gate+up fused with GEGLU (42 per token, 91 us each, ~72 % of the memory peak) | 3.84 | 38 % |
| MMVQ Q4_0, the other matrices (218 per token: q/k/v/o, down, per-layer projections, the 262144-row head) | 3.73 | 37 % |
| RMS norm, plain and fused (~235 per token, 4-5 us each) | 1.10 | 11 % |
| quantize_q8_1 of the activations (302 per token) | 0.45 | 4.5 % |
| flash attention (35 layers with head 256, 7 with head 512) + stream-k fixup | 0.47 | 4.6 % |
| mul, GELU, set_rows, get_rows, the rest | 0.5 | 5 % |

The matrix kernels read 2.66 GB in 7.6 ms (351 GB/s, 78 % of 448); the decode as a whole reaches 56 %. What is
left is model-shaped, not card-shaped: ~2.5 ms of small kernels per token (norms that could fuse into the next
quantization, the same activation quantized again for q, k and v), MMVQ efficiency on the small matrices, and the
output head (377 MB, ~1 ms, OPT-4).

### Sampling in the device graph (backend sampling), 2026-10-21

NanoLLM now samples in the CUDA graph (device feature `backend-sampling`; `models.overrides.<id>.sampling: auto |
host | device`; `bench --set sampling=host,device`): the target's chain (greedy, or top-k/top-p/min-p/temp/dist, logit
bias) and the drafter's top-10 run on the GPU, and only the token and the ten candidates come back. A grammar,
penalties or logprobs keep a request on the host. A new chain makes llama.cpp reserve the graph again (+18 ms to the
first token in `raw/dev-samp-*.json`), so a greedy chain with the same settings stays attached between requests.

`raw/dev-samp3-*.json`, cool GPU, order device / host / host / device:

| | host | device |
| --- | --- | --- |
| MTP chat, tok/s | 147.8 / 150.4 | **158.1 / 159.0** (+6 %) |
| MTP echo, tok/s | 238 / 235 | **259 / 261** (+10 %) |
| tg / chat, tok/s | 90.8-91.7 / 89.9-91.3 | 91.7-93.8 / 91.0-92.9 (+1.5 %) |
| time to first token, ms | 23.6-23.8 | 22.9-23.5 |

(`raw/dev-samp-*.json` is the first build, without keeping the chain; a second run while another program used the
GPU at 30 % was discarded.) The output is the same token for token (`TestNativeBackendSampling`, and the same
fingerprint 8271256b in every run).

Iris Xe (Vulkan), uma-4g, one run with `sampling=device` (`raw/igpu-uma4g-sampling-device.json`, not an A/B): pp 52.6,
time to first token 1015 ms, tg 11.3, MTP 14.9, echo 34.9 tok/s, against the n7 standard suite with host sampling
(pp 464, tg 11.6, MTP 22.3). Same output (97e62ed1). Device sampling stays off on Vulkan; why it is this slow there
(one-time pipeline compilation in a single repetition, the top-k over 262144 logits on the Iris Xe) is not studied.
