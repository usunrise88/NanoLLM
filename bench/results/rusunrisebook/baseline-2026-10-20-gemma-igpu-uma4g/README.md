# Baseline: gemma.json on the Iris Xe iGPU, uma-4g, MTP-3 and MTP-2

Taken 2026-10-20. This is the reference for measuring what the next changes do: re-run the same four
variants after a change and compare with `baseline-summary.json` (and the raw `answers.json` of each run).

## What was run

The five requests of `gemma.json` (the Gemma 4 E4B QAT summaries of a meeting transcript, 4.4k, 4.6k, 4.6k, 2.0k
and 0.7k prompt tokens, temperature 0), one after another against one `nanollm serve`, preset `uma-4g`
(mode auto), speculative decoding `mtp`, `p_min` 0, `n_max` 3 or 2, device `llamacpp:vulkan0` (the Iris Xe;
the RTX 3070 Ti is hidden with `GGML_VK_VISIBLE_DEVICES=1`, `devices.fallback: false`). Every variant ran twice,
the runs alternated (MTP-3, MTP-2, MTP-3, MTP-2) with a fresh server for each.

| | |
|---|---|
| Machine | i7-12700H, 32 GiB DDR4-3200 dual channel, Iris Xe (96 EU, shared memory); on AC, power plan Performance |
| GPU driver | Intel 32.0.101.7092 (2026-09-03). Texts differ between drivers, so do not mix them |
| NanoLLM | `b11600-n4-11-ge6ce070` binary (the repo HEAD 722897c only adds docs and results); llama.cpp b11600 `aad9f95e3` with the n4 patches |
| SHA-256 (first 16) | nanollm.exe `8EDB90DADFD9738D`, ggml-vulkan.dll `C9C0F9280B0AEEB3`, llama.dll `38D48E2B57EEA705`, ggml-base.dll `6D0FC59BA045A8DC` |
| Model | gemma-4-e4b-qat (UD-Q4_K_XL, 4,215,695,776 B) + MTP drafter (59,678,016 B) |
| RAM before the run | about 14 GiB free of 32 GiB; Chrome, Telegram and the editor were open, nothing heavy |

## Result (mean of the two repetitions)

The speed numbers come from the server's own `stats` of each response. `Out` is the number of generated
tokens, the reference count from `gemma.json` in brackets. `Sim` is the word-level similarity to the
reference answer (1.00 means the same words).

| Variant | Part | Prompt tok | Out (ref) | TTFT, s | Prefill, tok/s | Decode, tok/s | Draft accept | Wall, s | Sim |
|---|---|---|---|---|---|---|---|---|---|
| MTP-3 | 1 | 4365 | 124 (156) | 11.2 | 391 | 10.9 | 0.37 | 22.5 | 0.75 |
| MTP-3 | 2 | 4625 | 116 (124) | 11.7 | 383 | 9.0 | 0.25 | 24.5 | 0.20 |
| MTP-3 | 3 | 4572 | 111 (128) | 11.5 | 385 | 10.6 | 0.34 | 22.0 | 0.54 |
| MTP-3 | 4 | 2010 | 124 (111) | 4.4 | 423 | 11.2 | 0.30 | 15.4 | 0.59 |
| MTP-3 | 5 | 726 | 489 (345) | 1.5 | 433 | 18.8 | 0.67 | 27.5 | 0.48 |
| MTP-2 | 1 | 4365 | 146 (156) | 11.3 | 386 | 11.1 | 0.42 | 24.4 | 0.71 |
| MTP-2 | 2 | 4625 | 105 (124) | 11.9 | 377 | 9.5 | 0.30 | 22.9 | 0.23 |
| MTP-2 | 3 | 4572 | 130 (128) | 11.8 | 375 | 11.2 | 0.45 | 23.4 | 0.54 |
| MTP-2 | 4 | 2010 | 124 (111) | 4.5 | 414 | 12.2 | 0.41 | 14.6 | 0.59 |
| MTP-2 | 5 | 726 | 344 (345) | 1.5 | 430 | 17.6 | 0.69 | 21.0 | 1.00 |

| Variant | Total wall, s (rep 1 / rep 2) | Mean | Output tokens | Peak GPU memory of the process | Peak working set |
|---|---|---|---|---|---|
| MTP-3 | 112.1 / 111.6 | 111.8 | 964 | 3.20-3.26 GiB (shared, dedicated 0) | 6.01 GiB |
| MTP-2 | 106.7 / 105.8 | 106.2 | 849 | 3.20 GiB (shared, dedicated 0) | 6.01 GiB |

## How to read it

- **Repeatability:** the total wall time of the two repetitions differs by 0.4% (MTP-3) and 0.8% (MTP-2); the
  texts are identical letter by letter between repetitions of one variant. So a difference of a few percent
  is a real change, not noise, if it is measured in the same session.
- **Between sessions the machine drifts:** the first run on this driver (earlier the same day, same binary) gave
  123.1 s (MTP-2) and 132.6 s (MTP-3), prefill 350-364 tok/s, against 106 s, 112 s and 380-440 tok/s now. The
  cause was not found (power state, temperature and background load are the suspects). Always re-run the
  baseline next to the changed build in one sitting.
- **MTP-2 vs MTP-3 by wall time is not a fair comparison:** the texts differ, and in part 5 MTP-3 writes 489
  tokens against 344 for MTP-2 (the reference has 345). Per token, MTP-2 decodes faster in parts 1-4, where
  the acceptance is low (0.25-0.45), MTP-3 only in part 5, where it is high (0.67).
- **Answers vs the reference:** none of the parts 1-4 repeats the reference text (they diverge at the 41st-389th
  character), part 5 is identical to the reference with MTP-2 only. They
  are summaries of the same meeting with a similar gist and length but other wording (Sim only shows how
  much the words differ, it is not a quality score). The reference comes from another run (the file does not record its backend); MTP verification in
  batches changes the numerics, so a letter-by-letter match is not expected.
- **Memory:** the iGPU memory is system RAM. The peak working set of 6.01 GiB already contains the 3.2-3.3 GiB
  the GPU counters show for the process (do not add them up); the private bytes peak at 3.70 GiB.

## Files

- `baseline-summary.json`: every step of every run, the totals and the memory peaks.
- `<variant>-base<N>/`: raw `answers.json` (the full responses), `gpumem.csv` (GPU and process memory every
  250 ms), `mem.json`, `serve.log`, the `nanollm.yaml` of the run (no keys, `auth.mode: open`), `models.json`.
- `run_variants.ps1`, `run_steps.py`, `run_baseline.ps1`, `baseline_summary.py`, `compare.py`: the scripts
  that made it. `run_baseline.ps1` takes the Intel adapter LUID (`-intelLuid` in `run_variants.ps1`), which
  changes after a driver reinstall: `Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory`
  lists them; the Intel one is the adapter with no dedicated memory.
- To compare a new run: `python -I baseline_summary.py gemma.json OUT_DIR mtp3-base1 mtp2-base1 ...` (the
  directory names must end in `-base<N>`), or `python -I compare.py gemma.json RUN_DIR` for text against the
  reference.
