# `nanollm managed`: the check on the laptop (2026-10-24)

nanollm-embedded.md and the plan of the managed mode, on rusunrisebook (i7-12700H 6P+8E, 32 GiB, RTX 3070 Ti Laptop 8 GiB
with the desktop on it, driver 581.57). A build of the `feat/managed` working tree over the release v0.1.0-rc.1 libraries with
the CUDA 13 add-on (`D:\n8rc\app\lib`, as `lib\` next to the binary), Gemma 4 E4B QAT with its MTP drafter. The model was not
downloaded: its files were hard-linked into a fresh data directory. `check.py` started the binary the way an application does
(pipes on the standard streams, `CREATE_NO_WINDOW`, `-parent-pid` its own), as an ordinary user, and recorded what it saw in
`check.json`. Model answers were neither printed nor kept: only status codes and token counts. The machine was idle (CPU 2-12 %,
GPU 0 %, 2.2 GiB of VRAM held by the desktop and its apps); one pass.

## First start: no settings, no profile

| Time | Event | Notes |
|---|---|---|
| 2.7 s | `ready` | `configured: false`, `missing_models: []`, `tune: {mode: silent, state: missing}`; chat `loading` |
| 3.5 s | `tune_started` `reason: missing`, `mode: silent` | a chat request now: **503, `Retry-After: 10`**, in 0.02 s |
| 41.9 s | `tune_done` silent | `llamacpp:cpu`, M9 mmap, 4608 context, MTP off, **0.67 GB**, reference answer 396 s; applied |
| | chat | 200 in 1.5 s (2 threads, below normal priority) |
| 42-44 s | `PATCH /api/v1/managed/settings {"mode":"optimal"}` | 200 at once, autotune `missing` |
| 44.1 s | `tune_started` `reason: settings`, `mode: optimal` | |
| 59.0 s | `tune_done` optimal | `llamacpp:cuda0`, **M3 vram-ple-disk**, 16384 context, MTP 2/0, **4.14 GB** (limit 4 GiB + 200 MiB), reference answer 2.6 s; applied |
| | chat | 200 in 0.23 s |
| 59.5 s | stdin closed → `stopping` `stdin closed` | exit 0 |

- Silent took 38 s and optimal 15 s, the survey and the wait for the GPU to cool included; the numbers agree with the CLI's
  measurement of the day before (optimize-2026-10-23: 0.63 GiB / 391 s and 3.85 GiB / 2.8 s; MTP chose 2/0 here, 3/0 there,
  a near tie).
- The priority was `BelowNormal` in both modes (`-low-priority` is on by default); the quiet mode adds EcoQoS, which was not
  read back here (internal/posture tests do).
- **No windows.** Every descendant seen while it measured: 17 sightings of the `nanollm` workers, 39 of `conhost`, one
  `powershell` (the survey): none had a window (`MainWindowHandle` 0).

## Second start: the settings and the profile are kept

`ready` 1.6 s after the start: `configured: true`, `tune: {mode: optimal, state: ready}`. No event for 15 s (nothing measured);
chat `ready` with the optimal profile, a chat request 200 in 0.36 s, priority `BelowNormal`, stdin closed → `stopping`, exit 0.

## Found on the way

- **The stamps of a model are bound to its path on Windows.** `.verified` is keyed by `path.Base(p)` (internal/registry
  store.go), which on Windows, with backslashes, is the whole path: a model directory copied or moved elsewhere counts as
  `corrupt` until `models pull` rehashes it, so the first try of this check started with `missing_models` and
  `tune: waiting_model`. The scratch copy's keys were rewritten for the check. `filepath.Base` would fix it, at the cost of one
  rehash of every Windows install; left for the owner to decide.

Linux and macOS are covered by CI (the install job runs the archive's binary as an application on the scripted backend); a
real model there was not run.
