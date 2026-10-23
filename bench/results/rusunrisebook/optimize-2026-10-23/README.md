# `nanollm optimize`: the check on the laptop (2026-10-23)

docs/OPTIMIZE.md, section 7, on rusunrisebook (i7-12700H 6P+8E, 32 GiB, RTX 3070 Ti Laptop 8 GiB with the desktop on it,
Iris Xe, two NVMe drives). A build of the working tree (`nanollm-opt.exe`) over the release v0.1.0-rc.1 libraries with the
CUDA 13 add-on (`D:\n8rc`), configuration `tune.mode: optimal`, Gemma 4 E4B QAT with its MTP drafter. Every command ran as
an ordinary user (medium integrity, not in the Administrators role); nothing needed more. The machine was not otherwise
quiet (`busy_machine` fires at 20-23 % CPU from the editor and the chat client), which the tool says itself.

## The three modes, one pass each (`optimal.json`, `powerful.json`, `silent.json`; `nanollm optimize --mode M --json`)

| Mode | Time | Chosen | Context x slots | MTP | Held (plan -> measured) | Typical answer (4096-token prompt, 300 tokens) |
|---|---|---|---|---|---|---|
| optimal | 11 s | RTX, CUDA, M3 vram-ple-disk | 16384 x 1 | 3 / 0 | 3.89 -> **3.85 GiB** (limit 4 GiB + 200 MiB; 0.67 private + 3.18 VRAM, PDH) | 2.8 s: 6087 tok/s, 140.9 tok/s |
| powerful | 15 s | RTX, CUDA, M1 vram-full, vram-6.5g | 16384 x 2 | 3 / 0 | 6.05 -> 5.97 GiB (0.69 + 5.28, no `vram_spill`) | 2.9 s: 5894 tok/s, 137.9 tok/s |
| silent | 39 s | CPU, ram-1g, M9 mmap, 2 threads, below-normal priority + EcoQoS | 4608 x 1 | off | 0.57 -> 0.63 GiB | 391 s: 21 tok/s (carried over from 512 tokens), 1.5 tok/s |

- **Time:** every mode is under the 2 minutes, including the survey (1.4 s) and the wait for the GPU to cool.
- **Optimal:** CUDA M3, 0.35 GiB under the limit; no CPU team is set because the model runs on the GPU (the mode's 7 threads
  apply to a CPU candidate).
- **Powerful:** no spill; `mtp3/0` and `mtp2/0` are within 5 % of each other at two slots (four passes chose 2/0, 2/0, 3/0 and 3/0):
  the static default stays unless the other is 5 % faster (`verified.gain` 1.0-1.08).
- **Silent:** the first rung of the ladder (ram-1g) already answers above 1 tok/s after the prompt, so the walk stops there
  (`skipped`: the six larger budgets). It is the least memory, not a pleasant speed; an answer takes 6.5 minutes. On a
  130 MB/s disk the same ladder stops at ram-3g or ram-4g (golden `silent-hdd`).
- `--dry-run` takes 1.8 s (optimal), 3.3 s (silent), 3.9 s (powerful): `dryrun-*.txt`.

## What was reproduced by hand

| Item | How | Result |
|---|---|---|
| Machine and plan limits | `--dry-run` per mode | candidates with plan, estimate, verdict; Iris Xe on `uma` M3 at 16k context under the cap; the RTX once (the same card through Vulkan is dropped) |
| `vram_busy` | the desktop holds 2.0-2.1 GiB of the card | reported on every run |
| `busy_machine`, `hot_gpu` | the machine's own load and the card after a run | reported |
| `cuda_addon_missing` | the library directory without the CUDA add-on (`C:\Projects\runtimes\libs\n8-nocuda`) | `dryrun-optimal-nocuda.txt`: the card runs through Vulkan, advice `-Cuda` |
| Stale profile | the stored profile's key edited to libraries `n7-1` and driver `575.10` | `--status`: "stale (libraries n7-1 -> dev; driver 575.10 -> 581.57)", JSON `stale: true` with both changes (`stale-status.txt`) |
| Through the service | `nanollm serve`, then `optimize --mode optimal --yes` | measured inside it in 11 s; a chat request during the measurement got **503 with `Retry-After: 10`** (`hold-503.txt`); the profile was applied and the model reloaded (`optimize-via-service.txt`, `serve-info.log`); `GET /api/v1/optimize` and `nanollm plan` show the profile as the source of the draft length |
| No administrator rights | everything above | as an ordinary user |
| Installers | the optimize step of `install.ps1` and `install.sh` cut out of the scripts and run against a stub `nanollm`, ten scenarios each (new install, upgrade with current and stale profile, `-Mode`, `-NoOptimize`, failed measurement, bad mode, no `--status` answer) | behave as docs/OPTIMIZE.md 4.3 says; `json_field` also read the real `--status --json` in its three states |

## Not checked here, and why

`scripts/check-optimize.ps1` (elevated PowerShell: the measurement as LocalSystem through a one-shot scheduled task, and the real
`install.ps1` into a scratch prefix) and `scripts/check-optimize.sh` (hyper: the hard disk, the NUMA nodes, the quiet mode that
must not stream, `install.sh` as root) run the items below on the machines that have them. Their plumbing was tried here as an
ordinary user (`-AsCurrentUser`; the shell script with this laptop's binary: all checks passed), not the parts that need the
rights or the hardware.

- **hyper (Linux, HDD, 4 NUMA nodes):** not reachable from here (the SSH keys are refused). The Linux code paths ran on WSL2 instead
  (`../optimize-wsl2-2026-10-23/`); what stays open is a real hard disk and several NUMA nodes. The HDD, NUMA and haswell paths are covered by the synthetic decisions
  (`testdata/tune/decide/silent-hdd.txt`, the blind run's hyper case, `TestNUMAPoliciesAreCandidatesOnAMultiNodeMachine`) and by
  the numbers of the 2026-10-19/20 runs in `bench/results/hyper/`, not by a run.
- **The installed service (LocalSystem):** PDH counters from LocalSystem and the real `install.ps1` need administrator rights.
  The service path was exercised with `nanollm serve` as the user. The real `install.sh` as a systemd service ran as root on WSL2 and
  passed (`../optimize-wsl2-2026-10-23/installer/`).
- **Battery, power saver, a slow disk, a VM or WSL, session 0:** this machine is on AC with both drives NVMe; these problems are
  reproduced by the unit tests on synthetic surveys (`TestProblemsOfASurvey`, `TestEnvironmentProblems`,
  `TestNoGPULibraryProblems`), not by hand.

## What the check changed in the code

- The comparison of drafted and plain decoding tolerates a near tie (`NearTieNats`): CUDA's batch and one-token kernels may
  part where the model hesitates between two words, and strict equality would have switched MTP (x1.9) off as `draft_mismatch`.
- `Profile.Hash` is the decisions only, not the numbers they were measured with: a repeat measurement that chooses the same
  made another hash, so the pool loaded the model again.
- The planner adds 448 MiB of driver host memory to the cap check for Vulkan as well as CUDA (the Iris plan measured 445 MiB
  over): the Iris candidate of the optimal mode plans 16k context instead of 32k.
- A speed baseline does not pass from an iGPU on shared memory to a card with memory of its own (the NVIDIA card through
  Vulkan was estimated at the Iris's 27 s instead of being measured).
- `vram_spill` compares PDH Shared with the plan's host memory without the driver's 448 MiB (vram-8g's +424 MiB would have been
  missed); the reference text is frozen by a hash; the native CUDA test no longer leaves `MMVQ_NWARPS` at 0 for the tests after it.
- The baselines of matrix.yaml are the calibrations of `optimize-stage0-2026-10-23/`.
