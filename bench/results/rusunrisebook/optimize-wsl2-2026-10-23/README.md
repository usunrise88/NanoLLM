# `nanollm optimize` on Linux: WSL2 Ubuntu 24.04 (2026-10-23)

The Linux code paths of the autotuner (procfs memory, nice per thread, `/sys` drive kind, the Linux survey, `install.sh`'s status
reading) on the only Linux this laptop has: WSL2 Ubuntu 24.04 on the same i7-12700H (20 threads, 15.5 GiB for the VM, a virtual
disk, no GPU libraries in the Linux archive). An ordinary user, no root; the branch's Linux binary over the release v0.1.0-rc.1
Linux libraries (llama.cpp c96d19c52), the model copied in and re-verified offline. `scripts/check-optimize.sh` did the runs.

| | |
|---|---|
| Dry runs | 1.5 s (silent), 1.4 s (optimal), 1.8 s (powerful) |
| silent | 28 s: CPU, ram-1g, M9, 2 threads, 0.59 GiB held (plan 0.57), 1.2 tok/s after a prompt |
| optimal | 26 s: CPU, ram-11.5g M7 ram-ple-disk, 5 threads (half of the 10 cores WSL shows), MTP 2 / 0.5, **3.37 GiB held** (plan 3.43), draft check passed |
| powerful | 30 s: CPU, M6 ram-full, 10 threads, 6.45 GiB held (plan 6.47), drafting 9.7 % faster than the rule's, draft check passed |
| Priority | the quiet mode's worker has every thread at nice 10 (`posture.txt`) |
| Memory | the plan and the `RssAnon + RssShmem` sampling agree within 2 % on all three |

What the run found, now fixed in the branch:
- **A virtual disk is not a hard disk.** WSL2 (like most VMs and cloud volumes) reports every disk as rotational, so the first
  run said "the model is on a slow disk (2712 MB/s)". The measurement now decides whether a disk is slow, and a drive the OS calls
  rotational that reads 1000 MiB/s or more is taken for a solid-state or virtual one (`DiskInfo.trustTheSpeed`).
- The CUDA-add-on advice said `install.ps1 -Cuda` on Linux; it names `install.sh --cuda` there.
- `scripts/check-optimize.sh` asserts "no streaming in the quiet mode" only for a disk the autotuner calls a hard disk, which
  after the fix means a real one; here the quiet mode streams (ram-1g) because the virtual disk reads 2.7 GB/s, as the rule says.

Not covered by this machine: a real hard disk, several NUMA nodes (WSL has one), a systemd service (`install.sh` would install a
unit into the distribution; it was not run).

`machine.txt`, `dryrun-*.txt`, `silent.json`, `optimal.json`, `powerful.json` are the script's outputs; `posture-silent.json` is the
silent run that `posture.txt` was sampled from.
