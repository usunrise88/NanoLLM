#!/usr/bin/env python3
# Stage 2 plan task 11 (OPT-7): what a re-ordered model file could save on a disk.
#
#   scripts/hyper-exclusive.sh -- python3 -I scripts/layout-probe.py PLAN.json MODEL.gguf [REPS]
#
# PLAN.json is `nanollm plan gemma-4-e4b-qat --preset ram-2g --json`. The file is
# already in layer order; a layer's cold tensors (ffn_down, ffn_gate, ffn_up) are
# three ranges with ~10 KiB of ffn_norm between them. A relayout would make them
# one range per layer and pack the pinned tensors elsewhere. This reads the plan's
# stream ranges cold, in file order, (a) as the plan has them and (b) with every gap
# smaller than 1 MiB read through, which is what one contiguous range per layer
# costs. If the two are within noise, a relayout buys nothing on this disk.
# Only the model file's own page cache is dropped (posix_fadvise), never the host's.
import json
import os
import sys
import time

plan = json.load(open(sys.argv[1]))
path = sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ranges = sorted((r["offset"], r["length"]) for r in plan["stream_ranges"])
MiB = 1 << 20


def merged(rs, gap):
    out = [list(rs[0])]
    for off, ln in rs[1:]:
        last = out[-1]
        if off - (last[0] + last[1]) < gap:
            last[1] = off + ln - last[0]
        else:
            out.append([off, ln])
    return out


def read(rs, fd):
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    total = 0
    t0 = time.perf_counter()
    for off, ln in rs:
        done = 0
        while done < ln:
            chunk = os.pread(fd, min(4 * MiB, ln - done), off + done)
            if not chunk:
                break
            done += len(chunk)
        total += done
    dt = time.perf_counter() - t0
    return total, dt


fd = os.open(path, os.O_RDONLY)
variants = {"as planned": ranges, "gaps under 1 MiB read through": merged(ranges, MiB)}
print(f"{len(ranges)} stream ranges, {sum(l for _, l in ranges) / MiB:.0f} MiB; merged: {len(variants['gaps under 1 MiB read through'])} ranges")
res = {k: [] for k in variants}
for rep in range(reps):
    for name in (list(variants) if rep % 2 == 0 else list(variants)[::-1]):
        total, dt = read(variants[name], fd)
        res[name].append(total / dt / MiB)
        print(f"rep {rep} {name:32s} {total / MiB:8.0f} MiB in {dt:6.1f} s = {total / dt / MiB:6.1f} MiB/s")
for name, v in res.items():
    v.sort()
    print(f"median {name:32s} {v[len(v) // 2]:6.1f} MiB/s")
