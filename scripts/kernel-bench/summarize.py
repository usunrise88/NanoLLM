#!/usr/bin/env python3
"""Summarize the output of scripts/ab-llama-bench.sh: per variant, profile and test the median of the round means,
the spread (min..max of the round means) and the mean within-run standard deviation, in tok/s, and the ratio to the
first variant.

    scripts/kernel-bench/summarize.py bench/results/hyper/opt16-kernels-full.txt [...]
"""
import re
import statistics
import sys
from collections import OrderedDict, defaultdict


def parse(path):
    runs = []  # (variant, profile, mode, round, test, mean, std)
    cur = None
    for line in open(path, errors="replace"):
        m = re.match(r"### variant=(\S+) profile=(\S+) mode=(\S+) round=(\d+)", line)
        if m:
            cur = (m.group(1), m.group(2), m.group(3), int(m.group(4)))
            continue
        if line.startswith("### end"):
            cur = None
            continue
        if cur and line.startswith("|") and ("pp" in line or "tg" in line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            test, val = cells[-2], cells[-1]
            if not re.fullmatch(r"(pp|tg)\d+.*", test):
                continue
            mm = re.match(r"([\d.]+)\s*±\s*([\d.]+)", val)
            if mm:
                runs.append(cur + (test, float(mm.group(1)), float(mm.group(2))))
    return runs


def main():
    for path in sys.argv[1:]:
        runs = parse(path)
        by = defaultdict(list)
        order = OrderedDict()
        for v, p, mode, r, test, mean, std in runs:
            by[(p, test, v)].append((mean, std))
            order.setdefault(v, None)
        print(f"## {path}")
        print("| profile | test | variant | median tok/s | spread of the round means | within-run sd | vs first |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for p in ("weak", "strong"):
            for test in sorted({t for (pp, t, _) in by if pp == p}):
                base = None
                for v in order:
                    vals = by.get((p, test, v))
                    if not vals:
                        continue
                    means = [m for m, _ in vals]
                    med = statistics.median(means)
                    if base is None:
                        base = med
                    print(f"| {p} | {test} | {v} | {med:.2f} | {min(means):.2f}..{max(means):.2f} (n={len(means)}) | "
                          f"{statistics.mean(s for _, s in vals):.2f} | {med / base:.2f}x |")


if __name__ == "__main__":
    main()
