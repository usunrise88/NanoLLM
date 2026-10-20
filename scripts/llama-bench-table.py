#!/usr/bin/env python3
"""Medians of llama-bench markdown blocks over repeated rounds.

    python3 -I scripts/llama-bench-table.py bench/results/hyper/2026-10-20-numa-distribute.txt

The files of scripts/night-hdd.sh, tools-ab.sh and tn-curve.sh are blocks of
"## round N <what>" followed by llama-bench's markdown table. Blocks with the
same title once "round N" is dropped are one configuration; each test (pp512,
tg128, ...) gets the median of the rounds' means and their range. Decimal comma
as docs/AUTOTUNE.md; --dot for a point.
"""
import re
import statistics
import sys


def main(argv):
    dot = "--dot" in argv
    files = [a for a in argv if a != "--dot"]
    if not files:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    fmt = (lambda v, n=1: f"{v:.{n}f}") if dot else (lambda v, n=1: f"{v:.{n}f}".replace(".", ","))
    for path in files:
        groups, order, title = {}, [], None
        for line in open(path):
            line = line.rstrip("\n")
            if line.startswith("## "):
                title = re.sub(r"^round \d+ ?", "", line[3:]).strip() or line[3:]
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if title and len(cells) >= 2 and re.fullmatch(r"(pp|tg)\d+( @ d\d+)?", cells[-2] or ""):
                m = re.match(r"([\d.]+)", cells[-1])
                if m:
                    key = (title, cells[-2])
                    if key not in groups:
                        groups[key] = []
                        order.append(key)
                    groups[key].append(float(m.group(1)))
        print(f"### {path}\n")
        print("| конфигурация | тест | медиана, ток/с | разброс по раундам | раундов |")
        print("| --- | --- | --- | --- | --- |")
        for key in order:
            v = groups[key]
            print(f"| {key[0]} | {key[1]} | {fmt(statistics.median(v))} | {fmt(min(v))}…{fmt(max(v))} | {len(v)} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
