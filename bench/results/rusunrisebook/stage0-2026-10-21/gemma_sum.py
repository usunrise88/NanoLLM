"""Totals of gemma.json runs (gemma/<name>/answers.json): wall, prefill and decode seconds, per-request speeds.

    python -I gemma_sum.py fs1-fa-mtp2 fs1-small-mtp2 ...   (no arguments: every directory)
Prints numbers only, never the answers (they hold a private transcript).
"""
import json
import os
import sys

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma")
names = sys.argv[1:] or sorted(d for d in os.listdir(here) if os.path.isfile(os.path.join(here, d, "answers.json")))
for n in names:
    p = os.path.join(here, n, "answers.json")
    if not os.path.isfile(p):
        print(f"{n:28} (no answers.json)")
        continue
    steps = json.load(open(p, encoding="utf-8"))
    wall = pre = dec = 0.0
    per = []
    for it in steps:
        s = it["response"]["stats"]
        st = s["stages_ms"]
        wall += it["wall_s"]
        pre += st["prefill"] / 1000
        dec += st["decode"] / 1000
        per.append(f"{s['prefill_tok_s']:.0f}/{s['decode_tok_s']:.1f}")
    print(f"{n:28} wall {wall:6.1f}  prefill {pre:5.1f}  decode {dec:5.1f}  pp/tg per request: {' '.join(per)}")
