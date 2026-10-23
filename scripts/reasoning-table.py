#!/usr/bin/env python3
# Sum up the runs of scripts/reasoning-eval.sh: accuracy per language and kind, and what the
# thinking cost in tokens and seconds.
#
#   python3 -I scripts/reasoning-table.py ~/nanollm-data/eval/reasoning-direct-*.json
import collections
import json
import statistics
import sys

print(f"{'run':14s} {'ru':>8s} {'en':>8s} {'all':>7s} {'tokens mean/med/max':>21s} {'reasoning mean':>15s} {'s/item':>7s} {'cut off':>8s}")
rows = []
for path in sys.argv[1:]:
    d = json.load(open(path))
    name = path.rsplit("reasoning-direct-", 1)[-1].removesuffix(".json")
    res = d["results"]
    ok = collections.Counter()
    n = collections.Counter()
    kinds = collections.defaultdict(lambda: [0, 0])
    for r in res:
        n[r["lang"]] += 1
        ok[r["lang"]] += bool(r["pass"])
        kinds[r["id"].split("-")[2]][1] += 1
        kinds[r["id"].split("-")[2]][0] += bool(r["pass"])
    toks = [r["usage"]["completion_tokens"] for r in res]
    reas = [r["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0) for r in res]
    secs = [r["ms"] / 1000 for r in res]
    cut = sum(r["finish"] == "length" for r in res)
    tot = sum(ok.values())
    print(f"{name:14s} {ok['ru']:>4d}/{n['ru']:<3d} {ok['en']:>4d}/{n['en']:<3d} {tot:>3d}/{len(res):<3d} "
          f"{statistics.mean(toks):7.0f}/{statistics.median(toks):.0f}/{max(toks):<6d} {statistics.mean(reas):15.0f} {statistics.mean(secs):7.1f} {cut:8d}")
    rows.append((name, {k: f"{v[0]}/{v[1]}" for k, v in sorted(kinds.items())}))
print()
for name, k in rows:
    print(f"{name:14s}", "  ".join(f"{a} {b}" for a, b in k.items()))
