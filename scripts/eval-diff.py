#!/usr/bin/env python3
"""Item-by-item comparison of two `nanollm eval` reports of the same suite.

    python3 -I scripts/eval-diff.py REF.json CAND.json

For a change that must not alter the output (prompt lookup, layer streaming,
threads, NUMA) the answers of a greedy run have to match word for word; the
pass rate would hide a different answer that passes as well. Prints the
pass rates, the number of identical answers, and every item whose answer,
finish reason or completion length differs. Exit status 1 when any differs.
"""
import json
import sys


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d, {(r["id"], bool(r.get("thinking"))): r for r in d["results"]}


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    (da, a), (db, b) = load(argv[0]), load(argv[1])
    print(f"suite {da['suite']} vs {db['suite']}: {len(a)} and {len(b)} items")
    only = sorted(set(a) ^ set(b))
    for k in only:
        print(f"  only in {'ref' if k in a else 'cand'}: {k}")
    same = diff = 0
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        ct = lambda r: (r.get("usage") or {}).get("completion_tokens")
        if (x["answer"], x.get("finish"), ct(x)) == (y["answer"], y.get("finish"), ct(y)):
            same += 1
            continue
        diff += 1
        print(f"  DIFF {k[0]} thinking={k[1]}: pass {x['pass']}→{y['pass']}, finish {x.get('finish')}→{y.get('finish')}, tokens {ct(x)}→{ct(y)}")
        print(f"    ref : {x['answer'][:160]!r}")
        print(f"    cand: {y['answer'][:160]!r}")
    for name, d in (("ref ", da), ("cand", db)):
        s = ", ".join(f"{k} {v['pass']}/{v['n']}" for k, v in d["summary"].items())
        print(f"  {name} pass: {s}")
    ms = lambda d: sum(r["ms"] for r in d["results"]) / 1000
    print(f"  identical {same}, different {diff}, only-in-one {len(only)}; time ref {ms(da):.0f} s, cand {ms(db):.0f} s (load differs: not a speed measurement)")
    return 1 if diff or only else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
