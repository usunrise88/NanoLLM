#!/usr/bin/env python3
"""Markdown tables of `nanollm bench --suite dev` results, one row per run.

    python3 -I scripts/bench-table.py bench/results/hyper/2026-10-19-dev-*.json

Numbers use a decimal comma, as docs/AUTOTUNE.md does (--dot for a point).
A run that failed shows its reason. Columns that no run of a file has are
left out: draft (MTP or prompt lookup) and echo appear only when measured.
"""
import json
import sys


def fmt(v, nd=1, dot=False):
    if v is None:
        return "—"
    s = f"{v:.{nd}f}"
    return s if dot else s.replace(".", ",")


def pm(v, sd, nd=1, dot=False):
    if v is None:
        return "—"
    if sd:
        return f"{fmt(v, nd, dot)} ± {fmt(sd, nd, dot)}"
    return fmt(v, nd, dot)


def main(argv):
    dot = "--dot" in argv
    files = [a for a in argv if a != "--dot"]
    if not files:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    for path in files:
        with open(path) as f:
            d = json.load(f)
        runs = d.get("runs", [])
        extra = {e["key"]: e["value"] for e in d.get("machine", {}).get("extra", []) or [] if isinstance(e, dict)}
        print(f"### {path}\n")
        if extra:
            print("machine: " + ", ".join(f"{k}={v}" for k, v in extra.items()) + "\n")
        has_draft = any(r.get("metrics", {}).get("mtp") for r in runs)
        has_echo = any(r.get("metrics", {}).get("echo") for r in runs)
        head = ["preset", "variant", "mode", "pp tok/s", "tg tok/s", "tg@depth", "chat tok/s", "TTFT ms", "load s", "cgroup GiB"]
        if has_draft:
            head += ["draft tok/s", "accept", "×"]
        if has_echo:
            head += ["echo tok/s", "echo accept", "echo ×"]
        print("| " + " | ".join(head) + " |")
        print("|" + " --- |" * len(head))
        for r in runs:
            m = r.get("metrics", {}) or {}
            sd = m.get("stddev", {}) or {}
            tg = m.get("tg_tok_s", {}) or {}
            plan = r.get("plan", {}) or {}
            row = [r.get("preset", ""), r.get("variant", "") or "по плану", plan.get("mode", "")]
            if r.get("status") != "ok":
                row += [f"{r.get('status')}: {r.get('reason', '')}"] + [""] * (len(head) - 4)
                print("| " + " | ".join(row) + " |")
                continue
            cg = m.get("peak_cgroup")
            row += [
                pm(m.get("pp_tok_s"), sd.get("pp_tok_s"), 1, dot),
                pm(tg.get("d0"), sd.get("tg_tok_s.d0"), 2, dot),
                fmt(tg.get("dN"), 2, dot),
                pm(m.get("chat_tok_s"), sd.get("chat_tok_s"), 2, dot),
                fmt(m.get("ttft_ms"), 0, dot),
                fmt(m["load_ms"] / 1000 if m.get("load_ms") else None, 1, dot),
                fmt(cg / 2**30 if cg else None, 2, dot),
            ]
            if has_draft:
                x = m.get("mtp") or {}
                row += [pm(x.get("tok_s"), sd.get("mtp_tok_s"), 2, dot), fmt(x.get("acceptance"), 2, dot), fmt(x.get("speedup"), 2, dot)]
            if has_echo:
                x = m.get("echo") or {}
                row += [fmt(x.get("tok_s"), 2, dot), fmt(x.get("acceptance"), 2, dot), fmt(x.get("speedup"), 2, dot)]
            print("| " + " | ".join(row) + " |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
