"""Tabulate the nanollm bench result files of raw/ (one line per run)."""
import glob
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
rows = []
for p in sorted(glob.glob(os.path.join(here, "raw", "*.json"))):
    d = json.load(open(p, encoding="utf-8"))
    name = os.path.basename(p)[:-5]
    for r in d["runs"]:
        m = r.get("metrics", {})
        tg = m.get("tg_tok_s", {})
        mtp = m.get("mtp", {}) or {}
        rows.append((
            name,
            r.get("variant") or "",
            r.get("status", "?"),
            m.get("pp_tok_s"), tg.get("d0"), tg.get("dN"), m.get("chat_tok_s"),
            mtp.get("tok_s"), mtp.get("acceptance"), mtp.get("speedup"),
            (m.get("peak_rss") or 0) / 2**30, m.get("load_ms"),
            r["plan"].get("mode"), r["plan"].get("swa_full"),
        ))


def f(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) else "-"


print(f"{'file':22} {'variant':44} {'st':3} {'pp':>6} {'tg0':>5} {'tgN':>5} {'chat':>5} {'mtp':>5} {'acc':>4} {'spd':>4} {'rssGiB':>6} {'load_s':>6} mode")
for n, v, st, pp, t0, tn, ch, mt, ac, sp, rss, ld, mode, swa in rows:
    print(f"{n:22} {v[:44]:44} {st[:3]:3} {f(pp, '6.1f')} {f(t0, '5.1f')} {f(tn, '5.1f')} {f(ch, '5.1f')} {f(mt, '5.1f')} {f(ac, '4.2f')} {f(sp, '4.2f')} {f(rss, '6.2f')} {f((ld or 0) / 1000, '6.1f')} {mode} swa={swa}")
