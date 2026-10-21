"""One line per dev-suite result file: python sum.py raw/dev-shapes-*.json

Columns: file tag, variant, pp, tg (d0), chat, MTP chat (acceptance), echo, TTFT, output sha prefix.
"""
import glob
import json
import os
import sys


def main(patterns):
    files = []
    for p in patterns:
        files += sorted(glob.glob(p), key=lambda f: (len(f), f))
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # a half-written file of a run in progress
            print(f"{os.path.basename(f)}: {e}")
            continue
        for r in d.get("runs", []):
            m = r.get("metrics") or {}
            mtp = m.get("mtp") or {}
            echo = m.get("echo") or {}
            tg = (m.get("tg_tok_s") or {}).get("d0", 0)
            out = (r.get("output") or {}).get("sha256", "")[:8]
            print(
                "{:<46} {:<58} pp {:7.0f}  tg {:6.1f}  chat {:6.1f}  mtp {:6.1f} ({:.2f})  echo {:6.1f}  ttft {:5.1f}  {}".format(
                    os.path.basename(f)[:46],
                    (r.get("variant") or "")[:58],
                    m.get("pp_tok_s", 0),
                    tg,
                    m.get("chat_tok_s", 0),
                    mtp.get("tok_s", 0),
                    mtp.get("acceptance", 0),
                    echo.get("tok_s", 0),
                    m.get("ttft_ms", 0),
                    out,
                )
            )


if __name__ == "__main__":
    main(sys.argv[1:] or ["raw/*.json"])
