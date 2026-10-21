"""Per-request table of a gemma.json run (gemma/<name>/answers.json) and its memory peaks (mem.json), as Markdown.
Numbers only, never the answers (they hold a private transcript).

    python -I gemma_table.py final2-cpu-default final2-igpu-mtp2
"""
import json
import os
import sys

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma")


def gb(x):
    return f"{x / 1e9:.2f}".replace(".", ",")


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


for name in sys.argv[1:]:
    steps = json.load(open(os.path.join(here, name, "answers.json"), encoding="utf-8"))
    mem = json.load(open(os.path.join(here, name, "mem.json"), encoding="utf-8-sig"))
    print(f"### {name}\n")
    print("| № | Вход, токенов (из кэша) | Выход, токенов | Время, с | Prefill, ток/с | Decode, ток/с | Принятие MTP |")
    print("|---|---|---|---|---|---|---|")
    tin = tout = twall = tpre = tdec = 0.0
    for i, it in enumerate(steps, 1):
        s = it["response"]["stats"]
        st = s["stages_ms"]
        tin += s["prompt_tokens"] - s["cached_tokens"]
        tout += s["completion_tokens"]
        twall += it["wall_s"]
        tpre += st["prefill"] / 1000
        tdec += st["decode"] / 1000
        print(f"| {i} | {s['prompt_tokens']} ({s['cached_tokens']}) | {s['completion_tokens']} | {num(it['wall_s'])} | "
              f"{num(s['prefill_tok_s'], 0)} | {num(s['decode_tok_s'])} | {num(s['draft_acceptance'], 2)} |")
    print(f"| Σ | {int(tin)} новых | {int(tout)} | {num(twall)} | {num(tin / tpre, 0)} | {num(tout / tdec)} | |")
    print(f"\nПиковый рабочий набор процесса {gb(mem['peak_working_set_os'])} ГБ; пик private bytes "
          f"{gb(mem['peak_private_sampled'])} ГБ; пик GPU shared {gb(mem['peak_proc_shared'])} ГБ.\n")
