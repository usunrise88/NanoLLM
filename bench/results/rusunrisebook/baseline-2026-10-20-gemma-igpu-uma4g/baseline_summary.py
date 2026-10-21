# Collects the baseline runs (variant x repetition) into one summary: per-step
# speed, totals, GPU/RAM peaks, similarity to the reference answers in
# gemma.json and whether repeated runs give the same text.
#
#   python -I baseline_summary.py gemma.json OUT_DIR RUN_DIR...
# RUN_DIR is named <variant>-base<rep>, e.g. mtp3-base1; its answers.json and
# mem.json are the run's raw results.
import difflib
import json
import os
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
ref_path, out_dir, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
ref = json.load(open(ref_path, encoding="utf-8"))["steps"]
G = 1024 ** 3


def answer(resp):
    return resp["choices"][0]["message"].get("content") or ""


summary = {"runs": {}}
texts = {}
for run in runs:
    name = os.path.basename(run.rstrip("/\\"))
    variant, rep = name.rsplit("-base", 1)
    ours = json.load(open(os.path.join(run, "answers.json"), encoding="utf-8"))
    steps = []
    for r, o in zip(ref, ours):
        st = o["response"].get("stats") or {}
        a, b = answer(r["response"]), answer(o["response"])
        steps.append({
            "part": r["part"],
            "prompt_tokens": st.get("prompt_tokens"),
            "cached_tokens": st.get("cached_tokens"),
            "completion_tokens": st.get("completion_tokens"),
            "ref_completion_tokens": r["response"]["usage"]["completion_tokens"],
            "ttft_s": round(st.get("ttft_ms", 0) / 1000, 2),
            "prefill_tok_s": round(st.get("prefill_tok_s", 0), 1),
            "decode_tok_s": round(st.get("decode_tok_s", 0), 2),
            "draft_acceptance": round(st.get("draft_acceptance") or 0, 3),
            "wall_s": round(o["wall_s"], 2),
            "finish": o["response"]["choices"][0].get("finish_reason"),
            "exact_vs_ref": a == b,
            "sim_vs_ref": round(difflib.SequenceMatcher(None, a.split(), b.split()).ratio(), 3),
        })
        texts.setdefault((variant, r["part"]), {})[rep] = b
    mem = json.load(open(os.path.join(run, "mem.json"), encoding="utf-8-sig"))
    total = {
        "wall_s": round(sum(s["wall_s"] for s in steps), 1),
        "prompt_tokens": sum(s["prompt_tokens"] or 0 for s in steps),
        "completion_tokens": sum(s["completion_tokens"] or 0 for s in steps),
        "decode_tok_s_end_to_end": round(
            sum(s["completion_tokens"] or 0 for s in steps) / sum(s["wall_s"] for s in steps), 2),
    }
    summary["runs"][name] = {
        "variant": variant, "rep": int(rep), "steps": steps, "total": total,
        "memory_gib": {
            "gpu_process_shared_peak": round(mem["peak_proc_shared"] / G, 2),
            "gpu_process_dedicated_peak": round(mem["peak_proc_dedicated"] / G, 2),
            "gpu_adapter_shared_baseline": round(mem["adapter_shared_baseline"] / G, 2),
            "gpu_adapter_shared_peak": round(mem["peak_adapter_shared"] / G, 2),
            "working_set_peak": round(mem["peak_working_set_os"] / G, 2),
            "private_peak": round(mem["peak_private_sampled"] / G, 2),
        },
    }

# Repetitions of one variant: the same text?
summary["repeat_identical_text"] = {}
for (variant, part), by_rep in sorted(texts.items()):
    vals = list(by_rep.values())
    summary["repeat_identical_text"].setdefault(variant, {})[str(part)] = all(v == vals[0] for v in vals)

json.dump(summary, open(os.path.join(out_dir, "baseline-summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# Table.
for run, d in sorted(summary["runs"].items()):
    t, m = d["total"], d["memory_gib"]
    print(f"### {run}: wall {t['wall_s']} s, {t['completion_tokens']} out tok, end-to-end {t['decode_tok_s_end_to_end']} tok/s | "
          f"GPU proc {m['gpu_process_shared_peak']} GiB (dedicated {m['gpu_process_dedicated_peak']}), WS peak {m['working_set_peak']} GiB")
    for s in d["steps"]:
        print(f"  part {s['part']}: prompt {s['prompt_tokens']} (cached {s['cached_tokens']}) out {s['completion_tokens']} "
              f"(ref {s['ref_completion_tokens']}) | ttft {s['ttft_s']} s prefill {s['prefill_tok_s']} decode {s['decode_tok_s']} "
              f"accept {s['draft_acceptance']} wall {s['wall_s']} s | sim {s['sim_vs_ref']} exact {s['exact_vs_ref']}")
print()
for variant in sorted({d["variant"] for d in summary["runs"].values()}):
    rs = [d for d in summary["runs"].values() if d["variant"] == variant]
    walls = [d["total"]["wall_s"] for d in rs]
    print(f"{variant}: wall {', '.join(map(str, walls))} s -> mean {statistics.mean(walls):.1f} s"
          + (f", spread {max(walls) - min(walls):.1f} s ({(max(walls) - min(walls)) / statistics.mean(walls) * 100:.1f}%)" if len(walls) > 1 else "")
          + f"; same text between reps: {summary['repeat_identical_text'][variant]}")
