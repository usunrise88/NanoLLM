# Compares the answers of one or more runs with the reference answers in
# gemma.json: exact match, the shared prefix, word-level similarity, and the
# speed the server reported.
#
#   python -I compare.py gemma.json RUN_DIR... [--text]
import difflib
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
show_text = "--text" in sys.argv
ref = json.load(open(args[0], encoding="utf-8"))["steps"]


def answer(resp):
    return resp["choices"][0]["message"].get("content") or ""


def prefix_words(a, b):
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


for run in args[1:]:
    ours = json.load(open(os.path.join(run, "answers.json"), encoding="utf-8"))
    print(f"### {os.path.basename(run.rstrip('/\\'))}")
    tot_wall = tot_out = tot_prompt = 0
    for i, (r, o) in enumerate(zip(ref, ours)):
        a, b = answer(r["response"]), answer(o["response"])
        st = o["response"].get("stats") or {}
        ru = r["response"]["usage"]
        sim = difflib.SequenceMatcher(None, a.split(), b.split()).ratio()
        tot_wall += o["wall_s"]
        tot_out += st.get("completion_tokens", 0)
        tot_prompt += st.get("prompt_tokens", 0)
        print(
            f"part {r['part']}: exact={a == b} prefix_words={prefix_words(a, b)}/{len(a.split())} "
            f"sim={sim:.2f} | ref {ru['completion_tokens']} tok, ours {st.get('completion_tokens')} tok "
            f"(prompt ref {ru['prompt_tokens']}, ours {st.get('prompt_tokens')}) | "
            f"ttft {st.get('ttft_ms', 0) / 1000:.1f} s, prefill {st.get('prefill_tok_s', 0):.0f} tok/s, "
            f"decode {st.get('decode_tok_s', 0):.1f} tok/s, accept {st.get('draft_acceptance') or 0:.2f}, "
            f"wall {o['wall_s']:.1f} s, finish {o['response']['choices'][0].get('finish_reason')}"
        )
        if show_text:
            print("  REF :", a)
            print("  OURS:", b)
    print(f"total: wall {tot_wall:.1f} s, prompt {tot_prompt} tok, out {tot_out} tok")
    mem = os.path.join(run, "mem.json")
    if os.path.exists(mem):
        m = json.load(open(mem, encoding="utf-8-sig"))
        g = 1024 ** 3
        print(
            f"memory: proc GPU shared peak {m['peak_proc_shared'] / g:.2f} GiB, dedicated {m['peak_proc_dedicated'] / g:.2f} GiB, "
            f"adapter shared {m['adapter_shared_baseline'] / g:.2f} -> {m['peak_adapter_shared'] / g:.2f} GiB, "
            f"working set peak {m['peak_working_set_os'] / g:.2f} GiB, private peak {m['peak_private_sampled'] / g:.2f} GiB"
        )
    print()
