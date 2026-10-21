# One table for all runtimes: per-request tokens/time and totals. Reads our
# baseline-summary.json and the answers.json of the streaming client.
import difflib, json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
REF = json.load(open(r"C:/Users/rusun/Downloads/Telegram Desktop/gemma.json", encoding="utf-8"))["steps"]
RES = "C:/Projects/runtimes/results"
BASE = json.load(open("C:/Projects/nanollm/bench/results/rusunrisebook/baseline-2026-10-20-gemma-igpu-uma4g/baseline-summary.json", encoding="utf-8"))["runs"]
def ref_text(i): return REF[i]["response"]["choices"][0]["message"].get("content") or ""
rows = {}
def add_stream(name, path):
    a = json.load(open(path, encoding="utf-8"))
    st = []
    for i, r in enumerate(a):
        ttft = r["ttft_s"] or 0
        st.append(dict(prompt=r["prompt_tokens"], out=r["completion_tokens"], ttft=ttft, wall=r["wall_s"],
                       prefill=(r["prompt_tokens"] / ttft if ttft else 0), decode=r["decode_tok_s"] or 0,
                       sim=difflib.SequenceMatcher(None, ref_text(i).split(), r["text"].split()).ratio()))
    rows[name] = st
for k, v in BASE.items():
    if k.endswith("base1"):
        rows["NanoLLM iGPU " + v["variant"].upper().replace("MTP", "MTP-")] = [
            dict(prompt=s["prompt_tokens"], out=s["completion_tokens"], ttft=s["ttft_s"], wall=s["wall_s"],
                 prefill=s["prefill_tok_s"], decode=s["decode_tok_s"], sim=s["sim_vs_ref"]) for s in v["steps"]]
for v in ("nomtp", "mtp2", "mtp3"):
    p = f"{RES}/ik-{v}/answers.json"
    if os.path.exists(p):
        add_stream("ik_llama CPU " + {"nomtp": "no MTP", "mtp2": "MTP-2", "mtp3": "MTP-3"}[v], p)
def add_nano(name, path):
    a = json.load(open(path, encoding="utf-8"))
    st = []
    for i, o in enumerate(a):
        s = o["response"].get("stats") or {}
        text = o["response"]["choices"][0]["message"].get("content") or ""
        st.append(dict(prompt=s.get("prompt_tokens"), out=s.get("completion_tokens"), ttft=s.get("ttft_ms", 0) / 1000,
                       wall=o["wall_s"], prefill=s.get("prefill_tok_s", 0), decode=s.get("decode_tok_s", 0),
                       sim=difflib.SequenceMatcher(None, ref_text(i).split(), text.split()).ratio()))
    rows[name] = st
for extra in sys.argv[1:]:
    name, path = extra.split("=", 1)
    (add_nano if name.startswith("NanoLLM") else add_stream)(name, path)
print(f"{'runtime':28} {'total wall s':>12} {'out tok':>8}   per request: wall s (out tok)")
for n, st in rows.items():
    print(f"{n:28} {sum(s['wall'] for s in st):12.1f} {sum(s['out'] for s in st):8d}   " +
          "  ".join(f"{s['wall']:.0f} ({s['out']})" for s in st))
print()
print(f"{'runtime':28} " + " ".join(f"p{i+1}:prefill/decode" .rjust(19) for i in range(5)))
for n, st in rows.items():
    print(f"{n:28} " + " ".join(f"{s['prefill']:7.0f} / {s['decode']:5.1f}".rjust(19) for s in st))
print()
print("similarity to the reference answers (1.0 = same words):")
for n, st in rows.items():
    print(f"{n:28} " + "  ".join(f"{s['sim']:.2f}" for s in st))
