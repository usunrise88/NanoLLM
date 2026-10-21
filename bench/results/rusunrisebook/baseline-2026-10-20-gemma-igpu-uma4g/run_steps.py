# Sends every step of gemma.json to a running NanoLLM server, one after
# another, and records the answer, the server's stats and the wall time.
#
#   python -I run_steps.py BASE_URL INPUT.json OUTPUT.json
import json
import sys
import time
import urllib.request

base, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
doc = json.load(open(src, encoding="utf-8"))
out = []
for i, step in enumerate(doc["steps"]):
    rq = step["request"]
    body = {
        "model": "gemma-4-e4b-qat",
        "messages": rq["messages"],
        "max_tokens": rq["max_tokens"],
        "temperature": rq.get("temperature", 0),
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.load(r)
    wall = time.perf_counter() - t0
    st = resp.get("stats") or {}
    out.append({"part": step.get("part"), "wall_s": wall, "response": resp})
    print(
        f"step {i}: prompt {st.get('prompt_tokens')} (cached {st.get('cached_tokens')}), "
        f"out {st.get('completion_tokens')}, ttft {st.get('ttft_ms', 0):.0f} ms, "
        f"prefill {st.get('prefill_tok_s', 0):.1f} tok/s, decode {st.get('decode_tok_s', 0):.2f} tok/s, "
        f"accept {st.get('draft_acceptance')}, wall {wall:.1f} s, finish {resp['choices'][0].get('finish_reason')}",
        flush=True,
    )
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
