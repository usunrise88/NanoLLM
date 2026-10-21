# Sends every step of gemma.json to any OpenAI-compatible server in streaming
# mode, one after another, and records for each step the text, the wall time,
# the time to the first content token, the token counts and whatever timing
# block the server adds. The same client is used for every runtime, so the
# numbers measure the same thing.
#
#   python -I run_steps_stream.py BASE_URL INPUT.json OUTPUT.json [MODEL]
import http.client
import json
import sys
import time
import urllib.parse

base, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
model = sys.argv[4] if len(sys.argv) > 4 else "gemma-4-e4b"
doc = json.load(open(src, encoding="utf-8"))
u = urllib.parse.urlparse(base)
out = []
for i, step in enumerate(doc["steps"]):
    rq = step["request"]
    body = {
        "model": model,
        "messages": rq["messages"],
        "max_tokens": rq["max_tokens"],
        "temperature": rq.get("temperature", 0),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body).encode("utf-8"),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise SystemExit(f"step {i}: HTTP {resp.status} {resp.read()[:500]!r}")
    text, chunks, ttft, usage, timings, finish = [], 0, None, None, None, None
    t_last = t0
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            usage = ev["usage"]
        if ev.get("timings"):
            timings = ev["timings"]
        for ch in ev.get("choices") or []:
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = ch.get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
                t_last = time.perf_counter()
                text.append(piece)
    wall = time.perf_counter() - t0
    conn.close()
    n_out = (usage or {}).get("completion_tokens") or chunks
    n_prompt = (usage or {}).get("prompt_tokens")
    # From the first token to the end of the stream (the stream may deliver
    # text in bursts, so the last text chunk is not the end of generation).
    gen_s = wall - (ttft or 0)
    rec = {
        "part": step.get("part"), "wall_s": wall, "ttft_s": ttft, "prompt_tokens": n_prompt,
        "completion_tokens": n_out, "content_chunks": chunks, "finish": finish,
        # (n-1) tokens are generated after the first one arrives.
        "decode_tok_s": ((n_out - 1) / gen_s) if gen_s > 0 and n_out > 1 else None,
        "usage": usage, "timings": timings, "text": "".join(text),
    }
    out.append(rec)
    d = rec["decode_tok_s"]
    print(f"step {i}: prompt {n_prompt}, out {n_out}, ttft {ttft or 0:.2f} s, decode "
          f"{d if d is None else round(d, 2)} tok/s, wall {wall:.1f} s, finish {finish}", flush=True)
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
