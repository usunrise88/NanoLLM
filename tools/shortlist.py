#!/usr/bin/env python3
# OPT-4, stage A (docs/AUTOTUNE.md "OPT-4"): what a shortlist of the output head would cost
# in quality. The head is token_embd (262144 x 2560, Q4_0, 360 MiB of the ~2.5 GiB a token
# reads); computing the logits of K of its rows instead of all of them saves (1 - K/262144)
# of that. Tokens outside the list cannot be produced, so the cost is the mass the model
# gives them. This builds the list from a large ru/en/code corpus and measures that mass on
# the reference logits of the KLD gate.
#
#   tools/shortlist.py fetch  DIR [MB_PER_SITE]   # Wikipedia-family intros, random pages
#   tools/shortlist.py local  DIR [MB]            # source code and docs on this machine
#   tools/shortlist.py count  DIR MODEL.gguf      # llama-tokenize every DIR/*.txt -> DIR/counts-*.npy
#   tools/shortlist.py curve  DIR MODEL.gguf LOGITS.bin [TOOLS_DIR]
#
# The texts are CC BY-SA (Wikipedia) or the licences of the sources; only counts leave DIR.
import glob
import json
import os
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

UA = "NanoLLM-shortlist/1 (https://github.com/usunrise88/NanoLLM; token statistics)"
SITES = {  # name: (api URL, share of the per-site budget)
    "ru-wiki": ("https://ru.wikipedia.org/w/api.php", 1.0),
    "en-wiki": ("https://en.wikipedia.org/w/api.php", 1.0),
    "ru-news": ("https://ru.wikinews.org/w/api.php", 0.25),
    "ru-books": ("https://ru.wikibooks.org/w/api.php", 0.25),
    "en-books": ("https://en.wikibooks.org/w/api.php", 0.25),
}


def fetch(out, mb):
    os.makedirs(out, exist_ok=True)
    for name, (api, share) in SITES.items():
        path = os.path.join(out, name + ".txt")
        have = os.path.getsize(path) if os.path.exists(path) else 0
        want = int(mb * share * 1e6)
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as f:
            while have < want:
                q = urllib.parse.urlencode({
                    "action": "query", "generator": "random", "grnnamespace": 0, "grnlimit": 20,
                    "prop": "extracts", "explaintext": 1, "exintro": 1, "exlimit": 20, "format": "json", "formatversion": 2,
                })
                try:
                    req = urllib.request.Request(api + "?" + q, headers={"User-Agent": UA})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        pages = json.load(r).get("query", {}).get("pages", [])
                except Exception as e:  # a failed request is retried after a pause
                    print(name, "retry:", e, file=sys.stderr)
                    time.sleep(5)
                    continue
                for p in pages:
                    t = (p.get("extract") or "").strip()
                    if len(t) > 200:
                        f.write(t + "\n\n")
                        have += len(t.encode())
                time.sleep(0.4)
        print(f"{name}: {have / 1e6:.1f} MB in {time.time() - t0:.0f} s")


def local(out, mb):
    os.makedirs(out, exist_ok=True)
    roots = {
        "code-py": (glob.glob("/usr/lib/python3*/**/*.py", recursive=True), 0.4),
        "code-go": (glob.glob(os.path.expanduser("~/.local/go/src/**/*.go"), recursive=True), 0.4),
        "docs-repo": (glob.glob("docs/**/*.md", recursive=True) + glob.glob("*.md") + glob.glob("internal/**/*.go", recursive=True), 0.2),
    }
    for name, (files, share) in roots.items():
        budget, got = int(mb * share * 1e6), 0
        files.sort()
        with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8") as f:
            for p in files:
                if got >= budget:
                    break
                try:
                    if os.path.getsize(p) > 300_000:
                        continue
                    t = open(p, encoding="utf-8").read()
                except Exception:
                    continue
                f.write(t + "\n")
                got += len(t.encode())
        print(f"{name}: {got / 1e6:.1f} MB")


def read_vocab(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    f.read(4)
    _, nkv = struct.unpack("<QQ", f.read(16))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n)

    def rval(t):
        if t == 8:
            return rstr()
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if t in sz:
            return f.read(sz[t])
        if t == 9:
            at, n = struct.unpack("<IQ", f.read(12))
            return [rval(at) for _ in range(n)]
        raise Exception("type %d" % t)

    out = {}
    for _ in range(nkv):
        k = rstr().decode()
        t, = struct.unpack("<I", f.read(4))
        v = rval(t)
        if k in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type"):
            out[k] = v
        if len(out) == 2:
            break
    toks = out["tokenizer.ggml.tokens"]
    ty = [struct.unpack("<i", t)[0] for t in out["tokenizer.ggml.token_type"]]
    return toks, ty


def tokenize_counts(txt, model, tools, n):
    cnt = np.zeros(n, np.int64)
    env = dict(os.environ, LD_LIBRARY_PATH=tools)
    # 8 MB pieces keep the id list a tokenizer prints (as text) small enough to parse.
    with open(txt, encoding="utf-8") as f:
        while True:
            chunk = f.read(4_000_000)
            if not chunk:
                break
            p = subprocess.run([os.path.join(tools, "llama-tokenize"), "-m", model, "--stdin", "--ids", "--no-bos", "--log-disable"],
                               input=chunk.encode(), capture_output=True, env=env, check=True)
            ids = np.array(json.loads(p.stdout.decode().strip() or "[]"), dtype=np.int64)
            cnt += np.bincount(ids[ids < n], minlength=n)
    return cnt


def count(out, model, tools):
    toks, _ = read_vocab(model)
    n = len(toks)
    for txt in sorted(glob.glob(os.path.join(out, "*.txt"))):
        dst = os.path.join(out, "counts-" + os.path.basename(txt)[:-4] + ".npy")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        c = tokenize_counts(txt, model, tools, n)
        np.save(dst, c)
        print(f"{os.path.basename(txt)}: {c.sum()} tokens, {(c > 0).sum()} distinct, {time.time() - t0:.0f} s")


def script_mask(toks, ty):
    """Tokens made only of Latin, Cyrillic, common punctuation and symbols, plus the control kinds."""
    def ok(c):
        return (c < 0x80 or 0x400 <= c <= 0x4ff or 0x2000 <= c <= 0x206f or 0x20a0 <= c <= 0x20cf or 0x2190 <= c <= 0x22ff
                or 0x2500 <= c <= 0x25ff or 0x2600 <= c <= 0x27bf or 0x1f300 <= c <= 0x1faff or c == 0x2581
                or c in (0xa0, 0xab, 0xbb, 0xb0, 0xb7, 0xa7, 0xa9, 0x2116))
    m = np.zeros(len(toks), bool)
    for i, t in enumerate(toks):
        if ty[i] in (2, 3, 4, 5, 6):
            m[i] = True
            continue
        try:
            s = t.decode("utf-8")
        except Exception:
            continue
        if all(ok(ord(c)) for c in s):
            m[i] = True
    return m


def curve(out, model, logits, tools):
    toks, ty = read_vocab(model)
    n = len(toks)
    special = np.array([t in (2, 3, 4, 5, 6) for t in ty])
    per = {os.path.basename(p)[7:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(out, "counts-*.npy")))}
    total = sum(per.values())
    print("sources:", {k: int(v.sum()) for k, v in per.items()}, "| distinct tokens seen:", int((total > 0).sum()))
    script = script_mask(toks, ty)
    lists = {}
    order = np.argsort(-total, kind="stable")
    for k in (32768, 49152, 65536, 81920, 98304, 114688, 131072, 163840, 196608):
        m = special.copy()
        m[order[:k]] = True
        lists[f"freq{k // 1024}k"] = m
        ms = special.copy()
        ms[np.argsort(-(total * script), kind="stable")[:k]] = True
        ms &= script | special
        lists[f"freq{k // 1024}k+script"] = ms
    lists["seen"] = special | (total > 0)
    lists["seen+script"] = special | (total > 0) | script
    lists["script"] = script
    fh = open(logits, "rb")
    fh.read(8)
    n_ctx, = struct.unpack("<I", fh.read(4))
    nv, nc = struct.unpack("<ii", fh.read(8))
    tokens = np.frombuffer(fh.read(4 * n_ctx * nc), dtype=np.int32).reshape(nc, n_ctx)
    stride = 2 * ((nv + 1) // 2) + 4
    npos = n_ctx - 1 - n_ctx // 2
    names = list(lists)
    mat = np.stack([lists[k] for k in names], axis=1).astype(np.float32)  # [vocab, lists]
    res = {k: ([], [], []) for k in lists}
    for c in range(nc):
        raw = np.frombuffer(fh.read(2 * stride * npos), dtype=np.uint16).reshape(npos, stride)
        hdr = raw[:, :4].copy().view(np.float32)
        lp = hdr[:, 1:2] + hdr[:, 0:1] * raw[:, 4:4 + nv].astype(np.float32)
        p = np.exp(lp)
        am = raw[:, 4:4 + nv].argmax(1)
        nxt = tokens[c, n_ctx // 2 + 1: n_ctx // 2 + 1 + npos]
        mass = p @ mat  # the probability each list keeps, per position
        for j, k in enumerate(names):
            m = lists[k]
            res[k][0].append(-np.log(np.clip(mass[:, j], 1e-9, 1)))
            res[k][1].append(m[am])
            res[k][2].append(m[nxt])
    print(f"{'list':18s} {'K':>7s} {'rows kept':>9s} {'mean -log mass':>15s} {'p99':>8s} {'max':>7s} {'argmax in':>10s} {'next in':>9s}")
    for k, m in lists.items():
        lm, ai, ni = (np.concatenate(x) for x in res[k])
        print(f"{k:18s} {m.sum():7d} {m.sum() / n:9.1%} {lm.mean():15.5f} {np.percentile(lm, 99):8.4f} {lm.max():7.3f} {ai.mean():10.5f} {ni.mean():9.5f}")
    np.savez_compressed(os.path.join(out, "lists.npz"), **{k: m for k, m in lists.items()})


if __name__ == "__main__":
    cmd, out = sys.argv[1], sys.argv[2]
    tools = os.path.abspath("dist/linux_amd64-upstream/tools")
    if cmd == "fetch":
        fetch(out, float(sys.argv[3]) if len(sys.argv) > 3 else 15)
    elif cmd == "local":
        local(out, float(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif cmd == "count":
        count(out, sys.argv[3], tools)
    elif cmd == "curve":
        curve(out, sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else tools)
    else:
        sys.exit(__doc__)
