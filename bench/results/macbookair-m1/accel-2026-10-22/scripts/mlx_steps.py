# Runs the steps of gemma.json through mlx-lm in one process, one after another, greedy, and prints the same
# per-step line as run_steps.py (prompt tokens, output tokens, TTFT, prefill and decode tok/s, wall).
#
#   python -I mlx_steps.py MODEL_DIR INPUT.json OUTPUT.json [prefill_step_size]
import json
import struct
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model_dir, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
step_size = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
host_ple = "host-ple" in sys.argv[5:]
no_think = "no-think" in sys.argv[5:]
mx.set_cache_limit(512 << 20)


class HostPLE:
    """The per-layer embedding table read row by row from the mmapped safetensors file on the CPU (like NanoLLM's
    M3 mode keeps PLE in the file), instead of 1.48 GiB of it resident in GPU memory."""

    def __init__(self, path, prefix, group_size, bits):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        base = 8 + n

        def view(name, dtype):
            t = header[prefix + name]
            a, b = t["data_offsets"]
            return np.memmap(path, dtype=dtype, mode="r", offset=base + a, shape=tuple(t["shape"]))

        self.w = view("weight", np.uint32)
        self.s = view("scales", np.uint16)
        self.b = view("biases", np.uint16)
        self.group_size, self.bits = group_size, bits

    @staticmethod
    def _bf16(u16):
        return (u16.astype(np.uint32) << 16).view(np.float32)

    def __call__(self, ids):
        idx = np.array(ids, copy=False).reshape(-1)
        w = self.w[idx]  # [n, cols/8] uint32, 8 nibbles each, lowest first
        shifts = np.arange(0, 32, self.bits, dtype=np.uint32)
        q = ((w[:, :, None] >> shifts) & ((1 << self.bits) - 1)).reshape(len(idx), -1).astype(np.float32)
        q = q.reshape(len(idx), -1, self.group_size)
        out = q * self._bf16(self.s[idx])[:, :, None] + self._bf16(self.b[idx])[:, :, None]
        out = out.reshape(*ids.shape, -1)
        return mx.array(out).astype(mx.bfloat16)


t0 = time.perf_counter()
model, tok = load(model_dir)
if host_ple:
    inner = model.language_model.model
    emb = inner.embed_tokens_per_layer
    hp = HostPLE(model_dir + "/model.safetensors", "language_model.model.embed_tokens_per_layer.",
                 emb.group_size, emb.bits)
    probe = mx.array([[0, 7, 1234, 262143]])
    err = mx.abs(hp(probe).astype(mx.float32) - emb(probe).astype(mx.float32)).max().item()
    inner.embed_tokens_per_layer = hp
    del emb
    mx.clear_cache()
    print(f"host PLE: max abs difference against the GPU table {err:.3g}", flush=True)
print(f"load {time.perf_counter() - t0:.1f} s, active {mx.get_active_memory() / 2**30:.2f} GiB", flush=True)

doc = json.load(open(src, encoding="utf-8"))
greedy = make_sampler(temp=0.0)
out = []
total = 0.0
for i, step in enumerate(doc["steps"]):
    rq = step["request"]
    prompt = tok.apply_chat_template(rq["messages"], add_generation_prompt=True)
    if no_think:
        # An empty thought channel, as the template writes when thinking is off after a tool response.
        prompt = list(prompt) + tok.encode("<|channel>thought\n<channel|>", add_special_tokens=False)
    t0 = time.perf_counter()
    ttft = None
    text = []
    last = None
    for r in stream_generate(model, tok, prompt, max_tokens=rq["max_tokens"], sampler=greedy,
                             prefill_step_size=step_size):
        if ttft is None:
            ttft = time.perf_counter() - t0
        text.append(r.text)
        last = r
    wall = time.perf_counter() - t0
    total += wall
    out.append({"part": step.get("part"), "wall_s": wall, "text": "".join(text)})
    print(
        f"step {i}: prompt {last.prompt_tokens}, out {last.generation_tokens}, ttft {ttft * 1000:.0f} ms, "
        f"prefill {last.prompt_tps:.1f} tok/s, decode {last.generation_tps:.2f} tok/s, wall {wall:.1f} s, "
        f"finish {last.finish_reason}, peak {last.peak_memory:.2f} GB",
        flush=True,
    )
print(f"total wall {total:.1f} s", flush=True)
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
