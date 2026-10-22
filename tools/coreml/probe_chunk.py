#!/usr/bin/env python3
"""Go/no-go probe of the Neural Engine for one chunk of Gemma 4 E4B (docs/research/coreml-ane.md, stage 1a).

    python3 tools/coreml/probe_chunk.py build VARIANT OUT.mlpackage
    python3 tools/coreml/probe_chunk.py run   VARIANT OUT.mlpackage [--units ane|gpu|cpu|all]

The chunk has the shapes of E4B's layers 0-5 (five sliding-window layers and one
full-attention layer: d=2560, FFN 10240, 8 query heads, 2 KV heads, head dim 256
and 512) and random weights. The answers it gives do not depend on the values:
how much memory the loaded chunk holds, how many operations the Neural Engine
takes, how long the first compile and a decode step take. It says nothing about
quality; the converter of stage 1b does, with the real weights. The chunk is one
decode step (one token against KV caches that arrive as inputs, the host holds
them); 42 layers are seven of these.

VARIANT is the weight encoding: fp16 (reference), pal4 and pal6 (4- and 6-bit LUT per group
of 16 channels), i8c (int8 per channel), q8b (int8 blockwise, 32), q4b (int4
blockwise, 32: Q4_0's own block shape).

Needs coremltools and numpy; `run` needs macOS 15 or later.
"""

import argparse
import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

D, FFN, NH, NKV = 2560, 10240, 8, 2
LAYERS = [  # (head_dim, kv_len): five sliding-window layers, then a full-attention one
    (256, 512), (256, 512), (256, 512), (256, 512), (256, 512), (512, 1024),
]
EPS = 1e-6
SEED = int(os.environ.get("PROBE_SEED", 20261022))  # another seed makes another chunk, for loading several at once
# For bisecting what the Neural Engine refuses: PROBE_LAYERS=1 keeps the first layers only, and
# PROBE_PARTS=ffn (any of norm, attn, ffn, comma separated) drops the other parts of each block.
if os.environ.get("PROBE_IDX"):  # PROBE_IDX=5 keeps the layers at these indices (the full-attention one is 5)
    LAYERS = [LAYERS[int(i)] for i in os.environ["PROBE_IDX"].split(",")]
else:
    LAYERS = LAYERS[:int(os.environ.get("PROBE_LAYERS", len(LAYERS)))]
PARTS = set(os.environ.get("PROBE_PARTS", "norm,attn,ffn").split(","))
Q = int(os.environ.get("PROBE_Q", 1))  # tokens per call: 1 is a decode step, 64 a prefill chunk (the attention is only costed)


VOCAB, HEAD_SLICE, HEAD_PARTS = 262144, 16384, 4
# The output head is VOCAB logits in slices of HEAD_SLICE (one convolution may not have 262144 channels) and in
# HEAD_PARTS models, so that no build holds more than a quarter of its 1.3 GiB of weights.

MAX_RSS_MIB = int(float(os.environ.get("PROBE_MAX_RSS_GIB", "3")) * 1024)
MIN_MEMORY_LEVEL = int(os.environ.get("PROBE_MIN_MEMORY_LEVEL", "35"))


def sysctl_int(name):
    try:
        return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10).stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def guard(step):
    """Keep a heavy step from taking the machine down. An 8 GiB Mac that runs out of memory and swap stops
    answering its watchdog and the kernel panics: the first build of the head here (16 slices at once, 4.6 GiB
    resident, a disk with 8 GiB free) did that on 2026-10-22. Refuses to start without memory or disk to
    spare; a thread ends the process when it grows past PROBE_MAX_RSS_GIB or the system runs short."""
    level = sysctl_int("kern.memorystatus_level")  # percent of memory the system could still hand out
    if level is not None and level < MIN_MEMORY_LEVEL:
        sys.exit(f"{step}: {level}% of memory is available, {MIN_MEMORY_LEVEL}% needed: close something first")
    disk = shutil.disk_usage(os.getcwd()).free / 2**30
    if disk < 6:
        sys.exit(f"{step}: {disk:.1f} GiB of disk free; swap needs room as well")

    def watch():
        while True:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
            lvl = sysctl_int("kern.memorystatus_level")
            if peak > MAX_RSS_MIB or (lvl is not None and lvl < 15):
                os.write(2, f"{step}: stopped at {peak:.0f} MiB resident, {lvl}% of memory available\n".encode())
                os._exit(3)
            time.sleep(0.25)
    threading.Thread(target=watch, daemon=True).start()


def weight(rng, o, i):
    w = rng.standard_normal((o, i, 1, 1), dtype=np.float32)
    w *= 0.02
    return w.astype(np.float16)


def build_head(variant, out):
    """One part of the output head (PROBE_KIND=head, PROBE_PART=0..3): 4 slices of 16384 logits."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import Function, Program, types

    part = int(os.environ.get("PROBE_PART", 0))
    rng = np.random.default_rng(SEED + part)
    prog = Program()
    with Function({"x": mb.TensorSpec((1, D, 1, 1), dtype=types.fp16)}, opset_version=ct.target.iOS18) as fn:
        outs = []
        for i in range(VOCAB // HEAD_SLICE // HEAD_PARTS):
            outs.append(mb.conv(x=fn.inputs["x"], weight=mb.const(val=weight(rng, HEAD_SLICE, D)), name=f"logits_{i}"))
        fn.set_outputs(outs)
    prog.add_function("main", fn)
    model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                       compute_precision=ct.precision.FLOAT16)
    save(model, variant, out)


def compress(model, variant):
    import coremltools.optimize.coreml as cto
    if variant == "fp16":
        return model
    if variant in ("pal4", "pal6"):
        cfg = cto.OpPalettizerConfig(mode="uniform", nbits=int(variant[3:]), granularity="per_grouped_channel", group_size=16)
        return cto.palettize_weights(model, cto.OptimizationConfig(global_config=cfg))
    kw = {"i8c": dict(dtype="int8", granularity="per_channel"),
          "q8b": dict(dtype="int8", granularity="per_block", block_size=32),
          "q4b": dict(dtype="int4", granularity="per_block", block_size=32)}[variant]
    cfg = cto.OpLinearQuantizerConfig(mode="linear_symmetric", **kw)
    return cto.linear_quantize_weights(model, cto.OptimizationConfig(global_config=cfg))


def save(model, variant, out):
    import coremltools as ct
    model = compress(model, variant)
    if os.path.exists(out):
        subprocess.run(["rm", "-rf", out], check=True)
    model.save(out)
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(out) for f in fs)
    peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    print(json.dumps({"variant": variant, "mlpackage_bytes": size, "coremltools": ct.__version__, "peak_rss_mib": peak}))


def build(variant, out):
    if os.environ.get("PROBE_KIND") == "head":
        return build_head(variant, out)
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import Function, Program, types

    rng = np.random.default_rng(SEED)

    def w(o, i):
        return weight(rng, o, i)

    G = NH // NKV
    specs = {"x": mb.TensorSpec((1, D, 1, Q), dtype=types.fp16)}
    for li, (hd, kv) in enumerate(LAYERS):
        specs[f"k{li}"] = mb.TensorSpec((1, NKV, kv, hd), dtype=types.fp16)  # K cache, held by the host
        specs[f"v{li}"] = mb.TensorSpec((1, NKV, kv, hd), dtype=types.fp16)  # V cache
        specs[f"m{li}"] = mb.TensorSpec((1, 1, G * Q if Q > 1 else 1, kv), dtype=types.fp16)  # additive mask

    prog = Program()
    with Function(specs, opset_version=ct.target.iOS18) as fn:
        x = fn.inputs["x"]

        def rms(t, name):
            sq = mb.mul(x=t, y=t)
            mean = mb.reduce_mean(x=sq, axes=[1], keep_dims=True)
            r = mb.rsqrt(x=mb.add(x=mean, y=np.float16(EPS)))
            g = mb.const(val=np.ones((1, D, 1, 1), dtype=np.float16))
            return mb.mul(x=mb.mul(x=t, y=r), y=g, name=name)

        def proj(t, o, i):
            return mb.conv(x=t, weight=mb.const(val=w(o, i)))

        outs = []
        for li, (hd, kv) in enumerate(LAYERS):
            k_cache, v_cache, mask = fn.inputs[f"k{li}"], fn.inputs[f"v{li}"], fn.inputs[f"m{li}"]
            h = rms(x, f"ln1_{li}") if "norm" in PARTS else x
            k_new = proj(h, NKV * hd, D)
            v_new = proj(h, NKV * hd, D)
            if "attn" in PARTS:
                q = proj(h, NH * hd, D)
                # grouped-query attention: the 4 query heads of a KV head are one matmul row block
                qg = mb.reshape(x=q, shape=[1, NKV, G * Q, hd])
                kt = mb.transpose(x=k_cache, perm=[0, 1, 3, 2])
                sc = mb.matmul(x=qg, y=kt)
                sc = mb.mul(x=sc, y=np.float16(1.0 / np.sqrt(hd)))
                sc = mb.softmax(x=mb.add(x=sc, y=mask), axis=-1)
                att = mb.matmul(x=sc, y=v_cache)
                att = mb.reshape(x=att, shape=[1, NH * hd, 1, Q])
                x = mb.add(x=x, y=proj(att, D, NH * hd))
            if "ffn" in PARTS:
                h = rms(x, f"ln2_{li}") if "norm" in PARTS else x
                gate = mb.gelu(x=proj(h, FFN, D), mode="TANH_APPROXIMATION")
                up = proj(h, FFN, D)
                x = mb.add(x=x, y=proj(mb.mul(x=gate, y=up), D, FFN))
            outs += [mb.identity(x=k_new, name=f"k_new_{li}"), mb.identity(x=v_new, name=f"v_new_{li}")]
        fn.set_outputs([mb.identity(x=x, name="hidden")] + outs)
    prog.add_function("main", fn)

    model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                       compute_precision=ct.precision.FLOAT16)
    save(model, variant, out)


def footprint():
    """phys_footprint of this process in bytes, from footprint(1); falls back to the peak RSS."""
    try:
        res = subprocess.run(["footprint", "-p", str(os.getpid()), "--noCategories"], capture_output=True, text=True, timeout=60)
        for line in res.stdout.splitlines():
            if "Footprint:" in line:
                num, unit = line.split("Footprint:")[1].split()[:2]
                return int(float(num) * {"KB": 1 << 10, "MB": 1 << 20, "GB": 1 << 30}[unit])
    except (OSError, subprocess.TimeoutExpired, KeyError, ValueError):
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def free_pages():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    size = 16384
    vals = {}
    for line in out.splitlines():
        if "page size of" in line:
            size = int(line.split("page size of")[1].split()[0])
        elif ":" in line:
            k, v = line.split(":")
            vals[k.strip()] = int(v.strip().rstrip("."))
    return (vals.get("Pages free", 0) + vals.get("Pages speculative", 0)) * size


def run(variant, path, units):
    import coremltools as ct
    cu = {"ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU,
          "cpu": ct.ComputeUnit.CPU_ONLY, "all": ct.ComputeUnit.ALL}[units]
    rep = {"variant": variant, "units": units, "coremltools": ct.__version__}
    free0 = free_pages()
    t = time.time()
    model = ct.models.MLModel(path, compute_units=cu)
    rep["load_s"] = round(time.time() - t, 2)  # the first load compiles the model for the device
    inputs = {}
    rng = np.random.default_rng(1)
    for spec in model.get_spec().description.input:
        shape = tuple(spec.type.multiArrayType.shape)
        inputs[spec.name] = (rng.standard_normal(shape) * 0.1).astype(np.float16)
    for name in list(inputs):
        if name.startswith("m"):  # the masks: nothing is masked out
            inputs[name] = np.zeros_like(inputs[name])
    model.predict(inputs)  # warm up
    rep["footprint_after_load_mib"] = round(footprint() / (1 << 20))
    rep["free_drop_mib"] = round((free0 - free_pages()) / (1 << 20))
    times = []
    for _ in range(40):
        t = time.time()
        model.predict(inputs)
        times.append((time.time() - t) * 1000)
    rep["step_ms_median"] = round(statistics.median(times), 2)
    rep["step_ms_p90"] = round(sorted(times)[int(len(times) * 0.9)], 2)
    rep["footprint_after_run_mib"] = round(footprint() / (1 << 20))
    try:
        from coremltools.models.compute_plan import MLComputePlan
        compiled = model.get_compiled_model_path()
        plan = MLComputePlan.load_from_path(path=compiled, compute_units=cu)
        counts = {}
        prog = plan.model_structure.program
        for op in prog.functions["main"].block.operations:
            if "const" in op.operator_name:  # const and constexpr_*: weights, not computation
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = type(usage.preferred_compute_device).__name__ if usage else "none"
            counts[dev] = counts.get(dev, 0) + 1
        total = sum(counts.values())
        rep["ops"] = counts
        rep["ane_share"] = round(sum(v for k, v in counts.items() if "Neural" in k) / total, 3) if total else 0
    except Exception as e:  # the compute plan API is new; the timings above stand without it
        rep["compute_plan_error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("variant", choices=["fp16", "pal4", "pal6", "i8c", "q8b", "q4b"])
    ap.add_argument("path")
    ap.add_argument("--units", default="ane", choices=["ane", "gpu", "cpu", "all"])
    a = ap.parse_args()
    guard(f"{a.cmd} {a.variant}")
    if a.cmd == "build":
        build(a.variant, a.path)
    else:
        run(a.variant, a.path, a.units)
