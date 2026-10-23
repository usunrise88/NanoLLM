#!/usr/bin/env python3
"""A carrier GGUF that measures what an encoding for the Neural Engine does to the model (docs/AUTOTUNE.md, stage 1b).

    python3 tools/coreml/reencode.py SRC.gguf DST.gguf [--attn E] [--ffn E] [--ple E] [--head E]
    python3 tools/coreml/reencode.py SRC.gguf --measure blk.0.ffn_gate.weight

The Neural Engine does not take the checkpoint's own Q4_0 (it refuses block-wise quantization), so the weights that
reach it are another encoding of the same matrices: int8 per output channel, or a look-up table of 2^N values per group
of output channels (Core ML's palettization). This script applies that encoding to the real weights and writes the
decoded values back into a GGUF as Q8_0, which llama.cpp runs on any backend: `llama-perplexity --kl-divergence`
against the reference configuration then gives the KLD of the encoding itself. Q8_0 adds a small noise of its own;
the control (every class `ctl`: Q4_0 decoded and stored as Q8_0, nothing else changed) measures it. What it does not
measure: the Neural Engine's float16 arithmetic and the Core ML graph; that is stage 1c.

E, the encoding of a class of tensors:
    keep      leave the checkpoint's Q4_0 as it is
    ctl       decode Q4_0, store as Q8_0 (the control)
    i8c       int8, one scale per output channel
    palN[:G]  N-bit look-up table (k-means, 1-D) per group of G output channels (default 16), N = 1..8
    tied      (--head only) no output.weight: the head is the checkpoint's token_embd, Q4_0
The classes: attn (attn_q, attn_k, attn_v, attn_output), ffn (ffn_gate, ffn_up, ffn_down), ple (inp_gate, proj and
per_layer_model_proj of the per-layer inputs) and head (the output matrix: a new output.weight made from token_embd,
which the host keeps reading as Q4_0 for the input embedding). per_layer_token_embd stays Q4_0: the host looks rows up.

The GGUF is written as a stream: tensor infos first, then each tensor's data as it is computed, so that neither the
source nor the carrier is ever in memory whole. A thread ends the process when its footprint (phys_footprint, which
leaves out clean mapped file pages) passes --max-footprint-gib or the system runs short of memory: an 8 GiB Mac that
runs out of both stops answering its watchdog and panics (docs/AUTOTUNE.md, "Падение Mac при сборке головы").
"""

import argparse
import ctypes
import os
import re
import struct
import subprocess
import sys
import threading
import time

import numpy as np

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType as QT
from gguf.quants import dequantize, quantize

CLASSES = [
    ("attn", re.compile(r"^blk\.\d+\.attn_(q|k|v|output)\.weight$")),
    ("ffn", re.compile(r"^blk\.\d+\.ffn_(gate|up|down)\.weight$")),
    ("ple", re.compile(r"^(blk\.\d+\.(inp_gate|proj)\.weight|per_layer_model_proj\.weight)$")),
]
HEAD_ROWS = 8192  # the head is read, encoded and written this many rows at a time
BINS = 4096


def footprint_bytes():
    """phys_footprint of this process (what the kernel counts against memory; clean mapped file pages are not in it)."""
    buf = ctypes.create_string_buffer(1024)
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    if libproc.proc_pid_rusage(os.getpid(), 4, buf) != 0:  # RUSAGE_INFO_V4
        return 0
    return struct.unpack_from("<Q", buf.raw, 16 + 8 * 7)[0]  # after the uuid: user, system, wakeups x2, pageins, wired, resident, footprint


def sysctl_int(name):
    try:
        return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10).stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def guard(step, max_gib, need_disk_gib, path):
    level = sysctl_int("kern.memorystatus_level")
    if level is not None and level < 35:
        sys.exit(f"{step}: {level}% of memory is available, 35% needed: close something first")
    import shutil
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free / 2**30
    if free < need_disk_gib:
        sys.exit(f"{step}: {free:.1f} GiB of disk free, {need_disk_gib:.1f} needed")

    def watch():
        while True:
            fp = footprint_bytes() / 2**30
            lvl = sysctl_int("kern.memorystatus_level")
            if fp > max_gib or (lvl is not None and lvl < 15):
                os.write(2, f"{step}: stopped at a footprint of {fp:.2f} GiB, {lvl}% of memory available\n".encode())
                os._exit(3)
            time.sleep(1)
    threading.Thread(target=watch, daemon=True).start()


# -- the encodings: each takes float32 weights [rows, cols] (rows are output channels) and returns what they decode to


def enc_ctl(w):
    return w


def enc_i8c(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    return np.clip(np.round(w / s), -127, 127) * s


def kmeans_1d(v, k):
    """Values v (1-D float32) replaced by the nearest of k centroids found by Lloyd's algorithm on a histogram."""
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return v
    width = (hi - lo) / BINS
    idx = np.minimum(((v - lo) / width).astype(np.int64), BINS - 1)
    cnt = np.bincount(idx, minlength=BINS).astype(np.float64)
    tot = np.bincount(idx, weights=v.astype(np.float64), minlength=BINS)
    mid = lo + (np.arange(BINS) + 0.5) * width
    cum = np.cumsum(cnt)
    pos = np.searchsorted(cum, (np.arange(k) + 0.5) / k * cum[-1])
    cent = np.unique(mid[np.minimum(pos, BINS - 1)])
    for _ in range(40):
        bounds = (cent[1:] + cent[:-1]) / 2
        a = np.searchsorted(bounds, mid)
        c = np.bincount(a, weights=cnt, minlength=len(cent))
        s = np.bincount(a, weights=tot, minlength=len(cent))
        new = np.where(c > 0, s / np.maximum(c, 1), cent)
        if np.allclose(new, cent, rtol=0, atol=width * 1e-3):
            cent = new
            break
        cent = new
    cent = np.sort(cent)
    a = np.searchsorted((cent[1:] + cent[:-1]) / 2, v)
    return cent[a].astype(np.float32)


def enc_pal(nbits, group):
    def enc(w):
        out = np.empty_like(w)
        for r in range(0, w.shape[0], group):
            blk = w[r:r + group]
            out[r:r + group] = kmeans_1d(blk.reshape(-1), 1 << nbits).reshape(blk.shape)
        return out
    return enc


def parse_encoding(spec, allow_tied=False):
    if spec in ("keep", "ctl", "i8c") or (allow_tied and spec == "tied"):
        return spec
    m = re.fullmatch(r"pal([1-8])(?::(\d+))?", spec)
    if not m:
        sys.exit(f"unknown encoding {spec!r}")
    return ("pal", int(m.group(1)), int(m.group(2) or 16))


def encoder(spec):
    if spec == "ctl":
        return enc_ctl
    if spec == "i8c":
        return enc_i8c
    return enc_pal(spec[1], spec[2])


def class_of(name):
    for cls, rx in CLASSES:
        if rx.match(name):
            return cls
    return None


def write_streamed(w, chunks, nbytes):
    """Write one tensor of nbytes as the pieces in chunks (GGUFWriter.write_tensor_data wants it whole)."""
    from gguf.gguf_writer import WriterState
    fout = w.fout[0]
    ti = w.tensors[0].pop(next(iter(w.tensors[0])))
    assert ti.nbytes == nbytes, (ti.nbytes, nbytes)
    w.write_padding(fout, fout.tell())
    done = 0
    for c in chunks:
        c.tofile(fout)
        done += c.nbytes
    assert done == nbytes, (done, nbytes)
    w.write_padding(fout, nbytes)
    w.state = WriterState.WEIGHTS


def q8_0(w):
    return quantize(np.ascontiguousarray(w, dtype=np.float32), QT.Q8_0)


def measure(src, name):
    r = GGUFReader(src, "r")
    t = next(t for t in r.tensors if t.name == name)
    w = dequantize(t.data, t.tensor_type).astype(np.float32)
    print(f"{name}: shape {w.shape}, {w.size / 1e6:.1f}M weights, rms {np.sqrt((w ** 2).mean()):.5f}")
    for spec in ["ctl", "i8c", "pal4:16", "pal4:4", "pal4:1", "pal6:16", "pal6:1", "pal8:16"]:
        t0 = time.time()
        e = encoder(parse_encoding(spec))
        d = e(w.copy())
        err = d - w
        rel = np.sqrt((err ** 2).mean()) / np.sqrt((w ** 2).mean())
        q = dequantize(q8_0(d), QT.Q8_0).astype(np.float32)
        relq = np.sqrt(((q - w) ** 2).mean()) / np.sqrt((w ** 2).mean())
        print(f"  {spec:8s} relative rms error {rel:.4%}  (after Q8_0: {relq:.4%})  {time.time() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--attn", default="ctl")
    ap.add_argument("--ffn", default="ctl")
    ap.add_argument("--ple", default="ctl")
    ap.add_argument("--head", default="tied")
    ap.add_argument("--measure", metavar="TENSOR", help="print the reconstruction error of each encoding on one tensor and exit")
    ap.add_argument("--max-footprint-gib", type=float, default=3.0)
    a = ap.parse_args()
    guard("reencode", a.max_footprint_gib, 0 if a.measure else 8, a.dst or a.src)
    if a.measure:
        return measure(a.src, a.measure)
    if not a.dst:
        sys.exit("DST is required")
    spec = {"attn": parse_encoding(a.attn), "ffn": parse_encoding(a.ffn), "ple": parse_encoding(a.ple)}
    head = parse_encoding(a.head, allow_tied=True)

    r = GGUFReader(a.src, "r")
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(a.dst, arch, use_temp_file=False)
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        sub = f.types[-1] if f.types and f.types[0] == GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), f.types[0], sub)

    plan = []  # (reader tensor, what to do): in the order the infos are written
    for t in r.tensors:
        cls = class_of(t.name)
        how = spec.get(cls) if cls else None
        if how is None or how == "keep":
            how = "copy"
        if how != "copy" and t.tensor_type != QT.Q4_0:
            sys.exit(f"{t.name} is {t.tensor_type.name}, not Q4_0")
        plan.append((t, how))
        if how == "copy":
            d = t.data
            w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, raw_dtype=t.tensor_type if d.dtype == np.uint8 else None)
        else:
            rows, cols = int(t.data.shape[0]), int(t.shape[0])
            w.add_tensor_info(t.name, (rows, cols // 32 * 34), np.uint8, rows * cols // 32 * 34, raw_dtype=QT.Q8_0)
    emb = next(t for t in r.tensors if t.name == "token_embd.weight")
    if head != "tied":
        rows, cols = int(emb.data.shape[0]), int(emb.shape[0])
        w.add_tensor_info("output.weight", (rows, cols // 32 * 34), np.uint8, rows * cols // 32 * 34, raw_dtype=QT.Q8_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    t0 = time.time()
    for i, (t, how) in enumerate(plan):
        if how == "copy":
            w.write_tensor_data(t.data)
        else:
            x = dequantize(t.data, t.tensor_type).astype(np.float32)
            q = q8_0(encoder(how)(x))
            del x
            w.write_tensor_data(q)
            del q
        if i % 40 == 0:
            print(f"{i}/{len(plan)} {t.name} {time.time() - t0:.0f} s, footprint {footprint_bytes() / 2**30:.2f} GiB", flush=True)
    if head != "tied":
        enc = encoder(head)

        def head_chunks():
            for r0 in range(0, emb.data.shape[0], HEAD_ROWS):
                x = dequantize(emb.data[r0:r0 + HEAD_ROWS], emb.tensor_type).astype(np.float32)
                q = q8_0(enc(x))
                del x
                yield q
                print(f"output.weight rows {min(r0 + HEAD_ROWS, emb.data.shape[0])}/{emb.data.shape[0]} {time.time() - t0:.0f} s", flush=True)

        rows, cols = int(emb.data.shape[0]), int(emb.shape[0])
        write_streamed(w, head_chunks(), rows * cols // 32 * 34)
    w.close()
    print(f"wrote {a.dst}: {os.path.getsize(a.dst) / 2**30:.2f} GiB in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
