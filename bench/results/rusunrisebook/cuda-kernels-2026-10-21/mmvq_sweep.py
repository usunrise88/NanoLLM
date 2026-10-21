"""Q4_0 mat-vec of the Gemma 4 E4B shapes under one GGML_CUDA_NANO_* setting: test-backend-ops perf, min us/run.

  python mmvq_sweep.py --tools <dir with test-backend-ops.exe> --env MMVQ_NWARPS --values 0 2 3 5 6 8 [--n 1] [--rounds 2]

Each shape runs every value forward and then backward (A B C C B A), `rounds` times, and keeps the fastest run of
each value: the laptop GPU throttles with heat, and the minimum is the least affected. The matrices are the ones in
gemma4_e4b_decode_shapes() of tests/test-backend-ops.cpp (m output rows, k inputs); GB/s counts the Q4_0 weights.
"""
import argparse
import os
import re
import subprocess
import sys

SHAPES = [  # (m, k, what)
    (2048, 2560, "q swa"), (4096, 2560, "q full"), (512, 2560, "k/v swa"), (1024, 2560, "k/v full"),
    (2560, 2048, "o swa"), (2560, 4096, "o full"), (10240, 2560, "ffn gate/up"), (2560, 10240, "ffn down"),
    (256, 2560, "ple gate"), (2560, 256, "ple proj"), (10752, 2560, "ple model proj"), (262144, 2560, "head"),
]
RUN = re.compile(r"([0-9]+) runs -\s+([0-9.]+) us/run")


def run(tools, env_name, value, m, k, n, op="MUL_MAT"):
    env = dict(os.environ)
    for e in env_name.split(","):
        env["GGML_CUDA_NANO_" + e] = str(value)
    if op == "fusion":  # the feed-forward gate and up with the GEGLU: m is the tokens, n the rows in that test
        args = ["-o", "MUL_MAT_VEC_FUSION", "-p", f"type=q4_0,.*m={n},n={m},k={k},"]
    else:
        args = ["-o", "MUL_MAT", "-p", f"type_a=q4_0,type_b=f32,m={m},n={n},k={k},"]
    p = subprocess.run(
        [os.path.join(tools, "test-backend-ops.exe"), "perf", "-b", "CUDA0"] + args,
        env=env, capture_output=True, text=True, timeout=300)
    mm = RUN.search(p.stdout)
    return float(mm.group(2)) if mm else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--op", default="MUL_MAT", help="MUL_MAT, or fusion for the gate/up GEGLU of 10240x2560 only")
    a = ap.parse_args()
    shapes = SHAPES
    if a.op == "fusion":
        shapes = [(10240, 2560, "gate/up geglu")]
    if a.shapes:
        want = {int(x) for x in a.shapes.split(",")}
        shapes = [s for s in SHAPES if s[0] in want]
    print(f"{'shape':<26}" + "".join(f"{v:>10}" for v in a.values) + "   best")
    for m, k, what in shapes:
        best = {v: float("inf") for v in a.values}
        order = list(a.values) + list(reversed(a.values))
        for _ in range(a.rounds):
            for v in order:
                t = run(a.tools, a.env, v, m, k, a.n, a.op)
                if t < best[v]:
                    best[v] = t
        nbytes = m * (k // 32) * 18 * (2 if a.op == "fusion" else 1)
        cells = []
        for v in a.values:
            cells.append(f"{best[v]:7.1f}us" if best[v] < 1e9 else "     n/a")
        gbs = [nbytes / (best[v] * 1e-6) / 1e9 for v in a.values if best[v] < 1e9]
        b = min(a.values, key=lambda v: best[v])
        print(f"{what + ' ' + str(m) + 'x' + str(k):<26}" + "".join(f"{c:>10}" for c in cells) +
              f"   {b} ({max(gbs):.0f} GB/s)")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
