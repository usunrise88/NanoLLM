"""Per-shape kernel times of one decode, from Nsight Systems runs of llama-bench tg.

  python nsys_mmvq.py run  --nsys <nsys.exe> --tools <dir> --name base [--env MMVQ_NWARPS=12 ...] [--n 64]
  python nsys_mmvq.py show base 12 32          # compare the runs side by side (median us per launch, per token)

`run` profiles `llama-bench -p 0 -n N -r 1` (CUDA graphs traced per node) into raw/nsys-mmvq-<name>.sqlite; `show`
groups the mul_mat_vec_q launches by (template, grid) and prints the median per launch and the total ms per token.
The L2 of this GPU is 4 MB: unlike test-backend-ops perf, a real decode never finds the weights of a matrix in it.
"""
import argparse
import collections
import os
import re
import sqlite3
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
MODEL = r"C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"


def run(a):
    os.makedirs(RAW, exist_ok=True)
    base = os.path.join(RAW, f"nsys-mmvq-{a.name}")
    env = dict(os.environ)
    for e in a.env or []:
        k, v = e.split("=", 1)
        env["GGML_CUDA_NANO_" + k] = v
    cmd = [a.nsys, "profile", "--trace=cuda", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
           "--force-overwrite=true", "--export=sqlite", "-o", base,
           os.path.join(a.tools, "llama-bench.exe"), "-m", MODEL, "-ngl", "99", "-p", str(a.p), "-n", str(a.n),
           "-r", str(a.r), "-o", "csv"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if "tg" in line and "," in line and line.count(",") > 20:
            print(line.split(",")[-2:], flush=True)
    print("wrote", base + ".sqlite")


def load(name):
    con = sqlite3.connect(os.path.join(RAW, f"nsys-mmvq-{name}.sqlite"))
    names = dict(con.execute("select id, value from StringIds"))
    rows = con.execute("select demangledName, gridX, start, end from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    return names, rows


def rpb_of(name):
    """rows per block of the run named sNN (warps*10 + rows), 1 for anything else"""
    if name.startswith("s") and name[1:].isdigit():
        return int(name[1:]) % 10
    return 1


def show(a):
    runs = {}
    for name in a.names:
        names, rows = load(name)
        rpb = rpb_of(name)
        g = collections.defaultdict(list)
        for dn, gx, s, e in rows:
            nm = names.get(dn, str(dn))
            if "mul_mat_vec_q" in nm:
                tmpl = nm.split("(const void")[0].replace("void mul_mat_vec_q", "mmvq").replace("(ggml_type)", "T").replace("(bool)", "").replace("(int)", "")
                is_small_k = ", 1, 0>" in tmpl
                g[(tmpl.strip(), gx * (1 if "1, 0>" in tmpl else rpb))].append(e - s)
        # tokens in the trace: one head mat-vec (262144 rows) per token
        ntok = max(len(v) for k, v in g.items() if k[1] == 262144)
        runs[name] = (g, ntok)
    keys = sorted({k for g, _ in runs.values() for k in g}, key=lambda k: -sum(runs[a.names[0]][0].get(k, [0])))
    print(f"{'kernel':<28}{'rows':>8}" + "".join(f"{n:>26}" for n in a.names))
    tot = {n: 0 for n in a.names}
    for k in keys:
        cells = []
        for n in a.names:
            g, ntok = runs[n]
            v = g.get(k, [])
            tot[n] += sum(v) / ntok
            cells.append(f"{statistics.median(v) / 1e3:7.1f}us {sum(v) / ntok / 1e6:6.3f}ms/token" if v else "")
        print(f"{k[0][:28]:<28}{k[1]:>8}" + "".join(f"{c:>26}" for c in cells))
    print(f"{'total mmvq per token':<36}" + "".join(f"{tot[n] / 1e6:>24.3f}ms" for n in a.names))


TEMPLATE = re.compile(r"mul_mat_vec_q<\(ggml_type\)(\d+), \(int\)(\d+), \(bool\)(\d), \(bool\)(\d), \(bool\)(\d)(?:, \(int\)(\d+), \(int\)(\d+))?>")


def summary(a):
    """one line per run: the fused gate/up kernel (has_fusion), the other matrices, per evaluation; a run of
    llama-bench -p N has one evaluation per repetition (its head mat-vec has one column)"""
    print(f"{'run':<8}{'gate/up us':>12}{'ms/eval':>9}{'other ms/eval':>15}{'total ms/eval':>15}")
    for name in a.names:
        names, rows = load(name)
        fused, other, heads = [], [], 0
        for dn, gx, s_, e in rows:
            nm = names.get(dn, str(dn))
            m = TEMPLATE.search(nm)
            if not m:
                continue
            t = e - s_
            if m.group(3) == "1":
                fused.append(t)
            else:
                other.append(t)
        for dn, gx, s_, e in rows:
            if gx * rpb_of(name) == 262144 or gx == 131072 or gx == 65536:
                heads += 1
        heads = max(heads, 1)
        print(f"{name:<8}{statistics.median(fused) / 1e3 if fused else 0:>12.1f}{sum(fused) / heads / 1e6:>9.3f}"
              f"{sum(other) / heads / 1e6:>15.3f}{(sum(fused) + sum(other)) / heads / 1e6:>15.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--nsys", required=True)
    r.add_argument("--tools", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--env", action="append")
    r.add_argument("--n", type=int, default=64, help="tokens decoded one at a time")
    r.add_argument("--p", type=int, default=0, help="prompt tokens per evaluation (4 = the shape of a draft's verification)")
    r.add_argument("--r", type=int, default=1, help="repetitions")
    s = sub.add_parser("show")
    s.add_argument("names", nargs="+")
    m = sub.add_parser("summary")
    m.add_argument("names", nargs="+")
    a = ap.parse_args()
    {"run": run, "show": show, "summary": summary}[a.cmd](a)




def total(a):
    """all kernels per token (a token = one launch of the 262144-row head mat-vec), and the kernel count per token"""
    print(f"{'run':<14}{'tokens':>7}{'kernels/tok':>13}{'ms/tok (kernel time)':>22}   top kernels (ms/token)")
    for name in a.names:
        names, rows = load(name)
        ntok = max(1, sum(1 for dn, gx, s, e in rows if gx in (262144, 131072, 65536) and "mul_mat_vec_q" in names.get(dn, "")
                          and "(int)1," in names.get(dn, "")))
        cls = collections.defaultdict(float)
        tot = 0.0
        for dn, gx, s, e in rows:
            nm = names.get(dn, "")
            t = e - s
            tot += t
            if "mul_mat_vec_q" in nm:
                k = "mmvq"
            elif "rms_norm" in nm:
                k = "norm"
            elif "quantize" in nm:
                k = "quantize"
            elif "flash_attn" in nm:
                k = "flash_attn"
            else:
                k = "other"
            cls[k] += t
        top = "  ".join(f"{k} {v / ntok / 1e6:.3f}" for k, v in sorted(cls.items(), key=lambda kv: -kv[1]))
        print(f"{name:<14}{ntok:>7}{len(rows) / ntok:>13.0f}{tot / ntok / 1e6:>22.3f}   {top}")


_old_main = main


def main():  # noqa: F811
    if len(sys.argv) > 1 and sys.argv[1] == "total":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("names", nargs="+")
        total(ap.parse_args())
        return
    _old_main()


if __name__ == "__main__":
    main()
