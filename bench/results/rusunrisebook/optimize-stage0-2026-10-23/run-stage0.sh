#!/usr/bin/env bash
# Stage 0b of the autotune plan (docs/OPTIMIZE.md, §6), measurements 1-4 on rusunrisebook, one pass each, run one after
# another (the CPU and the iGPU never run at the same time). The binary is a build of the working tree; the libraries are
# the n8 release with the CUDA 13 add-on (D:\n8rc).
#   bash run-stage0.sh /tmp/nanollm-opt.exe
set -u
EXE=${1:?path of nanollm.exe}
CFG=D:/n8rc/cfg.yaml
OUT=$(cd "$(dirname "$0")" && pwd)
cd "$OUT/../../../.."

CFG_NOCUDA=C:/Projects/runtimes/n8-nocuda-cfg.yaml # the release libraries without the CUDA add-on (CUDA_VISIBLE_DEVICES=-1 hangs the device listing)

run() { # name, then bench args; CFG is the configuration
  local name=$1; shift
  echo "=== $name $(date +%T) gpu $(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader)"
  "$EXE" bench -config "$CFG" -suite dev -advisory -machine rusunrisebook -out "$OUT/$name.json" -label "$name" "$@" > "$OUT/$name.log" 2>&1
  tail -3 "$OUT/$name.log"
}

cool() { # wait for the GPU to come down to 68 C, at most 120 s
  for _ in $(seq 24); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    [ "$t" -le 68 ] && return
    sleep 5
  done
}

SKIP_DONE=${SKIP_DONE:-}
# 1. M2 and M3 on the RTX at vram-4g, and a spill: vram-8g does not fit next to the desktop's 3 GiB.
if [ -z "$SKIP_DONE" ]; then
  cool; run m2-m3-cuda -preset vram-4g -set mode=M2,M3 -reps 3
  cool; run cuda-vram8 -preset vram-8g -reps 2
  # 3. The CPU at ram-4g (anonymous memory against the plan).
  run cpu-ram4 -preset ram-4g -reps 2
fi
# The Iris at uma-4g.
if [ -z "$SKIP_IRIS" ]; then
  CFG=$CFG_NOCUDA GGML_VK_VISIBLE_DEVICES=1 run iris-uma4 -preset uma-4g -reps 2
fi

# 4. Silent: two threads, from ram-4g down. The streaming budgets measure pp and a short decode only (-slow).
for p in ram-4g ram-3g; do
  run silent-$p -preset $p -set threads=2 -set threads_batch=2 -reps 2
done
for p in ram-2g ram-1.5g ram-1g; do
  run silent-$p -preset $p -set threads=2 -set threads_batch=2 -reps 1 -slow
done
echo "=== done $(date +%T)"
