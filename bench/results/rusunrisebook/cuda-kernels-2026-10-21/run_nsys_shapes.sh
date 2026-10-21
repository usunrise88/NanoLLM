#!/usr/bin/env bash
# nsys profiles of llama-bench tg64 for each Q4_0 mat-vec block shape (warps*10 + rows), each from a cool GPU.
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.3.2/target-windows-x64/nsys.exe"
TOOLS=/c/Projects/nanollm/.cache/llama.cpp/out-windows_amd64-cuda/tools
for v in "$@"; do
  t0=$SECONDS; while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" -gt ${COOL:-67} ] && [ $((SECONDS-t0)) -lt 150 ]; do sleep 5; done
  python3 nsys_mmvq.py run --nsys "$NSYS" --tools "$TOOLS" --name s$v --env MMVQ_NWARPS=$v | tail -1
done
echo ALL DONE
