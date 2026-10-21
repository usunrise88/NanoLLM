#!/usr/bin/env bash
# nsys profiles of llama-bench -p 4 (the shape of a draft's verification), each from a cool GPU:
#   run_nsys_n4.sh nofuse fuse 12 nf12 ...   (nofuse: FUSE_BATCH=0, upstream shapes; fuse: FUSE_BATCH=1, upstream shapes;
#   NN: fused, MMVQ_NWARPS_N=NN; nfNN: not fused, MMVQ_NWARPS_N=NN)
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.3.2/target-windows-x64/nsys.exe"
TOOLS=/c/Projects/nanollm/.cache/llama.cpp/out-windows_amd64-cuda/tools
for v in "$@"; do
  t0=$SECONDS; while [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)" -gt ${COOL:-67} ] && [ $((SECONDS-t0)) -lt 150 ]; do sleep 5; done
  case $v in
    nofuse) envs=(--env FUSE_BATCH=0 --env MMVQ_NWARPS_N=0) ;;
    fuse)   envs=(--env FUSE_BATCH=1 --env MMVQ_NWARPS_N=0) ;;
    nf*)    envs=(--env FUSE_BATCH=0 --env MMVQ_NWARPS_N=${v#nf}) ;;
    *)      envs=(--env FUSE_BATCH=1 --env MMVQ_NWARPS_N=$v) ;;
  esac
  python3 nsys_mmvq.py run --nsys "$NSYS" --tools "$TOOLS" --name n4_$v --p 4 --n 0 --r 64 "${envs[@]}" | tail -1
done
echo ALL DONE
