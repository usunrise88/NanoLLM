# The dev suite once per setting of the GGML_CUDA_NANO_* switches (nanollm bench --set env.*), each from a cool GPU:
# everything off (upstream's kernels), the defaults, and the defaults with one change off or on.
#   .\ablation.ps1 -Name abl1
param(
    [Parameter(Mandatory)] [string]$Name,
    [string]$Exe = 'C:\Projects\runtimes\bin\nanollm-cuda.exe',
    [int]$Reps = 2,
    [switch]$NoOff
)
$sweep = Join-Path $PSScriptRoot '..\rtx-2026-10-21\run_dev_sweep.ps1'
$cfg = Join-Path $PSScriptRoot '..\rtx-2026-10-21\cuda-dev.yaml'
$names = 'GRAPH_SHAPES', 'MMVQ_NWARPS', 'MMVQ_NWARPS_N', 'FUSE_BATCH', 'Q8_CACHE', 'NORM_LEAN', 'GELU_MUL', 'FA_VEC', 'NORM_Q8'
$off = ($names | ForEach-Object { "env.GGML_CUDA_NANO_$_=0" }) -join ';'
$variants = @()
if (-not $NoOff) { $variants += $off }
$variants += 'draft_max=3'   # the defaults
$variants += 'env.GGML_CUDA_NANO_NORM_LEAN=0'
$variants += 'env.GGML_CUDA_NANO_MMVQ_NWARPS=0;env.GGML_CUDA_NANO_MMVQ_NWARPS_N=0'
$variants += 'env.GGML_CUDA_NANO_NORM_Q8=0;env.GGML_CUDA_NANO_Q8_CACHE=0'
$variants += 'env.GGML_CUDA_NANO_GELU_MUL=0'
$variants += 'env.GGML_CUDA_NANO_GRAPH_SHAPES=0'
$variants += 'env.GGML_CUDA_NANO_FA_VEC=1'
& $sweep -Name $Name -Once -Reps $Reps -Exe $Exe -Config $cfg -MaxWait 150 -CoolC 67 -Variants $variants
