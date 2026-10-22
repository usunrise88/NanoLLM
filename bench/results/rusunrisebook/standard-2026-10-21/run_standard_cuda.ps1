# The public standard suite on the RTX through CUDA, sent to nano.llmto.day (--submit). Run it with the GPU quiet: no game,
# no streaming (Sunshine), no browser using the card (nvidia-smi: utilisation near 0, as little memory used as possible).
param(
    [string]$Exe = 'D:\n8rc\app\nanollm.exe'
)
$here = $PSScriptRoot
"=== cuda $(Get-Date -Format s)"
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader
& $Exe bench --config (Join-Path $here 'cuda.yaml') --submit --out (Join-Path $here 'standard-cuda.json') 2>&1 |
    ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $here 'standard-cuda.log')
"=== cuda exit $LASTEXITCODE $(Get-Date -Format s)"
