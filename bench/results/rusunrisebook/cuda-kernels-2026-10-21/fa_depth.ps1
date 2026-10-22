# Plan item 1.7: the flash attention kernel of one token (D=256 layers) on Ampere, the MMA kernel of upstream (FA_VEC=0)
# against the vector kernel (FA_VEC=1), at 0, 4096 and 8192 tokens of context: llama-bench tg64 -d 0,4096,8192, A B B A,
# each run from a cool GPU. -BenchDir holds llama-bench.exe, its DLLs and the ggml-cuda.dll under test.
#   .\fa_depth.ps1
param(
    [string]$BenchDir = 'D:\n8verify\bench',
    [string]$Model = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf',
    [string]$Depths = '0,4096,8192',
    [int]$Reps = 2,
    [int]$CoolC = 66,
    [int]$MaxWait = 150
)
$raw = Join-Path $PSScriptRoot 'raw'
New-Item -ItemType Directory -Force $raw | Out-Null
function Wait-Cool {
    $t0 = Get-Date
    while ($true) {
        $c = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        if ($c -le $CoolC -or ((Get-Date) - $t0).TotalSeconds -ge $MaxWait) { return $c }
        Start-Sleep -Seconds 5
    }
}
$order = @(0, 1, 1, 0)
$i = 0
foreach ($v in $order) {
    $i++
    $c = Wait-Cool
    $env:GGML_CUDA_NANO_FA_VEC = "$v"
    $out = Join-Path $raw ("fa-depth-$i-FA_VEC=$v.csv")
    "=== [$i/4] FA_VEC=$v start $(Get-Date -Format s) gpu ${c}C"
    & (Join-Path $BenchDir 'llama-bench.exe') -m $Model -ngl 99 -fa 1 -p 0 -n 64 -d $Depths -r $Reps -o csv 2>$null | Set-Content -Encoding ascii $out
    Remove-Item Env:GGML_CUDA_NANO_FA_VEC
}
'ALL DONE'
