# Decode (tg) of the Iris Xe against the mat-vec workgroup choice: llama-bench tg32 and pp1/pp3 (the speculative
# verification shape) with a dev build of ggml-vulkan.dll that reads GGML_VK_DEV_LARGE_M (the 64-lane variant is used
# up to this many rows; upstream 8192) and GGML_VK_DEV_RM (rows per workgroup of the Q4_0 mat-vec, upstream 2 on Intel).
param([string]$Out = 'vk-dmmv.txt', [string[]]$Cases = @())
$R = 'C:\Projects\runtimes\bench-vkdev'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$env:GGML_VK_VISIBLE_DEVICES = '1'
$outf = Join-Path $PSScriptRoot "raw\$Out"
if (-not $Cases) {
    $Cases = @('default||', 'large-all|10000000|', 'large-all-rm1|10000000|1', 'large-all-rm4|10000000|4', 'rm1||1', 'rm4||4', 'large-none|0|', 'default||')
}
foreach ($c in $Cases) {
    $name, $lm, $rm, $wm = $c -split '\|'
    if ($lm) { $env:GGML_VK_DEV_LARGE_M = $lm } else { Remove-Item Env:GGML_VK_DEV_LARGE_M -ErrorAction SilentlyContinue }
    if ($rm) { $env:GGML_VK_DEV_RM = $rm } else { Remove-Item Env:GGML_VK_DEV_RM -ErrorAction SilentlyContinue }
    if ($wm) { $env:GGML_VK_DEV_WGMULT = $wm } else { Remove-Item Env:GGML_VK_DEV_WGMULT -ErrorAction SilentlyContinue }
    Push-Location $R
    $o = & .\llama-bench.exe -m $m -ngl 99 -p 1,3 -n 32 -r 3 -o md 2>&1
    Pop-Location
    $rows = $o | ForEach-Object { "$_" } | Where-Object { $_ -match '\|\s*(pp|tg)\d+\s*\|' } | ForEach-Object { $f = ($_ -split '\|'); '{0}={1}' -f $f[-3].Trim(), ($f[-2].Trim() -split ' ')[0] }
    $line = '{0,-16} {1}' -f $name, ($rows -join '  ')
    $line; Add-Content $outf $line
    Start-Sleep -Seconds 5
}
Remove-Item Env:GGML_VK_DEV_LARGE_M, Env:GGML_VK_DEV_RM, Env:GGML_VK_DEV_WGMULT -ErrorAction SilentlyContinue


