# The batch mat-vec (2-8 tokens: the verification of MTP drafts) on the Iris Xe: workgroup variant and rows per workgroup of
# the q8_1 (MMVQ) pipelines, with a dev build of ggml-vulkan.dll (GGML_VK_DEV_MMVQ_WG 0 subgroup / 1 large / 2 large up to
# 8192 rows, GGML_VK_DEV_RMINT rows per workgroup, GGML_VK_DEV_LARGE_M as before). llama-bench pp2, pp3, pp4, pp6.
param([string]$Out = 'vk-mmvq.txt', [string[]]$Cases = @())
$R = 'C:\Projects\runtimes\bench-vkdev'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$env:GGML_VK_VISIBLE_DEVICES = '1'
$outf = Join-Path $PSScriptRoot "raw\$Out"
foreach ($c in $Cases) {
    $name, $wg, $rm, $lm = $c -split '\|'
    foreach ($v in 'GGML_VK_DEV_MMVQ_WG', 'GGML_VK_DEV_RMINT', 'GGML_VK_DEV_LARGE_M') { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
    if ($wg) { $env:GGML_VK_DEV_MMVQ_WG = $wg }
    if ($rm) { $env:GGML_VK_DEV_RMINT = $rm }
    if ($lm) { $env:GGML_VK_DEV_LARGE_M = $lm }
    Push-Location $R
    $o = & .\llama-bench.exe -m $m -ngl 99 -p 1,2,3,4,6 -n 0 -r 3 -o md 2>&1
    Pop-Location
    $rows = $o | ForEach-Object { "$_" } | Where-Object { $_ -match '\|\s*pp\d+\s*\|' } | ForEach-Object { $f = ($_ -split '\|'); '{0}={1}' -f $f[-3].Trim(), ($f[-2].Trim() -split ' ')[0] }
    $line = '{0,-16} {1}' -f $name, ($rows -join '  ')
    $line; Add-Content $outf $line
    Start-Sleep -Seconds 5
}
foreach ($v in 'GGML_VK_DEV_MMVQ_WG', 'GGML_VK_DEV_RMINT', 'GGML_VK_DEV_LARGE_M') { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
