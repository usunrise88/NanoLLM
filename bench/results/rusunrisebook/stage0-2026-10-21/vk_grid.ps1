# Vulkan MMQ tile experiments on the Iris Xe: llama-bench pp512 / pp2048 with the dev override
# GGML_VK_DEV_TILES (a dev build of ggml-vulkan.dll, see STAGE notes). One run per tile; the default is run
# first, in the middle and last to show the drift.
param([string]$Out = 'vk-grid.txt', [string[]]$Cases = @())
$R = 'C:\Projects\runtimes\bench-vkdev'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$env:GGML_VK_VISIBLE_DEVICES = '1'
$outf = Join-Path $PSScriptRoot "raw\$Out"
if (-not $Cases) {
    $Cases = @(
        'default|',
        'w16|m_mmq_int=128,64,64,32,16,32,2,2,2,1,16',
        'w32|m_mmq_int=128,64,64,32,32,32,2,2,2,1,32',
        't42|m_mmq_int=128,64,64,32,8,32,2,4,2,1,8',
        't24|m_mmq_int=128,64,64,32,8,32,2,2,4,1,8',
        't44-w16|m_mmq_int=128,64,64,32,16,32,1,4,4,1,16',
        'b256-128x64-w16|m_mmq_int=256,128,64,32,16,32,2,2,2,1,16',
        'b256-64x128-w16|m_mmq_int=256,64,128,32,16,32,2,2,2,1,16',
        'b256-128x128-w16|m_mmq_int=256,128,128,32,16,64,2,4,4,1,16',
        'b64-w16|m_mmq_int=64,64,64,32,16,16,2,2,2,1,16',
        'b512-w16|m_mmq_int=512,128,128,32,16,32,1,2,2,1,16',
        'bk64|m_mmq_int=128,64,64,64,8,32,2,2,2,1,8',
        'default|'
    )
}
foreach ($c in $Cases) {
    $name, $tiles = $c -split '\|', 2
    if ($tiles) { $env:GGML_VK_DEV_TILES = $tiles } else { Remove-Item Env:GGML_VK_DEV_TILES -ErrorAction SilentlyContinue }
    Push-Location $R
    $o = & .\llama-bench.exe -m $m -ngl 99 -p 512,2048 -n 0 -r 1 -o md 2>&1
    Pop-Location
    $rows = $o | ForEach-Object { "$_" } | Where-Object { $_ -match 'pp\d+' } | ForEach-Object { $f = ($_ -split '\|'); '{0}={1}' -f $f[-3].Trim(), $f[-2].Trim().Split(' ')[0] }
    $err = ($o | ForEach-Object { "$_" } | Where-Object { $_ -match 'rror|Shared memory' } | Select-Object -First 1)
    $line = '{0,-20} {1}  {2}' -f $name, ($rows -join '  '), $err
    $line; Add-Content $outf $line
    Start-Sleep -Seconds 5
}
Remove-Item Env:GGML_VK_DEV_TILES -ErrorAction SilentlyContinue
