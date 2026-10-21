# A/B of ggml-vulkan.dll builds on the Iris Xe, interleaved: llama-bench tg32, pp1, pp3 (the speculative
# verification shape) and pp512 against the upstream DLL of the pinned tag and the patched one (0006).
param([string]$Upstream = 'C:\Projects\runtimes\libs\n4-backup\ggml-vulkan.dll',
      [string]$Patched = 'C:\Projects\nanollm\.cache\llama.cpp\out-windows_amd64\lib\ggml-vulkan.dll',
      [int]$Rounds = 3,
      [string]$Out = 'vk-ab.txt')
$R = 'C:\Projects\runtimes\bench-vkdev'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$env:GGML_VK_VISIBLE_DEVICES = '1'
$outf = Join-Path $PSScriptRoot "raw\$Out"
for ($i = 1; $i -le $Rounds; $i++) {
    foreach ($c in @(@('upstream', $Upstream), @('patched', $Patched))) {
        Copy-Item -Force $c[1] "$R\ggml-vulkan.dll"
        Push-Location $R
        $o = & .\llama-bench.exe -m $m -ngl 99 -p 1,2,3,4,6,512 -n 32 -r 3 -o md 2>&1
        Pop-Location
        $rows = $o | ForEach-Object { "$_" } | Where-Object { $_ -match '\|\s*(pp|tg)\d+\s*\|' } | ForEach-Object { $f = ($_ -split '\|'); '{0}={1}' -f $f[-3].Trim(), ($f[-2].Trim() -split ' ')[0] }
        $line = 'round {0} {1,-9} {2}' -f $i, $c[0], ($rows -join '  ')
        $line; Add-Content $outf $line
        Start-Sleep -Seconds 8
    }
}

