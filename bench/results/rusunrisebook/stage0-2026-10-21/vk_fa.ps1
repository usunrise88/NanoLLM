# Flash-attention parameter sweep on the Iris Xe: a dev ggml-vulkan.dll that reads GGML_VK_DEV_FA_PP (prefill,
# n_rows > 8), GGML_VK_DEV_FA_TG (decode, n_rows <= 8) and GGML_VK_DEV_FA_CORES (split_k target when the driver
# reports no core count), llama-bench at given depths. One line per configuration in raw\<Out>.
param([string]$Dll = 'C:\Projects\nanollm\.cache\llama.cpp\out-windows_amd64\lib\ggml-vulkan.dll',
      [string[]]$Configs = @('base|'),
      [string]$P = '512', [string]$N = '0', [string]$Depths = '4096,8192', [int]$Reps = 2,
      [string]$Out = 'vk-fa.txt')
$R = 'C:\Projects\runtimes\bench-vkdev'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$env:GGML_VK_VISIBLE_DEVICES = '1'
$outf = Join-Path $PSScriptRoot "raw\$Out"
Copy-Item -Force $Dll "$R\ggml-vulkan.dll"
foreach ($c in $Configs) {
    $name, $envs = $c -split '\|', 2
    foreach ($v in @('GGML_VK_DEV_FA_PP', 'GGML_VK_DEV_FA_TG', 'GGML_VK_DEV_FA_CORES')) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
    foreach ($kv in ($envs -split ';')) {
        if ($kv -match '^([A-Z_]+)=(.*)$') { Set-Item "Env:$($Matches[1])" $Matches[2] }
    }
    Push-Location $R
    $o = & .\llama-bench.exe -m $m -ngl 99 -fa 1 -p $P -n $N -d $Depths -r $Reps -o md 2>&1 | ForEach-Object { "$_" }
    Pop-Location
    $dev = $o | Where-Object { $_ -match 'dev FA' } | ForEach-Object { ($_ -replace '^.*dev FA ', '') }
    $rows = $o | Where-Object { $_ -match '\|\s*(pp|tg)\d+' } | ForEach-Object { $f = ($_ -split '\|'); '{0}={1}' -f ($f[-3].Trim() -replace ' ', ''), ($f[-2].Trim() -split ' ')[0] }
    $line = '{0,-22} {1}' -f $name, ($rows -join '  ')
    $line
    Add-Content $outf $line
    foreach ($d in $dev) { Add-Content $outf ('    ' + $d) }
    Start-Sleep -Seconds 5
}
