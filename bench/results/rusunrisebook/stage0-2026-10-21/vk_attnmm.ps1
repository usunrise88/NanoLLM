# Prefill attention path on the Iris Xe: flash attention for every batch (upstream) against the matmul attention for
# batches of more than 8 tokens (dev llama.dll, LLAMA_DEV_ATTN_MM=8; the SWA layers keep flash attention unless
# LLAMA_DEV_ATTN_MM_SWA is set). LLAMA_DEV_SWA_FULL=1 gives the full-size SWA cache NanoLLM plans (llama-bench sets
# swa_full off). Interleaved rounds, one line per configuration in raw\<Out>.
param([string]$Llama = 'C:\Projects\nanollm\.cache\llama.cpp\out-novariants\lib\llama.dll',
      [string]$Vk = 'C:\Projects\nanollm\.cache\ggml-vulkan-0006.dll',
      [string[]]$Configs = @('fa|', 'mm8|LLAMA_DEV_ATTN_MM=8'),
      [string]$P = '512', [string]$N = '0', [string]$Depths = '0,4096,8192', [int]$Reps = 1, [int]$Rounds = 2,
      [string]$Common = 'LLAMA_DEV_SWA_FULL=1', [string]$Out = 'vk-attnmm.txt')
$R = 'C:\Projects\runtimes\bench-vkdev'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$env:GGML_VK_VISIBLE_DEVICES = '1'
$outf = Join-Path $PSScriptRoot "raw\$Out"
if (-not (Test-Path "$R\llama.dll.orig")) { Copy-Item "$R\llama.dll" "$R\llama.dll.orig" }
Copy-Item -Force $Llama "$R\llama.dll"
Copy-Item -Force $Vk "$R\ggml-vulkan.dll"
for ($i = 1; $i -le $Rounds; $i++) {
    foreach ($c in $Configs) {
        $name, $envs = $c -split '\|', 2
        foreach ($v in @('LLAMA_DEV_ATTN_MM', 'LLAMA_DEV_ATTN_MM_SWA', 'LLAMA_DEV_SWA_FULL')) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
        foreach ($kv in (($Common + ';' + $envs) -split ';')) {
            if ($kv -match '^([A-Z_]+)=(.*)$') { Set-Item "Env:$($Matches[1])" $Matches[2] }
        }
        Push-Location $R
        $o = & .\llama-bench.exe -m $m -ngl 99 -fa 1 -p $P -n $N -d $Depths -r $Reps -o md 2>&1 | ForEach-Object { "$_" }
        Pop-Location
        $rows = $o | Where-Object { $_ -match '\|\s*(pp|tg)\d+' } | ForEach-Object { $f = ($_ -split '\|'); '{0}={1}' -f ($f[-3].Trim() -replace ' ', ''), ($f[-2].Trim() -split ' ')[0] }
        $err = $o | Where-Object { $_ -match 'error|failed|abort' } | Select-Object -First 2
        $line = 'round {0} {1,-10} {2} {3}' -f $i, $name, ($rows -join '  '), ($err -join ' ')
        $line
        Add-Content $outf $line
        Start-Sleep -Seconds 5
    }
}
Copy-Item -Force "$R\llama.dll.orig" "$R\llama.dll"
