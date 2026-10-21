# gemma.json on the iGPU (uma-4g, MTP-2): flash attention everywhere against LLAMA_FLASH_ATTN_TYPE_SMALL_BATCH
# (patch 0007: the matmul attention for long batches on the full-attention layers), interleaved. Both use the
# same lib dir (dist n6 plus the 0007 llama.dll); the dev binary picks the mode from NANOLLM_DEV_FA_SMALL_BATCH.
param([int]$Rounds = 2, [string]$Variant = 'mtp2')
$here = $PSScriptRoot
$L = 'C:\Projects\runtimes\libs\fasmall'
if (-not (Test-Path $L)) { Copy-Item -Recurse 'C:\Projects\nanollm\dist\windows_amd64\lib' $L }
Copy-Item -Force 'C:\Projects\nanollm\.cache\llama.cpp\out-novariants\lib\llama.dll' "$L\llama.dll"
$log = Join-Path $here 'driver-fasmall.log'
for ($i = 1; $i -le $Rounds; $i++) {
    foreach ($c in @(@('fa', @()), @('small', @('NANOLLM_DEV_FA_SMALL_BATCH=1')))) {
        Remove-Item Env:NANOLLM_DEV_FA_SMALL_BATCH -ErrorAction SilentlyContinue
        & "$here\run_gemma.ps1" -Variants $Variant -Name ("fs$i-" + $c[0]) -LibDir $L -Device llamacpp:vulkan0 -Preset uma-4g -EnvPairs $c[1] 2>&1 |
            ForEach-Object { try { Add-Content $log ("{0} {1}" -f $c[0], $_) } catch {} ; $_ }
        Remove-Item Env:NANOLLM_DEV_FA_SMALL_BATCH -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 30
    }
}
'ALL DONE' | Add-Content $log
