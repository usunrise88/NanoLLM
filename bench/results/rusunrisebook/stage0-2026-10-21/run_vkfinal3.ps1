# gemma.json on the iGPU (uma-4g): upstream ggml-vulkan.dll against the patched one (0006), MTP-2 and MTP-3, interleaved.
$here = $PSScriptRoot
$L = 'C:\Projects\runtimes\libs\vkpatch2'
$U = 'C:\Projects\nanollm\dist\windows_amd64\lib'
if (Test-Path $L) { Remove-Item -Recurse -Force $L }
Copy-Item -Recurse $U $L
Copy-Item -Force 'C:\Projects\nanollm\.cache\llama.cpp\out-windows_amd64\lib\ggml-vulkan.dll' "$L\ggml-vulkan.dll"
$log = Join-Path $here 'driver-vkfinal3.log'
if (Test-Path $log) { Remove-Item $log }
foreach ($c in @(@('B-mtp2', $L, 'mtp2'), @('A-mtp2', $U, 'mtp2'), @('B-mtp3', $L, 'mtp3'), @('A-mtp3', $U, 'mtp3'), @('B-mtp2b', $L, 'mtp2'), @('A-mtp2b', $U, 'mtp2'))) {
    & "$here\run_gemma.ps1" -Variants $c[2] -Name ('f3-' + $c[0]) -LibDir $c[1] -Device llamacpp:vulkan0 -Preset uma-4g 2>&1 | ForEach-Object { Add-Content $log ("{0} {1}" -f $c[0], $_) }
    Start-Sleep -Seconds 30
}
'ALL DONE' | Add-Content $log
