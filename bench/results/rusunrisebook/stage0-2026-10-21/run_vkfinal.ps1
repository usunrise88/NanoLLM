# gemma.json on the iGPU (uma-4g, MTP-2), upstream ggml-vulkan.dll against the patched one (0006), interleaved.
$here = $PSScriptRoot
$L = 'C:\Projects\runtimes\libs\vkpatch'
$U = 'C:\Projects\nanollm\dist\windows_amd64\lib'
if (-not (Test-Path $L)) {
    Copy-Item -Recurse $U $L
    Copy-Item -Force 'C:\Projects\nanollm\.cache\llama.cpp\out-windows_amd64\lib\ggml-vulkan.dll' "$L\ggml-vulkan.dll"
}
$log = Join-Path $here 'driver-vkfinal.log'
if (Test-Path $log) { Remove-Item $log }
foreach ($c in @(@('vkB1', $L), @('vkA1', $U), @('vkB2', $L), @('vkA2', $U))) {
    & "$here\run_gemma.ps1" -Variants mtp2 -Name $c[0] -LibDir $c[1] -Device llamacpp:vulkan0 -Preset uma-4g 2>&1 | ForEach-Object { Add-Content $log ("{0} {1}" -f $c[0], $_) }
    Start-Sleep -Seconds 30
}
'ALL DONE' | Add-Content $log
