# Same-day iGPU reference on gemma.json (uma-4g, MTP-2), twice.
$here = $PSScriptRoot
$driver = Join-Path $here 'driver-s2.log'
function Log($m) { $l = '{0:HH:mm:ss} {1}' -f (Get-Date), $m; $l; try { Add-Content $driver $l } catch {} }
foreach ($n in 'vk-g-1', 'vk-g-2') {
    Log "gemma.json $n"
    & "$here\run_gemma.ps1" -Variants mtp2 -Name $n -Device llamacpp:vulkan0 -Preset uma-4g 2>&1 | ForEach-Object { Log "  $_" }
    Start-Sleep -Seconds 30
}
Log 'ALL DONE'
