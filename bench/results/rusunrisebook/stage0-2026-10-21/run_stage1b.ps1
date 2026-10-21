# Stage 1b: MTP p_min and the batch team size on gemma.json, with the VNNI library (CPU, ram-6g).
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$driver = Join-Path $here 'driver-s1b.log'
function Log($m) { $l = '{0:HH:mm:ss} {1}' -f (Get-Date), $m; $l; try { Add-Content $driver $l } catch {} }
$lib = 'C:\Projects\runtimes\libs\vnni'
foreach ($c in @(@('mtp2', 6, 20), @('mtp2p3', 6, 20), @('mtp2p5', 6, 20), @('mtp2p75', 6, 20), @('nomtp', 6, 12), @('nomtp', 6, 16), @('nomtp', 6, 20))) {
    Log "gemma.json $($c -join ' ')"
    & "$here\run_gemma.ps1" -Variants $c[0] -Name "s1b-t$($c[1])-$($c[2])" -LibDir $lib -Device llamacpp:cpu -Preset ram-6g -CpuThreads $c[1] -CpuThreadsBatch $c[2] 2>&1 | ForEach-Object { Log "  $_" }
    Start-Sleep -Seconds 30
}
Log 'ALL DONE'
