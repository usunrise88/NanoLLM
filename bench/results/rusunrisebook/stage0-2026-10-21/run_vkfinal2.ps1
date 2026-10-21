# gemma.json on the iGPU without MTP: upstream ggml-vulkan.dll against the patched one, to see whether the speculative
# path hides the mat-vec gain.
$here = $PSScriptRoot
$L = 'C:\Projects\runtimes\libs\vkpatch'
$U = 'C:\Projects\nanollm\dist\windows_amd64\lib'
$log = Join-Path $here 'driver-vkfinal2.log'
if (Test-Path $log) { Remove-Item $log }
foreach ($c in @(@('nA1', $U), @('nB1', $L), @('nA2', $U), @('nB2', $L))) {
    & "$here\run_gemma.ps1" -Variants nomtp -Name $c[0] -LibDir $c[1] -Device llamacpp:vulkan0 -Preset uma-4g 2>&1 | ForEach-Object { Add-Content $log ("{0} {1}" -f $c[0], $_) }
    Start-Sleep -Seconds 20
}
'ALL DONE' | Add-Content $log
