# Aggregate DRAM read bandwidth when the iGPU and the CPU read at the same time: vkbw (compute shader, 1 GiB buffer)
# alone, membw (AVX2, CPU threads) alone, then both together. If the sum stays at the CPU's 37 GB/s, splitting
# matrices between the two devices cannot speed decode up.
$here = $PSScriptRoot
$out = Join-Path $here 'raw\bwmix.txt'
$vk = Join-Path $here 'vkbw'
$membw = 'C:\Projects\runtimes\bin\membw.exe'
'--- vkbw alone' | Set-Content $out
Push-Location $vk
& .\vkbw.exe read.spv 2>&1 | Add-Content $out
Pop-Location
Start-Sleep -Seconds 5
'--- membw alone' | Add-Content $out
& $membw -scan 4,6,8,14 -sec 4 2>&1 | ForEach-Object { "$_" } | Add-Content $out
Start-Sleep -Seconds 5
foreach ($t in @(4, 6, 8, 14)) {
    "--- together, membw $t threads" | Add-Content $out
    $job = Start-Job -ScriptBlock { param($d) Set-Location $d; & .\vkbw.exe read.spv 2>&1 } -ArgumentList $vk
    Start-Sleep -Seconds 4
    & $membw -scan $t -sec 20 2>&1 | ForEach-Object { "$_" } | Add-Content $out
    Receive-Job -Wait $job | Add-Content $out
    Remove-Job $job
    Start-Sleep -Seconds 5
}
'DONE' | Add-Content $out
