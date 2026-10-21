# gemma.json on the CPU (ram-6g, planner defaults: MTP 2 / p_min 0.5): the n6 flash attention, the grouped-query
# kernel of patch 0008 (under test) and flash attention off, interleaved rounds. Directories gemma\cf<round>-<config>.
param([int]$Rounds = 2, [string]$NewDll = 'C:\Projects\nanollm\.cache\llama.cpp\out-fagqa3\lib\ggml-cpu-alderlake.dll')
$here = $PSScriptRoot
$A = 'C:\Projects\runtimes\libs\fasmall'
$B = 'C:\Projects\runtimes\libs\fagqa'
if (-not (Test-Path $B)) { Copy-Item -Recurse $A $B }
Copy-Item -Force $NewDll "$B\ggml-cpu-alderlake.dll"
$log = Join-Path $here 'driver-cpufa.log'
$cfgs = @(@('n6-on', $A, 'on'), @('gqa-on', $B, 'on'), @('n6-off', $A, 'off'))
for ($i = 1; $i -le $Rounds; $i++) {
    foreach ($c in $cfgs) {
        & "$here\run_gemma.ps1" -Variants 'default' -Name ("cf$i-" + $c[0]) -LibDir $c[1] -Device 'llamacpp:cpu' -Preset 'ram-6g' -FlashAttention $c[2] 2>&1 |
            ForEach-Object { try { Add-Content $log ("{0} {1}" -f $c[0], $_) } catch {}; $_ }
        Start-Sleep -Seconds 30
    }
}
'ALL DONE' | Add-Content $log
