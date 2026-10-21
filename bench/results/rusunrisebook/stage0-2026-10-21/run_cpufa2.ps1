# gemma.json on the CPU (ram-6g, planner defaults), flash attention on: the n6 CPU backend and the grouped-query kernel
# of patch 0008, both built locally with the same compiler (VS 18 clang 22), in the order A B B A so that the
# warming of the machine within a series does not favour either. Directories gemma\cg<n>-<config>.
param([string]$OldDll = 'C:\Projects\nanollm\.cache\llama.cpp\out-n6local\lib\ggml-cpu-alderlake.dll',
      [string]$NewDll = 'C:\Projects\nanollm\.cache\llama.cpp\out-fagqa4\lib\ggml-cpu-alderlake.dll')
$here = $PSScriptRoot
$A = 'C:\Projects\runtimes\libs\n6local'
$B = 'C:\Projects\runtimes\libs\fagqa'
foreach ($d in @($A, $B)) { if (-not (Test-Path $d)) { Copy-Item -Recurse 'C:\Projects\runtimes\libs\fasmall' $d } }
Copy-Item -Force $OldDll "$A\ggml-cpu-alderlake.dll"
Copy-Item -Force $NewDll "$B\ggml-cpu-alderlake.dll"
$log = Join-Path $here 'driver-cpufa2.log'
$i = 0
foreach ($c in @(@('n6l', $A), @('gqa', $B), @('gqa', $B), @('n6l', $A))) {
    $i++
    & "$here\run_gemma.ps1" -Variants 'default' -Name ("cg$i-" + $c[0]) -LibDir $c[1] -Device 'llamacpp:cpu' -Preset 'ram-6g' -FlashAttention 'on' 2>&1 |
        ForEach-Object { try { Add-Content $log ("{0} {1}" -f $c[0], $_) } catch {}; $_ }
    Start-Sleep -Seconds 30
}
'ALL DONE' | Add-Content $log
