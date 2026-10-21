# gemma.json with flash attention on and off on the CPU (ram-6g, planner defaults: MTP 2 / p_min 0.5) and on the
# iGPU (uma-4g, MTP 2), plus small (patch 0007) on the iGPU; the dist n6 libraries with the 0007 llama.dll,
# interleaved rounds. Directories gemma\fm<round>-<config>-<variant>.
param([int]$Rounds = 2)
$here = $PSScriptRoot
$L = 'C:\Projects\runtimes\libs\fasmall'
$log = Join-Path $here 'driver-famatrix.log'
$cfgs = @(
    @('cpu-on', 'llamacpp:cpu', 'ram-6g', 'default', 'on'),
    @('vk-on', 'llamacpp:vulkan0', 'uma-4g', 'mtp2', 'on'),
    @('cpu-off', 'llamacpp:cpu', 'ram-6g', 'default', 'off'),
    @('vk-off', 'llamacpp:vulkan0', 'uma-4g', 'mtp2', 'off'),
    @('vk-small', 'llamacpp:vulkan0', 'uma-4g', 'mtp2', 'small')
)
for ($i = 1; $i -le $Rounds; $i++) {
    foreach ($c in $cfgs) {
        & "$here\run_gemma.ps1" -Variants $c[3] -Name ("fm$i-" + $c[0]) -LibDir $L -Device $c[1] -Preset $c[2] -FlashAttention $c[4] 2>&1 |
            ForEach-Object { try { Add-Content $log ("{0} {1}" -f $c[0], $_) } catch {}; $_ }
        Start-Sleep -Seconds 30
    }
}
'ALL DONE' | Add-Content $log
