# gemma.json once on each backend with its best settings, a minute apart: CPU ram-6g (planner defaults: MTP 2 /
# p_min 0.5, flash attention on) and iGPU uma-4g (MTP 2, flash attention auto = small_batch through patch 0007), on
# the n6 libraries with the 0007 llama.dll. Directories gemma\final2-cpu-default, gemma\final2-igpu-mtp2.
$here = $PSScriptRoot
$L = 'C:\Projects\runtimes\libs\fasmall'
& "$here\run_gemma.ps1" -Variants 'default' -Name 'final2-cpu' -LibDir $L -Device 'llamacpp:cpu' -Preset 'ram-6g'
Start-Sleep -Seconds 60
& "$here\run_gemma.ps1" -Variants 'mtp2' -Name 'final2-igpu' -LibDir $L -Device 'llamacpp:vulkan0' -Preset 'uma-4g'
'ALL DONE'
