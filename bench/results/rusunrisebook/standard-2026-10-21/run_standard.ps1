# The public standard suite on the CPU and then on the iGPU, a minute apart, sent to nano.llmto.day (--submit).
# One device at a time: the two share the memory controller.
#
# The suite runs on the first GPU the backend lists and ignores devices.default; Vulkan lists the RTX 3070 Ti first
# when it is awake, so only the Iris Xe is left visible (GGML_VK_VISIBLE_DEVICES=1). With a GPU in sight the quick
# choice is GPU presets, so the CPU run names its presets: the suite's CPU representatives and the 6 GB of this phase.
param(
    [string]$Exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm.exe',
    [string[]]$Devices = @('cpu', 'igpu')
)
$here = $PSScriptRoot
$env:GGML_VK_VISIBLE_DEVICES = '1'
$presets = @{ cpu = @('--preset', 'ram-8g', '--preset', 'ram-6g', '--preset', 'ram-4g'); igpu = @() }
$first = $true
foreach ($d in $Devices) {
    if (-not $first) { Start-Sleep -Seconds 60 }
    $first = $false
    "=== $d $(Get-Date -Format s)"
    $a = @('bench', '--config', (Join-Path $here "$d.yaml")) + $presets[$d] + @('--submit', '--out', (Join-Path $here "standard-$d.json"))
    & $Exe @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $here "standard-$d.log")
    "=== $d exit $LASTEXITCODE $(Get-Date -Format s)"
}
'ALL DONE'
