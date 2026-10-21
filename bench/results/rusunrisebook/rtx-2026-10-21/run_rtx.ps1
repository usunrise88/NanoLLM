# The standard suite (quick choice) on the RTX 3070 Ti Laptop, Vulkan and then CUDA, a minute apart, not submitted.
# Vulkan sees only the RTX (GGML_VK_VISIBLE_DEVICES=0; the Iris Xe is vulkan1 while the RTX is awake); the CUDA lib
# directory carries no ggml-vulkan.dll, so the suite's "first GPU" is cuda0.
param(
    [string]$Exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm.exe',
    [string[]]$Backends = @('vulkan', 'cuda')
)
$here = $PSScriptRoot
$env:GGML_VK_VISIBLE_DEVICES = '0'
$first = $true
foreach ($b in $Backends) {
    if (-not $first) { Start-Sleep -Seconds 60 }
    $first = $false
    "=== $b $(Get-Date -Format s)"
    nvidia-smi --query-gpu=memory.used,pstate,temperature.gpu,power.draw --format=csv,noheader
    $a = @('bench', '--config', (Join-Path $here "$b.yaml"), '--out', (Join-Path $here "standard-$b.json"))
    & $Exe @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $here "standard-$b.log")
    "=== $b exit $LASTEXITCODE $(Get-Date -Format s)"
}
'ALL DONE'
