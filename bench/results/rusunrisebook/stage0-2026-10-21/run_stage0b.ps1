# Stage 0b: llama-bench cases for the hybrid (prefill on the iGPU, decode on the
# CPU) with no patch to the scheduler: -ngl 0 with the Vulkan device in the
# list and the CPU weights not repacked, so that ggml's op offload sends the
# matrix products of batches of 32 tokens or more to the iGPU. One run per case.
#
#   powershell -ExecutionPolicy Bypass -File run_stage0b.ps1
param([string[]]$Cases = @('vk-full', 'cpu-p6', 'cpu-p6-norepack', 'hy', 'hy-ub1024', 'hy-ub2048', 'hy-nopo', 'cpu-p6-again', 'vk-full-again'),
      [string]$R = 'C:\Projects\runtimes')
$ErrorActionPreference = 'Continue'
$raw = Join-Path $PSScriptRoot 'raw'
New-Item -ItemType Directory -Force $raw | Out-Null
$driver = Join-Path $PSScriptRoot 'driver-b.log'
function Log($m) { $l = '{0:HH:mm:ss} {1}' -f (Get-Date), $m; $l; Add-Content $driver $l }
$model = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$env:GGML_VK_VISIBLE_DEVICES = '1'

# the hybrid directory: our patched CPU libraries and llama.dll plus the Vulkan one
$hy = "$R\bench-hybrid"
if (-not (Test-Path $hy)) {
    Copy-Item -Recurse "$R\bench-patched" $hy
    Copy-Item 'C:\Projects\nanollm\dist\windows_amd64\lib\ggml-vulkan.dll' $hy
}

# name = dir, llama-bench arguments, affinity mask (0 = none), environment
$cfg = [ordered]@{
    'vk-full'         = @("$R\bench-vulkan", @('-ngl', '99', '-p', '512,2048', '-n', '128'), 0, @{})
    'cpu-p6'          = @("$R\bench-patched", @('-ngl', '0', '-t', '6', '-p', '512,2048', '-n', '128'), 0x555, @{})
    'cpu-p6-norepack' = @("$R\bench-patched", @('-ngl', '0', '-t', '6', '--repack', '0', '-p', '512,2048', '-n', '128'), 0x555, @{})
    'hy'              = @($hy, @('-ngl', '0', '-dev', 'Vulkan0', '-t', '6', '--repack', '0', '-p', '512,2048', '-n', '128'), 0, @{})
    'hy-ub1024'       = @($hy, @('-ngl', '0', '-dev', 'Vulkan0', '-t', '6', '--repack', '0', '-ub', '1024', '-p', '512,2048', '-n', '128'), 0, @{})
    'hy-ub2048'       = @($hy, @('-ngl', '0', '-dev', 'Vulkan0', '-t', '6', '--repack', '0', '-ub', '2048', '-p', '512,2048', '-n', '128'), 0, @{})
    'hy-nopo'         = @($hy, @('-ngl', '0', '-dev', 'Vulkan0', '-t', '6', '--repack', '0', '-nopo', '1', '-p', '512,2048', '-n', '128'), 0, @{})
    'cpu-p6-again'    = @("$R\bench-patched", @('-ngl', '0', '-t', '6', '-p', '512,2048', '-n', '128'), 0x555, @{})
    'vk-full-again'   = @("$R\bench-vulkan", @('-ngl', '99', '-p', '512,2048', '-n', '128'), 0, @{})
}
foreach ($c in $Cases) {
    $dir, $a, $mask, $e = $cfg[$c]
    $a = @('-m', $model, '-r', '1', '-o', 'md') + $a
    Log "case $c : $($a -join ' ')"
    $out = "$raw\lb-$c.txt"
    $p = Start-Process -FilePath "$dir\llama-bench.exe" -ArgumentList $a -WorkingDirectory $dir -PassThru -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError "$out.err"
    if ($mask) { try { $p.ProcessorAffinity = [IntPtr]$mask } catch { Log "affinity: $_" } }
    if (-not $p.WaitForExit(1200000)) { $p.Kill(); Log "  TIMEOUT $c"; continue }
    Get-Content $out | Where-Object { $_ -match 'pp\d+|tg\d+' } | ForEach-Object {
        $f = ($_ -replace '[^\x20-\x7E]', '') -split '\|'
        Log ('  {0} {1}' -f $f[-3].Trim(), $f[-2].Trim())
    }
    Start-Sleep -Seconds 15
}
Log 'ALL DONE'
