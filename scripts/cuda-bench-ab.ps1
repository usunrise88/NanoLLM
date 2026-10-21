#Requires -Version 5.1
<#
.SYNOPSIS
  llama-bench on the GPU for several builds or settings, each started on a cool GPU, forward and then backward
  (A B C C B A): the laptop RTX throttles within a minute, so back-to-back runs measure the heat.

.DESCRIPTION
  A variant is 'label|dir|ENV=V;ENV=V': the directory holds llama-bench.exe and the DLLs of one build (the tools
  directory of scripts\build-libs-windows.ps1 -Cuda -Tools, or an upstream bench directory), the third part is the
  environment of that run (GGML_CUDA_NANO_* switches).

  .\scripts\cuda-bench-ab.ps1 -Variants 'up|C:\Projects\runtimes\bench-cuda-n7|','ours|C:\x\tools|' -Once

  Prints pp / tg tokens per second per variant (mean of -Reps) and appends them to -Csv.
#>
param(
    [Parameter(Mandatory)] [string[]]$Variants,
    [string]$Model = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf',
    [string]$P = '512',
    [string]$N = '128',
    [string]$D = '0',
    [int]$Reps = 3,
    [int]$CoolC = 66,
    [int]$MaxWait = 240,
    [string]$Csv = '',
    [switch]$Once
)
function Wait-Cool {
    $t0 = Get-Date
    while ($true) {
        $c = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        if ($c -le $CoolC -or ((Get-Date) - $t0).TotalSeconds -ge $MaxWait) { return $c }
        Start-Sleep -Seconds 5
    }
}
$order = @($Variants)
if (-not $Once) { $order += @($Variants[($Variants.Count - 1)..0]) }
foreach ($v in $order) {
    $label, $dir, $envs = $v -split '\|', 3
    $c = Wait-Cool
    $saved = @{}
    foreach ($e in ($envs -split ';' | Where-Object { $_ })) {
        $k, $val = $e -split '=', 2
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        [Environment]::SetEnvironmentVariable($k, $val)
    }
    $out = & (Join-Path $dir 'llama-bench.exe') -m $Model -ngl 99 -p $P -n $N -d $D -r $Reps -o csv 2>$null
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    $rows = $out | ConvertFrom-Csv
    $line = ($rows | ForEach-Object {
        $t = if ($_.n_prompt -gt 0) { "pp$($_.n_prompt)" } else { "tg$($_.n_gen)" }
        if ([int]$_.n_depth -gt 0) { $t += "@$($_.n_depth)" }
        $t + ' ' + ([double]$_.avg_ts).ToString('0.0', [cultureinfo]::InvariantCulture)
    }) -join '   '
    "{0,-12} gpu {1}C   {2}" -f $label, $c, $line
    if ($Csv) { "{0},{1},{2},{3}" -f (Get-Date -Format s), $label, $c, $line | Add-Content -Encoding ascii $Csv }
}
