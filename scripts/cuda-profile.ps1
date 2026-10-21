#Requires -Version 5.1
<#
.SYNOPSIS
  Where the GPU time of a NanoLLM run goes, kernel by kernel: Nsight Systems around `nanollm bench`, then the
  cuda_gpu_kern_sum report as a table.

.DESCRIPTION
  .\scripts\cuda-profile.ps1 -Name base                                   # the dev suite once, default settings
  .\scripts\cuda-profile.ps1 -Name pmin -Set 'draft_p_min=0.5'            # one variant (key=value pairs of bench --set)
  .\scripts\cuda-profile.ps1 -Name nofuse -Env 'GGML_CUDA_DISABLE_FUSION=1'

  The worker process the bench starts is a child of the traced one, and nsys follows it. The dev suite runs pp512,
  tg128, chat, MTP chat and echo one after the other, so one trace holds them all; the kernel names tell the batch
  width apart (mul_mat_vec_q<Q4_0,1,...> is a token, <Q4_0,4,...> a verify of four). CUDA graphs are traced per node.
  Waits for the GPU to cool first (a throttled GPU changes every number).

  Output (default <repo>\bench\results\<machine>\cuda-kernels-<date>\raw): nsys-<Name>.nsys-rep, nsys-<Name>.json (the
  bench result) and nsys-<Name>_cuda_gpu_kern_sum.csv; the table goes to the console and nsys-<Name>.txt.
#>
param(
    [Parameter(Mandatory)] [string]$Name,
    [string]$Exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm.exe',
    [string]$Config = 'C:\Projects\nanollm\bench\results\rusunrisebook\rtx-2026-10-21\cuda-dev.yaml',
    [string]$Preset = 'vram-4g',
    [string[]]$Set = @(),
    [string[]]$Env = @(),
    [string]$OutDir = 'C:\Projects\nanollm\bench\results\rusunrisebook\cuda-kernels-2026-10-21\raw',
    [int]$CoolC = 66,
    [int]$MaxWait = 240,
    [int]$Top = 30
)
$ErrorActionPreference = 'Stop'
$cuEnv = @{}
foreach ($l in Get-Content (Join-Path (Split-Path -Parent $PSScriptRoot) '.cache\cuda-setup\cuda.env')) {
    if ($l -match '^([A-Z_]+)=(.*)$') { $cuEnv[$Matches[1]] = $Matches[2] }
}
$nsys = $cuEnv.NSYS
if (-not $nsys -or -not (Test-Path $nsys)) { throw 'no nsys: run scripts\setup-windows-cuda.cmd' }
New-Item -ItemType Directory -Force $OutDir | Out-Null

$t0 = Get-Date
while ($true) {
    $c = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    if ($c -le $CoolC -or ((Get-Date) - $t0).TotalSeconds -ge $MaxWait) { break }
    Start-Sleep -Seconds 5
}
$util = [int](nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
"gpu ${c}C, utilization ${util}% before the run"
if ($util -gt 15) { Write-Warning 'another program is using the GPU: the profile will include its kernels' }

$base = Join-Path $OutDir "nsys-$Name"
$a = @('profile', '--trace=cuda', '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none', '--force-overwrite=true', '-o', $base,
    $Exe, 'bench', '--suite', 'dev', '--config', $Config, '--preset', $Preset, '--reps', '1', '--out', "$base.json")
foreach ($kv in $Set) { $a += @('--set', $kv) }
foreach ($e in $Env) {
    $k, $v = $e -split '=', 2
    Set-Item -Path "Env:$k" -Value $v
}
$ErrorActionPreference = 'Continue'   # nsys warns on stderr; that is not an error
& $nsys @a 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match '^\[' }
$ErrorActionPreference = 'Stop'
foreach ($e in $Env) { Remove-Item -Path ("Env:" + ($e -split '=', 2)[0]) -ErrorAction SilentlyContinue }
if (-not (Test-Path "$base.nsys-rep")) { throw 'nsys wrote no report' }

& $nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true --output $base "$base.nsys-rep" | Out-Null
$csv = Get-ChildItem $OutDir -Filter "nsys-${Name}_cuda_gpu_kern_sum.csv" | Select-Object -First 1
if (-not $csv) { throw 'no kernel summary' }
$rows = Import-Csv $csv.FullName
$total = ($rows | Measure-Object -Property 'Total Time (ns)' -Sum).Sum
$table = $rows | Select-Object -First $Top | ForEach-Object {
    $n = $_.Name
    if ($n.Length -gt 100) { $n = $n.Substring(0, 100) }
    [pscustomobject]@{
        'time %' = '{0,5:N1}' -f (100 * [double]$_.'Total Time (ns)' / $total)
        'ms'     = '{0,9:N1}' -f ([double]$_.'Total Time (ns)' / 1e6)
        'count'  = $_.Instances
        'avg us' = '{0,7:N1}' -f ([double]$_.'Avg (ns)' / 1e3)
        'kernel' = $n
    }
}
$txt = $table | Format-Table -AutoSize | Out-String -Width 220
$txt = "profile $Name  preset $Preset  set [$($Set -join ' ')]  env [$($Env -join ' ')]  total kernel time {0:N0} ms`r`n{1}" -f ($total / 1e6), $txt
$txt | Set-Content -Encoding ascii "$base.txt"
$txt
