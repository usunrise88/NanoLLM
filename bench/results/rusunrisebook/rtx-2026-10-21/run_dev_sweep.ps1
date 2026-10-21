# Dev-suite variants on the RTX through CUDA, one worker per variant, each started on a cool GPU: the laptop GPU
# reaches its power cap and then 85-87 C within a minute of decode and drops its clock from 1785 to ~1425 MHz, so
# back-to-back variants measure the cooling, not the variant. Every variant waits until the GPU is at most $CoolC
# (at most $MaxWait seconds), and the list runs forward and then backward (A B C C B A).
#
#   .\run_dev_sweep.ps1 -Name mtp -Variants 'draft_max=2','draft_max=3','draft_max=4'
#   (a variant is one or more key=value pairs joined by ';', each passed as --set key=value)
param(
    [Parameter(Mandatory)] [string]$Name,
    [Parameter(Mandatory)] [string[]]$Variants,
    [string]$Exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm.exe',
    [string]$Config = (Join-Path $PSScriptRoot 'cuda.yaml'),
    [string]$Preset = 'vram-4g',
    [int]$Reps = 3,
    [int]$CoolC = 66,
    [int]$MaxWait = 240,
    [switch]$Once
)
$raw = Join-Path $PSScriptRoot 'raw'
New-Item -ItemType Directory -Force $raw | Out-Null
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
$i = 0
foreach ($v in $order) {
    $i++
    $c = Wait-Cool
    $tag = ($v -replace '[^A-Za-z0-9_.=-]', '_')
    if ($tag.Length -gt 60) { $tag = $tag.Substring(0, 40) + '_etc' }   # a long name overflows the Windows path limit
    $out = Join-Path $raw ("dev-$Name-$i-$tag.json")
    $a = @('bench', '--suite', 'dev', '--config', $Config, '--preset', $Preset, '--reps', $Reps, '--out', $out)
    foreach ($kv in ($v -split ';')) { if ($kv) { $a += @('--set', $kv) } }
    "=== [$i/$($order.Count)] $v start $(Get-Date -Format s) gpu ${c}C"
    & $Exe @a 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match '^\[' -and $_ -notmatch '…$' }
    "=== [$i/$($order.Count)] $v exit $LASTEXITCODE $(Get-Date -Format s)"
}
'ALL DONE'
