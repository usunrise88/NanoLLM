# Stage 1: the CPU VNNI kernels (patch 0005) against the shipped library and the rebuilt unchanged series.
#   ab     : dev suite ram-6g with each of the three lib dirs, alternated, twice (default threads: 6 decode / 20 batch)
#   mtp    : the VNNI lib with MTP on the CPU (draft_max 1..3) and the n-gram drafter
#   gemma  : gemma.json end to end on the CPU: vnni nomtp, mtp2, mtp3; shipped lib nomtp as the reference
param([string[]]$Blocks = @('ab', 'mtp', 'gemma'),
      [string]$Tag = 's1')
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$raw = Join-Path $here 'raw'
$driver = Join-Path $here 'driver-s1.log'
function Log($m) { $l = '{0:HH:mm:ss} {1}' -f (Get-Date), $m; $l; try { Add-Content $driver $l } catch {} }
$exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm-dev.exe'
$libs = 'C:\Projects\runtimes\libs'
$env:GGML_VK_VISIBLE_DEVICES = '1'

function Bench([string]$Name, [string]$Cfg, [string]$Preset, [string[]]$Sets = @()) {
    $a = @('bench', '--config', "$libs\$Cfg.yaml", '--suite', 'dev', '--advisory', '--reps', '1', '--preset', $Preset, '--label', $Name, '--out', "$raw\$Name.json")
    foreach ($s in $Sets) { $a += @('--set', $s) }
    Log "bench $Name : $Cfg $Preset $($Sets -join ' ')"
    $p = Start-Process -FilePath $exe -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput "$raw\$Name.out" -RedirectStandardError "$raw\$Name.log"
    if (-not $p.WaitForExit(2400000)) { $p.Kill(); Log "  TIMEOUT $Name" }
    Get-Content "$raw\$Name.log" | Where-Object { $_ -match '^\[\d+/\d+\].*(ok|fail|slow)' } | ForEach-Object { Log "  $_" }
    Start-Sleep -Seconds 20
}

foreach ($b in $Blocks) {
    Log "=== block $b"
    switch ($b) {
        'ab' {
            foreach ($rep in 1, 2) { foreach ($c in 'ship', 'base', 'vnni') { Bench "$Tag-ab-$c-r$rep" $c 'ram-6g' } }
        }
        'mtp' {
            Bench "$Tag-mtp-vnni" 'vnni' 'ram-6g' @('draft_max=1,2,3')
            Bench "$Tag-mtp-off-vnni" 'vnni' 'ram-6g' @('mtp=off')
            Bench "$Tag-mtp-base" 'base' 'ram-6g' @('draft_max=2')
            Bench "$Tag-ngram-vnni" 'vnni' 'ram-6g' @('ngram=2,3')
        }
        'gemma' {
            foreach ($c in @(@('ship', 'nomtp'), @('vnni', 'nomtp'), @('vnni', 'mtp2'), @('vnni', 'mtp3'))) {
                $lib = if ($c[0] -eq 'ship') { 'C:\Projects\nanollm\dist\windows_amd64\lib' } else { "$libs\$($c[0])" }
                Log "gemma.json $($c[0]) $($c[1])"
                & "$here\run_gemma.ps1" -Variants $c[1] -Name "$Tag-g-$($c[0])" -LibDir $lib -Device llamacpp:cpu -Preset ram-6g -CpuThreads 6 -CpuThreadsBatch 20 2>&1 | ForEach-Object { Log "  $_" }
                Start-Sleep -Seconds 30
            }
        }
    }
}
Log 'ALL DONE'
