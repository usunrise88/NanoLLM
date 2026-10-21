# Stage 0c: the follow-ups of what stage 0 showed.
#   tb     : threads_batch refined and repeated (dev suite, ram-6g)
#   gemma  : gemma.json end to end on the CPU, threads 6/6 against 6/20, then a repeat of the first
#   disk   : raw read speed of both NVMe drives (tools/diskbw)
param([string[]]$Blocks = @('tb', 'gemma', 'disk'))
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$raw = Join-Path $here 'raw'
$driver = Join-Path $here 'driver-c.log'
function Log($m) { $l = '{0:HH:mm:ss} {1}' -f (Get-Date), $m; $l; try { Add-Content $driver $l } catch {} }
$exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm-dev.exe'
$env:GGML_VK_VISIBLE_DEVICES = '1'

function Bench([string]$Name, [string]$Preset, [string[]]$Sets) {
    $a = @('bench', '--suite', 'dev', '--advisory', '--reps', '1', '--preset', $Preset, '--label', $Name, '--out', "$raw\$Name.json")
    foreach ($s in $Sets) { $a += @('--set', $s) }
    Log "bench $Name : $Preset $($Sets -join ' ')"
    $p = Start-Process -FilePath $exe -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput "$raw\$Name.out" -RedirectStandardError "$raw\$Name.log"
    if (-not $p.WaitForExit(2400000)) { $p.Kill(); Log "  TIMEOUT $Name" }
    Get-Content "$raw\$Name.log" | Where-Object { $_ -match '^\[\d+/\d+\].*(ok|fail|slow)' } | ForEach-Object { Log "  $_" }
    Start-Sleep -Seconds 20
}

foreach ($b in $Blocks) {
    Log "=== block $b"
    switch ($b) {
        'tb' {
            # one value per invocation, alternated, so that a drift does not sit on one value
            foreach ($rep in 1, 2) {
                foreach ($tb in 6, 14, 16, 18, 20) { Bench "tb-$tb-r$rep" 'ram-6g' @('threads=6', "threads_batch=$tb") }
            }
        }
        'gemma' {
            foreach ($c in @(@('cpu-6-6', 6, 6), @('cpu-6-20', 6, 20), @('cpu-6-6-again', 6, 6), @('cpu-6-20-again', 6, 20))) {
                Log "gemma.json $($c[0])"
                & "$here\run_gemma.ps1" -Variants nomtp -Name $c[0] -Device llamacpp:cpu -Preset ram-6g -CpuThreads $c[1] -CpuThreadsBatch $c[2] 2>&1 | ForEach-Object { Log "  $_" }
                Start-Sleep -Seconds 30
            }
        }
        'disk' {
            $diskbw = 'C:\Projects\runtimes\bin\diskbw.exe'
            foreach ($d in @('C:\Projects\nanollm\.cache\diskbw', 'D:\diskbw')) {
                Log "diskbw $d"
                & $diskbw -dir $d 2>&1 | Tee-Object -FilePath ("$raw\diskbw-" + ($d -replace '[:\\]', '_') + '.txt') | ForEach-Object { Log "  $_" }
                Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
            }
        }
    }
}
Log 'ALL DONE'
