# Stage 0 of the throughput phase: measurements that need no C/C++ changes.
# One run per configuration (the machine's drift is handled by running the
# baseline of a group next to its variants). Results land in .\raw\<name>.json
# (nanollm bench result files) and .\raw\<name>.log; driver.log is the timeline.
#
#   powershell -ExecutionPolicy Bypass -File run_stage0.ps1 -Blocks membw,cpu-base,cpu-threads
#
# Blocks: membw cpu-base cpu-threads cpu-omp cpu-swa cpu-repack cpu-mode
#         vk-base vk-swa vk-env vk-ubatch vk-mtp
param([string[]]$Blocks = @('membw', 'cpu-base', 'cpu-threads', 'cpu-omp', 'cpu-swa', 'cpu-repack', 'cpu-mode', 'vk-base', 'vk-swa', 'vk-env', 'vk-ubatch', 'vk-mtp'),
      [string]$Exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm-dev.exe',
      [string]$Bin = 'C:\Projects\runtimes\bin')
$ErrorActionPreference = 'Continue'
$raw = Join-Path $PSScriptRoot 'raw'
New-Item -ItemType Directory -Force $raw | Out-Null
$driver = Join-Path $PSScriptRoot 'driver.log'
$env:GGML_VK_VISIBLE_DEVICES = '1'   # the RTX stays hidden

function Log($m) { $l = '{0:HH:mm:ss} {1}' -f (Get-Date), $m; $l; Add-Content $driver $l }

# Bench runs one dev-suite invocation: every --set axis is a cartesian product.
function Bench([string]$Name, [string]$Preset, [string[]]$Sets = @(), [switch]$Full, [hashtable]$EnvVars = @{}) {
    $a = @('bench', '--suite', 'dev', '--advisory', '--reps', '1', '--preset', $Preset, '--label', $Name, '--out', "$raw\$Name.json")
    if ($Full) { $a += '--full' }
    foreach ($s in $Sets) { $a += @('--set', $s) }
    $saved = @{}
    foreach ($k in $EnvVars.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $EnvVars[$k]) }
    Log "bench $Name : $Preset $($Sets -join ' ') $(if ($Full) { '--full' })"
    $t = Get-Date
    $p = Start-Process -FilePath $Exe -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput "$raw\$Name.out" -RedirectStandardError "$raw\$Name.log"
    if (-not $p.WaitForExit(2400000)) { $p.Kill(); Log "  TIMEOUT $Name" }
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    Log ("  done in {0:N0} s, exit {1}" -f ((Get-Date) - $t).TotalSeconds, $p.ExitCode)
    Get-Content "$raw\$Name.log" | Where-Object { $_ -match '^\[\d+/\d+\].*(ok|fail|slow)' } | ForEach-Object { Log "  $_" }
    Start-Sleep -Seconds 15   # let the machine settle between runs
}

foreach ($b in $Blocks) {
    Log "=== block $b"
    switch ($b) {
        'membw' {
            $m = "$Bin\membw.exe"
            $cases = @(
                @('p-cores, one thread per core (0x555)', '0x555', '1,2,3,4,5,6'),
                @('p-cores with SMT (0xFFF)', '0xFFF', '8,12'),
                @('e-cores (0xFF000)', '0xFF000', '2,4,8'),
                @('all 20 logical CPUs', '0xFFFFF', '6,10,14,20'))
            foreach ($c in $cases) {
                Log "membw $($c[0])"
                $o = & $m -scan $c[2] -mask $c[1] -mib 256 -sec 2 2>&1
                $o | Out-File -Encoding ascii "$raw\membw-$($c[1]).txt"
                $o | ForEach-Object { Log "  $_" }
            }
            Log 'membw, go kernel, six P-cores'
            & $m -scan 6 -mask 0x555 -mib 256 -sec 2 -kernel go 2>&1 | ForEach-Object { Log "  $_" }
        }
        'cpu-base' { Bench 'cpu-base' 'ram-6g' -Full }
        'cpu-threads' {
            Bench 'cpu-threads-a' 'ram-6g' @('threads=4,6,8,10,12,14')
            Bench 'cpu-threads-b' 'ram-6g' @('threads=6', 'threads_batch=10,12,14,20')
        }
        'cpu-omp' {
            Bench 'cpu-omp-blocktime' 'ram-6g' @('env.KMP_BLOCKTIME=0,200,infinite')
            Bench 'cpu-omp-wait' 'ram-6g' @('env.OMP_WAIT_POLICY=active,passive')
            Bench 'cpu-omp-bind' 'ram-6g' @('env.OMP_PLACES=cores', 'env.OMP_PROC_BIND=close,spread')
        }
        'cpu-swa' { Bench 'cpu-swa' 'ram-6g' @('swa_full=on,off') -Full }
        'cpu-repack' { Bench 'cpu-repack' 'ram-6g' @('repack=on,off') }
        'cpu-mode' { Bench 'cpu-mode' 'ram-6g' @('mode=M6,M7') }
        'vk-base' { Bench 'vk-base' 'uma-4g' -Full }
        'vk-swa' { Bench 'vk-swa' 'uma-4g' @('swa_full=on,off') -Full }
        'vk-env' {
            foreach ($e in @('GGML_VK_FORCE_MMVQ=1', 'GGML_VK_DISABLE_MMVQ=1', 'GGML_VK_DISABLE_F16=1', 'GGML_VK_PREFER_HOST_MEMORY=1',
                    'GGML_VK_MAX_NODES_PER_SUBMIT=20', 'GGML_VK_MAX_NODES_PER_SUBMIT=400', 'GGML_VK_DISABLE_FUSION=1', 'GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1')) {
                Bench ('vk-env-' + $e.Replace('GGML_VK_', '').Replace('=', '')) 'uma-4g' @('env.' + $e)
            }
        }
        'vk-ubatch' { Bench 'vk-ubatch' 'uma-4g' @('n_ubatch=256,512,1024,2048') }
        'vk-mtp' {
            Bench 'vk-mtp' 'uma-4g' @('draft_max=1,2,3,4', 'draft_p_min=0,0.5')
            Bench 'vk-ngram' 'uma-4g' @('ngram=2,3,4')
        }
    }
}
Log 'ALL DONE'
