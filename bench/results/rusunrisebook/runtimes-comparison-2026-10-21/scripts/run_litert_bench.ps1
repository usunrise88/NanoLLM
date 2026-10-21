# One `litert-lm benchmark` run per configuration, with the memory of every GPU
# adapter sampled during the run (to see which GPU the runtime picks) and the
# peak working set of the python processes.
param(
    [string]$Model = 'C:\Projects\runtimes\models\litert\gemma-4-E4B-it.litertlm',
    [string]$Out = 'C:\Projects\runtimes\results\litert',
    [int]$Prefill = 1024,
    [int]$Decode = 256,
    [string[]]$Cases = @('cpu-p6', 'cpu-p6-mtp', 'gpu', 'gpu-mtp', 'gpu-4k')
)
$exe = 'C:\Projects\runtimes\litertenv\Scripts\litert-lm.exe'
New-Item -ItemType Directory -Force $Out | Out-Null
function Args-Of($case) {
    switch ($case) {
        'cpu-def'    { @{ a = @('--backend', 'cpu', '--speculative-decoding', 'false'); mask = 0; p = $Prefill; d = $Decode } }
        'cpu-p6'     { @{ a = @('--backend', 'cpu', '--speculative-decoding', 'false', '--cpu-thread-count', '6'); mask = 0x555; p = $Prefill; d = $Decode } }
        'cpu-p6-mtp' { @{ a = @('--backend', 'cpu', '--speculative-decoding', 'true', '--cpu-thread-count', '6'); mask = 0x555; p = $Prefill; d = $Decode } }
        'gpu'        { @{ a = @('--backend', 'gpu', '--speculative-decoding', 'false'); mask = 0; p = $Prefill; d = $Decode } }
        'gpu-mtp'    { @{ a = @('--backend', 'gpu', '--speculative-decoding', 'true'); mask = 0; p = $Prefill; d = $Decode } }
        'gpu-4k'     { @{ a = @('--backend', 'gpu', '--speculative-decoding', 'false'); mask = 0; p = 4096; d = 128 } }
    }
}
# The NVIDIA Vulkan driver is hidden from the loader for every run: this machine's
# discrete GPU is not part of the test.
$env:VK_LOADER_DRIVERS_DISABLE = '*nv*'
$py = 'C:\Projects\runtimes\litertenv\Scripts\python.exe'
foreach ($c in $Cases) {
    $cfg = Args-Of $c
    $txt = Join-Path $Out "$c.txt"
    $csv = Join-Path $Out "$c-gpumem.csv"
    $stop = Join-Path $Out "$c.stop"
    Remove-Item -ErrorAction SilentlyContinue $stop
    $job = Start-Job -ArgumentList $csv, $stop -ScriptBlock {
        param($csv, $stop)
        'time,adapter,shared,dedicated,python_ws' | Set-Content -Encoding ascii $csv
        while (-not (Test-Path $stop)) {
            $ws = (Get-Process python, litert-lm -ErrorAction SilentlyContinue | Measure-Object WorkingSet64 -Sum).Sum
            foreach ($a in Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory) {
                Add-Content -Encoding ascii $csv ('{0},{1},{2},{3},{4}' -f (Get-Date).ToString('HH:mm:ss.fff'), $a.Name, $a.SharedUsage, $a.DedicatedUsage, $ws)
            }
            Start-Sleep -Milliseconds 500
        }
    }
    if ($c -like 'gpu*') { Get-ChildItem "$(Split-Path $Model)\*mldrift*cache*" | Remove-Item -Force }
    $t0 = Get-Date
    $cli = @((Join-Path $PSScriptRoot 'litert_cli.py'), 'benchmark', $Model,
        '-p', "$($cfg.p)", '-d', "$($cfg.d)", '--runs', '1') + $cfg.a
    $pr = Start-Process -FilePath $py -ArgumentList $cli -PassThru -NoNewWindow -RedirectStandardOutput $txt -RedirectStandardError "$txt.err"
    if ($cfg.mask) { try { $pr.ProcessorAffinity = [IntPtr]$cfg.mask } catch {} }
    $pr.WaitForExit()
    $dt = ((Get-Date) - $t0).TotalSeconds
    New-Item -ItemType File $stop | Out-Null
    Wait-Job $job -Timeout 10 | Out-Null; Remove-Job -Force $job
    "[$c] done in $([math]::Round($dt)) s"
    Get-Content $txt -Tail 25
    ''
}
'ALL DONE'
