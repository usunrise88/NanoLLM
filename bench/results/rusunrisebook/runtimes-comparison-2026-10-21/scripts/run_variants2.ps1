# Runs gemma.json against `nanollm serve` with a given lib dir and device
# (Vulkan, SYCL, OpenVINO ...), preset uma-4g, one server per speculative
# variant, and samples the GPU adapters' memory and the process memory.
param(
    [string[]]$Variants = @('mtp2'),
    [string]$LibDir = 'C:\Projects\nanollm\dist\windows_amd64\lib',
    [string]$Device = 'llamacpp:vulkan0',
    [string]$Name = 'nano-vulkan',
    [string]$OutRoot = 'C:\Projects\runtimes\results',
    [int]$ReadySeconds = 1500,
    [string]$Preset = 'uma-4g',
    # CPU thread budget for decode and prompt processing (0 = NanoLLM's default, one per
    # physical core) and an affinity mask for the server process (0 = none).
    [int]$CpuThreads = 0,
    [int]$Mask = 0,
    # NAME=VALUE pairs set in the server's environment.
    [string[]]$EnvPairs = @(),
    [string]$intelLuid = 'luid_0x00000000_0x0C4E07D4'
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm.exe'
$input_json = 'C:\Users\rusun\Downloads\Telegram Desktop\gemma.json'
$port = 8090
$base = "http://127.0.0.1:$port"
$data = Join-Path $OutRoot 'nano-data'
New-Item -ItemType Directory -Force $data | Out-Null

function Spec($v) {
    switch ($v) {
        'mtp3'  { "type: mtp`n        n_max: 3`n        p_min: 0.0" }
        'mtp2'  { "type: mtp`n        n_max: 2`n        p_min: 0.0" }
        'nomtp' { "type: none" }
    }
}

foreach ($v in $Variants) {
    $dir = Join-Path $OutRoot "$Name-$v"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $cfg = Join-Path $dir 'nanollm.yaml'
    $dataY = $data.Replace('\', '\\')
    $libY = $LibDir.Replace('\', '\\')
    $threadsYaml = ''
    if ($CpuThreads -gt 0) { $threadsYaml = "`n  cpu_threads: $CpuThreads`n  cpu_threads_batch: $CpuThreads" }
    @"
server:
  addr: "127.0.0.1:$port"
  data_dir: "$dataY"
auth:
  mode: open
backends:
  lib_dir: "$libY"
  enabled: [llamacpp]
devices:
  default: $Device
  fallback: false$threadsYaml
models:
  default: gemma-4-e4b-qat
  preload: [gemma-4-e4b-qat]
  overrides:
    gemma-4-e4b-qat:
      preset: $Preset
      mode: auto
      speculative:
        $(Spec $v)
registry:
  models_dir: "C:\\ProgramData\\NanoLLM\\models"
ui:
  enabled: false
log:
  level: info
  format: text
"@ | Set-Content -Encoding ascii $cfg

    $env:GGML_VK_VISIBLE_DEVICES = '1'
    foreach ($kv in $EnvPairs) { $k, $val = $kv -split '=', 2; Set-Item "Env:$k" $val }
    $p = Start-Process -FilePath $exe -ArgumentList @('serve', '-config', $cfg) -PassThru -WindowStyle Hidden `
        -RedirectStandardError (Join-Path $dir 'serve.log') -RedirectStandardOutput (Join-Path $dir 'serve.out')
    if ($Mask -gt 0) { try { $p.ProcessorAffinity = [IntPtr]$Mask } catch { "affinity: $_" } }
    $stop = Join-Path $dir 'stop'
    Remove-Item -ErrorAction SilentlyContinue $stop
    $csv = Join-Path $dir 'gpumem.csv'
    $job = Start-Job -ArgumentList $p.Id, $intelLuid, $csv, $stop -ScriptBlock {
        param($procId, $luid, $csv, $stop)
        'time,proc_shared,proc_dedicated,adapter_shared,working_set,private' | Set-Content -Encoding ascii $csv
        while (-not (Test-Path $stop)) {
            $m = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -Filter "Name LIKE 'pid_$($procId)_%'"
            $a = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -Filter "Name LIKE '$luid%'"
            $pr = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if (-not $pr) { break }
            Add-Content -Encoding ascii $csv ('{0},{1},{2},{3},{4},{5}' -f (Get-Date).ToString('HH:mm:ss.fff'),
                [int64]($m | Measure-Object SharedUsage -Sum).Sum, [int64]($m | Measure-Object DedicatedUsage -Sum).Sum,
                [int64]($a | Measure-Object SharedUsage -Sum).Sum, $pr.WorkingSet64, $pr.PrivateMemorySize64)
            Start-Sleep -Milliseconds 500
        }
    }

    $ready = $false
    for ($i = 0; $i -lt ($ReadySeconds * 2); $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "$base/readyz"
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
        if ($p.HasExited) { break }
    }
    "[$Name-$v] ready=$ready after $([math]::Round($i * 0.5, 1)) s"
    if ($ready) {
        & python -I (Join-Path $here 'run_steps.py') $base $input_json (Join-Path $dir 'answers.json')
        try { (Invoke-WebRequest -UseBasicParsing "$base/api/v1/models").Content | Set-Content -Encoding utf8 (Join-Path $dir 'models.json') } catch {}
    }
    New-Item -ItemType File $stop | Out-Null
    Wait-Job $job -Timeout 10 | Out-Null
    Remove-Job -Force $job
    $p.Refresh()
    $peakWS = $p.PeakWorkingSet64
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    $p.WaitForExit(15000) | Out-Null
    $rows = Import-Csv $csv
    $mx = { param($c) ($rows | ForEach-Object { [int64]$_.$c } | Measure-Object -Maximum).Maximum }
    $summary = [ordered]@{
        variant = "$Name-$v"; ready = $ready; peak_proc_shared = & $mx 'proc_shared'; peak_proc_dedicated = & $mx 'proc_dedicated'
        peak_adapter_shared = & $mx 'adapter_shared'; peak_working_set_os = $peakWS; peak_private_sampled = & $mx 'private'
    }
    $summary | ConvertTo-Json | Set-Content -Encoding ascii (Join-Path $dir 'mem.json')
    "[$Name-$v] mem: " + ($summary | ConvertTo-Json -Compress)
}
