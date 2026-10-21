# Runs gemma.json against `nanollm serve` on the Iris Xe (Vulkan), one server
# per variant, under preset uma-4g, and samples the GPU memory of the server
# process from the Windows performance counters while it loads and serves.
param(
    [string[]]$Variants = @('mtp3', 'mtp2', 'mtp1', 'nomtp'),
    # Appended to each variant's directory, so runs under another driver keep their own results.
    [string]$Tag = '',
    # The Iris Xe adapter's LUID; it changes when the driver is reinstalled.
    [string]$intelLuid = 'luid_0x00000000_0x000136ED'
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm.exe'
$input_json = 'C:\Users\rusun\Downloads\Telegram Desktop\gemma.json'
$port = 8090
$base = "http://127.0.0.1:$port"
$data = Join-Path $here 'data'
New-Item -ItemType Directory -Force $data | Out-Null

function Spec($v) {
    switch ($v) {
        'mtp3'  { "type: mtp`n        n_max: 3`n        p_min: 0.0" }
        'mtp2'  { "type: mtp`n        n_max: 2`n        p_min: 0.0" }
        'mtp1'  { "type: mtp`n        n_max: 1`n        p_min: 0.0" }
        'mtp3p5' { "type: mtp`n        n_max: 3`n        p_min: 0.5" }
        'nomtp' { "type: none" }
    }
}

function SharedOf($luid) {
    $a = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -Filter "Name LIKE '$luid%'"
    [int64]($a | Measure-Object SharedUsage -Sum).Sum
}

foreach ($v in $Variants) {
    $dir = Join-Path $here ($v + $Tag)
    New-Item -ItemType Directory -Force $dir | Out-Null
    $cfg = Join-Path $dir 'nanollm.yaml'
    $dataY = $data.Replace('\', '\\')
    @"
server:
  addr: "127.0.0.1:$port"
  data_dir: "$dataY"
auth:
  mode: open
backends:
  lib_dir: ./lib
  enabled: [llamacpp]
devices:
  default: llamacpp:vulkan0
  fallback: false
models:
  default: gemma-4-e4b-qat
  preload: [gemma-4-e4b-qat]
  overrides:
    gemma-4-e4b-qat:
      preset: uma-4g
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

    $baseline = SharedOf $intelLuid
    $env:GGML_VK_VISIBLE_DEVICES = '1'
    $p = Start-Process -FilePath $exe -ArgumentList @('serve', '-config', $cfg) -PassThru -WindowStyle Hidden `
        -RedirectStandardError (Join-Path $dir 'serve.log') -RedirectStandardOutput (Join-Path $dir 'serve.out')
    $stop = Join-Path $dir 'stop'
    Remove-Item -ErrorAction SilentlyContinue $stop
    $csv = Join-Path $dir 'gpumem.csv'
    $job = Start-Job -ArgumentList $p.Id, $intelLuid, $csv, $stop -ScriptBlock {
        param($procId, $luid, $csv, $stop)
        'time,proc_shared,proc_dedicated,proc_committed,adapter_shared,working_set,private' | Set-Content -Encoding ascii $csv
        while (-not (Test-Path $stop)) {
            $m = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -Filter "Name LIKE 'pid_$($procId)_$($luid)%'"
            $a = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -Filter "Name LIKE '$luid%'"
            $pr = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if (-not $pr) { break }
            $line = '{0},{1},{2},{3},{4},{5},{6}' -f (Get-Date).ToString('HH:mm:ss.fff'),
                [int64]($m | Measure-Object SharedUsage -Sum).Sum,
                [int64]($m | Measure-Object DedicatedUsage -Sum).Sum,
                [int64]($m | Measure-Object TotalCommitted -Sum).Sum,
                [int64]($a | Measure-Object SharedUsage -Sum).Sum,
                $pr.WorkingSet64, $pr.PrivateMemorySize64
            Add-Content -Encoding ascii $csv $line
            Start-Sleep -Milliseconds 250
        }
    }

    $ready = $false
    for ($i = 0; $i -lt 240; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "$base/readyz"
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
        if ($p.HasExited) { break }
    }
    "[$v] ready=$ready after $([math]::Round($i * 0.5, 1)) s"
    if ($ready) {
        & python -I (Join-Path $here 'run_steps.py') $base $input_json (Join-Path $dir 'answers.json')
        try {
            (Invoke-WebRequest -UseBasicParsing "$base/api/v1/models").Content | Set-Content -Encoding utf8 (Join-Path $dir 'models.json')
        } catch { "[$v] models: $_" }
    }
    New-Item -ItemType File $stop | Out-Null
    Wait-Job $job -Timeout 10 | Out-Null
    Remove-Job -Force $job
    $p.Refresh()
    $peakWS = $p.PeakWorkingSet64
    $peakPriv = $p.PeakPagedMemorySize64
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    $p.WaitForExit(15000) | Out-Null
    $rows = Import-Csv $csv
    $mx = { param($c) ($rows | ForEach-Object { [int64]$_.$c } | Measure-Object -Maximum).Maximum }
    $summary = [ordered]@{
        variant = $v; samples = $rows.Count
        adapter_shared_baseline = $baseline
        peak_proc_shared = & $mx 'proc_shared'; peak_proc_dedicated = & $mx 'proc_dedicated'
        peak_proc_committed = & $mx 'proc_committed'; peak_adapter_shared = & $mx 'adapter_shared'
        peak_working_set_sampled = & $mx 'working_set'; peak_private_sampled = & $mx 'private'
        peak_working_set_os = $peakWS; peak_paged_os = $peakPriv
    }
    $summary | ConvertTo-Json | Set-Content -Encoding ascii (Join-Path $dir 'mem.json')
    "[$v] mem: " + ($summary | ConvertTo-Json -Compress)
}
