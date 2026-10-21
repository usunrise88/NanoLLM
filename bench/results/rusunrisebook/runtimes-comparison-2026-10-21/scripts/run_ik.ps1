# Runs gemma.json against llama-server of ik_llama.cpp (CPU only), pinned to
# the six physical P-cores, one server per variant, and samples the memory of
# the server process.
param(
    [string[]]$Variants = @('nomtp', 'mtp2', 'mtp3'),
    [string]$Tag = '',
    [int]$Mask = 0x555,
    [int]$Threads = 6,
    [string]$Extra = ''
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$env:PATH = 'C:\msys64\mingw64\bin;' + $env:PATH
$exe = 'C:\Projects\runtimes\ik_llama.cpp\build\bin\llama-server.exe'
$md = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265'
$model = Join-Path $md 'gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$draft = Join-Path $md 'mtp-gemma-4-E4B-it.gguf'
$input_json = 'C:\Users\rusun\Downloads\Telegram Desktop\gemma.json'
$port = 8091
$base = "http://127.0.0.1:$port"

foreach ($v in $Variants) {
    $dir = "C:\Projects\runtimes\results\ik-$v$Tag"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $a = @('-m', $model, '-t', "$Threads", '-tb', "$Threads", '-c', '8192', '-np', '1', '--jinja',
        '--chat-template-kwargs', '{\"enable_thinking\":false}', '--host', '127.0.0.1', '--port', "$port")
    switch ($v) {
        'mtp2' { $a += @('--spec-type', 'mtp:n_max=2,p_min=0.0', '-md', $draft) }
        'mtp3' { $a += @('--spec-type', 'mtp:n_max=3,p_min=0.0', '-md', $draft) }
    }
    if ($Extra) { $a += ($Extra -split ' ' | Where-Object { $_ }) }
    ($a -join ' ') | Set-Content -Encoding ascii (Join-Path $dir 'cmdline.txt')
    $p = Start-Process -FilePath $exe -ArgumentList $a -PassThru -WindowStyle Hidden `
        -RedirectStandardError (Join-Path $dir 'serve.log') -RedirectStandardOutput (Join-Path $dir 'serve.out')
    try { $p.ProcessorAffinity = [IntPtr]$Mask } catch {}
    $csv = Join-Path $dir 'mem.csv'
    'time,working_set,private' | Set-Content -Encoding ascii $csv
    $stop = Join-Path $dir 'stop'
    Remove-Item -ErrorAction SilentlyContinue $stop
    $job = Start-Job -ArgumentList $p.Id, $csv, $stop -ScriptBlock {
        param($procId, $csv, $stop)
        while (-not (Test-Path $stop)) {
            $pr = Get-Process -Id $procId -ErrorAction SilentlyContinue
            if (-not $pr) { break }
            Add-Content -Encoding ascii $csv ('{0},{1},{2}' -f (Get-Date).ToString('HH:mm:ss.fff'), $pr.WorkingSet64, $pr.PrivateMemorySize64)
            Start-Sleep -Milliseconds 250
        }
    }
    $ready = $false
    for ($i = 0; $i -lt 240; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "$base/health"
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch {}
        if ($p.HasExited) { break }
    }
    "[ik-$v] ready=$ready after $([math]::Round($i * 0.5, 1)) s"
    if ($ready) {
        & python -I (Join-Path $here 'run_steps_stream.py') $base $input_json (Join-Path $dir 'answers.json') 'gemma-4-e4b'
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
    $sum = [ordered]@{ variant = "ik-$v"; peak_working_set_os = $peakWS; peak_working_set_sampled = & $mx 'working_set'; peak_private_sampled = & $mx 'private' }
    $sum | ConvertTo-Json | Set-Content -Encoding ascii (Join-Path $dir 'mem.json')
    "[ik-$v] mem: " + ($sum | ConvertTo-Json -Compress)
}
