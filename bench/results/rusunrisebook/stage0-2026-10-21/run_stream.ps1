# Cold first pass of the file-backed modes on Windows: the model is copied with unbuffered I/O (robocopy /J), so
# none of its pages is in the file cache, and the first prefill + 12 decode tokens of a fresh process read it from
# the disk. stream_ahead 0 is the old behaviour (page faults); the others use the Windows streamer.
param([string[]]$Presets = @('ram-1g', 'ram-2g'),
      [string[]]$Cases = @('0|off', '2|off', '4|off', '4|on', '0|off', '4|off'),
      [string]$Tag = 'stream')
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$raw = Join-Path $here 'raw'
$driver = Join-Path $here 'driver-stream.log'
function Log($m) { $l = '{0:HH:mm:ss} {1}' -f (Get-Date), $m; $l; try { Add-Content $driver $l } catch {} }
$exe = 'C:\Projects\nanollm\dist\windows_amd64\nanollm-dev.exe'
$src = 'C:\ProgramData\NanoLLM\models'
$cold = 'C:\Projects\runtimes\coldmodels'
$cfg = 'C:\Projects\runtimes\coldmodels.yaml'
@"
backends:
  lib_dir: "C:/Projects/nanollm/dist/windows_amd64/lib"
  enabled: [llamacpp]
devices:
  default: llamacpp:cpu
  fallback: false
registry:
  models_dir: "C:/Projects/runtimes/coldmodels"
ui:
  enabled: false
"@ | Set-Content -Encoding ascii $cfg
$env:GGML_VK_VISIBLE_DEVICES = '1'
foreach ($preset in $Presets) {
    foreach ($c in $Cases) {
        $ahead, $evict = $c -split '\|'
        $name = "$Tag-$preset-a$ahead-$evict"
        if (Test-Path $cold) { Remove-Item -Recurse -Force $cold }
        $t = Get-Date
        & robocopy $src $cold /E /J /NFL /NDL /NJH /NJS /NP | Out-Null
        Log ("copied cold in {0:N1} s" -f ((Get-Date) - $t).TotalSeconds)
        # the registry's verification marker names the files by absolute path: point it at the copy
        foreach ($vf in Get-ChildItem $cold -Recurse -Force -Filter '.verified') {
            (Get-Content -Raw $vf.FullName).Replace('C:\\ProgramData\\NanoLLM\\models', 'C:\\Projects\\runtimes\\coldmodels') | Set-Content -Encoding ascii -NoNewline $vf.FullName
        }
        Start-Sleep -Seconds 5
        $a = @('bench', '--config', $cfg, '--suite', 'dev', '--advisory', '--reps', '1', '--slow', '--preset', $preset, '--label', $name, '--out', "$raw\$name.json",
               '--set', "stream_ahead=$ahead", '--set', "stream_evict=$evict")
        Log "bench $name"
        $p = Start-Process -FilePath $exe -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput "$raw\$name.out" -RedirectStandardError "$raw\$name.log"
        if (-not $p.WaitForExit(1500000)) { $p.Kill(); Log "  TIMEOUT $name" }
        Get-Content "$raw\$name.log" | Where-Object { $_ -match '^\[\d+/\d+\].*(ok|fail|slow)|refused|error' } | ForEach-Object { Log "  $_" }
        Start-Sleep -Seconds 10
    }
}
if (Test-Path $cold) { Remove-Item -Recurse -Force $cold }
Log 'ALL DONE'
