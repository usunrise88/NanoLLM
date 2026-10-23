<#
.SYNOPSIS
  The checks of docs/OPTIMIZE.md section 7 that need administrator rights.

.DESCRIPTION
  Run from an elevated PowerShell (Run as administrator):

    .\scripts\check-optimize.ps1 -Exe .\nanollm.exe -Config .\nanollm.yaml
    .\scripts\check-optimize.ps1 -Exe .\nanollm.exe -Config .\nanollm.yaml -Archive .\nanollm-dev-windows-amd64.zip

  1. The measurement as LocalSystem. A one-shot scheduled task runs
     `nanollm optimize --mode MODE --json` as SYSTEM, the account the service
     runs as. The result says whether the process's video memory could be read
     there (PDH counters, memory.source "pdh"), what the OS session hides from a
     service (an iGPU, session 0), and what the mode chose. The task is removed
     afterwards.
  2. With -Archive: the real install.ps1 into a scratch prefix, data directory
     and service name (NanoLLMOptCheck, 127.0.0.1:18092), with the model copied
     from -ModelsDir so nothing is downloaded. The installer measures the
     machine through the running service; the script then reads the profile it
     wrote, asks `optimize --status`, and uninstalls everything it installed.

  Everything is written to -Out (check-optimize\ next to the script's working
  directory): send that folder back. Nothing is changed outside it, the scratch
  directories and the task and service named above.

  -AsCurrentUser runs part 1 as the current user without administrator rights:
  it tests the script's own plumbing, not LocalSystem.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Exe,
    [Parameter(Mandatory)] [string] $Config,
    [string] $Archive,
    [string] $Mode = 'optimal',
    [string] $ModelsDir = (Join-Path $env:ProgramData 'NanoLLM\models'),
    [string] $Out = (Join-Path (Get-Location) 'check-optimize'),
    [int] $TimeoutSeconds = 600,
    [switch] $AsCurrentUser
)

$ErrorActionPreference = 'Stop'
$TaskName = 'NanoLLM-optimize-check'
$Service = 'NanoLLMOptCheck'
$Addr = '127.0.0.1:18092'
$failed = @()

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Pass($m) { Write-Host "    PASS  $m" -ForegroundColor Green }
function Fail($m) { Write-Host "    FAIL  $m" -ForegroundColor Red; $script:failed += $m }
function Info($m) { Write-Host "    $m" }

function Test-Admin {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not $AsCurrentUser -and -not (Test-Admin)) {
    throw 'Run this from an elevated PowerShell (Run as administrator), or add -AsCurrentUser to test the script itself.'
}
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$Config = (Resolve-Path -LiteralPath $Config).Path
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path -LiteralPath $Out).Path

# Show-Profile prints what matters of an `optimize --json` result and returns it.
function Show-Result($path, $who) {
    $text = Get-Content -LiteralPath $path -Raw -ErrorAction Stop
    $r = $text | ConvertFrom-Json
    $p = if ($r.PSObject.Properties.Name -contains 'profile') { $r.profile } else { $r }
    Info "mode $($p.mode), took $([math]::Round($p.duration_s, 1)) s"
    Info "chosen $($p.chosen.device.value), $($p.chosen.plan_mode.value), context $($p.chosen.n_ctx), draft $($p.chosen.mtp.value | ConvertTo-Json -Compress)"
    Info "memory plan $($p.memory.plan_bytes) private $($p.memory.private_bytes) vram $($p.memory.vram_bytes) measured $($p.memory.measured_bytes) source '$($p.memory.source)'"
    $codes = @($p.problems | ForEach-Object { $_.code })
    Info "problems: $($codes -join ', ')"
    $p
}

# ---- 1. the measurement as LocalSystem ------------------------------------------------

Say "the $Mode measurement as $(if ($AsCurrentUser) { 'the current user (plumbing test)' } else { 'LocalSystem' })"
$wrapper = Join-Path $Out 'run-as-task.cmd'
$json = Join-Path $Out 'system.json'
$err = Join-Path $Out 'system.err'
Remove-Item -LiteralPath $json, $err -ErrorAction SilentlyContinue
Set-Content -LiteralPath $wrapper -Encoding ASCII -Value @(
    '@echo off',
    ('"{0}" optimize -config "{1}" --mode {2} --json > "{3}" 2> "{4}"' -f $Exe, $Config, $Mode, $json, $err),
    'exit /b %errorlevel%'
)
try {
    $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument ('/c "{0}"' -f $wrapper)
    $principal = if ($AsCurrentUser) {
        New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    } else {
        New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    }
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Seconds $TimeoutSeconds) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Register-ScheduledTask -TaskName $TaskName -Action $action -Principal $principal -Settings $settings | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        Start-Sleep -Seconds 2
        $state = (Get-ScheduledTask -TaskName $TaskName).State
    } while ($state -ne 'Ready' -and (Get-Date) -lt $deadline)
    $last = (Get-ScheduledTaskInfo -TaskName $TaskName).LastTaskResult
    if ($state -ne 'Ready') { Fail "the task did not finish in $TimeoutSeconds s"; Stop-ScheduledTask -TaskName $TaskName }
    elseif ($last -ne 0) { Fail "the task exited with $last (see $err)" }
    else { Pass "the task finished" }
} finally {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
if (Test-Path -LiteralPath $json) {
    try {
        $p = Show-Result $json 'task'
        if ($AsCurrentUser) { Info 'plumbing only: the PDH verdict below is about the current user' }
        if ($p.memory.source -eq 'pdh') { Pass 'the video memory of the process was read from PDH' }
        elseif ($p.chosen.device.value -match 'cpu') { Info "the CPU was chosen, so there is no video memory to read (memory source '$($p.memory.source)')" }
        else { Fail "the video memory came from '$($p.memory.source)', not PDH: the counters are not readable here" }
        if ($p.memory.measured_bytes -gt 0) { Pass "measured $([math]::Round($p.memory.measured_bytes / 1GB, 2)) GiB" } else { Fail 'nothing was measured' }
    } catch { Fail "the result cannot be read: $($_.Exception.Message)" }
} else { Fail "no result in $json (see $err)" }

# ---- 2. the real installer ------------------------------------------------------------

if ($Archive) {
    Say 'the installer: install.ps1 -Archive into a scratch prefix, with the measurement through the service'
    $installer = Join-Path $PSScriptRoot 'install.ps1'
    $scratch = Join-Path ([IO.Path]::GetTempPath()) 'nanollm-optcheck'
    $prefix = Join-Path $scratch 'prefix'
    $data = Join-Path $scratch 'data'
    $Archive = (Resolve-Path -LiteralPath $Archive).Path
    try {
        Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force -Path $prefix, $data | Out-Null
        if (Test-Path -LiteralPath $ModelsDir) {
            Info "copying the model from $ModelsDir (no download)"
            Copy-Item -LiteralPath $ModelsDir -Destination (Join-Path $data 'models') -Recurse -Force
        } else { Info "no ${ModelsDir}: the installer will download the model" }
        $log = Join-Path $Out 'install.log'
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer -Archive $Archive -Prefix $prefix -DataDir $data `
            -ServiceName $Service -Addr $Addr -NoPath -NoCuda -Mode $Mode -NonInteractive *>&1 | Tee-Object -FilePath $log | ForEach-Object { Info "$_" }
        if ($LASTEXITCODE -ne 0) { Fail "install.ps1 exited with $LASTEXITCODE" } else { Pass 'install.ps1 finished' }

        $svc = Get-Service -Name $Service -ErrorAction SilentlyContinue
        if ($svc -and $svc.Status -eq 'Running') { Pass "service $Service is running" } else { Fail "service $Service is not running" }
        $account = (Get-CimInstance Win32_Service -Filter "Name='$Service'" -ErrorAction SilentlyContinue).StartName
        Info "the service runs as: $account"
        try {
            $health = Invoke-RestMethod -Uri "http://$Addr/healthz" -TimeoutSec 10
            if ($health.status -eq 'ok') { Pass '/healthz answers ok' } else { Fail "/healthz: $($health | ConvertTo-Json -Compress)" }
        } catch { Fail "/healthz: $($_.Exception.Message)" }

        $status = & $(Join-Path $prefix 'nanollm.exe') optimize --status --json -config (Join-Path $data 'nanollm.yaml') 2>&1
        Set-Content -LiteralPath (Join-Path $Out 'install-status.json') -Value $status
        $st = ($status -join "`n") | ConvertFrom-Json
        if ($st.mode -and $st.profile -and -not $st.stale) { Pass "the profile $($st.profile.id) is current for the $($st.mode) mode" }
        else { Fail "optimize --status: mode '$($st.mode)', stale $($st.stale), missing $($st.missing)" }

        $profile = Get-ChildItem -LiteralPath (Join-Path $data 'tune') -Filter '*.json' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($profile) {
            Copy-Item -LiteralPath $profile.FullName -Destination (Join-Path $Out 'install-profile.json')
            $p = Show-Result $profile.FullName 'service'
            if ($p.memory.source -eq 'pdh') { Pass 'inside the service the video memory was read from PDH' }
            elseif ($p.chosen.device.value -match 'cpu') { Info 'the CPU was chosen: no video memory to read' }
            else { Fail "inside the service the video memory came from '$($p.memory.source)'" }
        } else { Fail 'the installer wrote no profile' }
    } catch {
        Fail "the installer check stopped: $($_.Exception.Message)"
    } finally {
        Say 'removing what the check installed'
        try {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer -Uninstall -Purge -Prefix $prefix -DataDir $data -ServiceName $Service -NonInteractive *>&1 |
                Tee-Object -FilePath (Join-Path $Out 'uninstall.log') | Out-Null
        } catch { Info "uninstall: $($_.Exception.Message)" }
        if (Get-Service -Name $Service -ErrorAction SilentlyContinue) {
            Fail "service $Service is still registered: sc.exe delete $Service"
        }
        Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# ---- the verdict ----------------------------------------------------------------------

Say 'result'
if ($failed.Count -eq 0) { Write-Host '    all checks passed' -ForegroundColor Green }
else { $failed | ForEach-Object { Write-Host "    FAIL  $_" -ForegroundColor Red } }
Info "send the folder $Out back"
exit $(if ($failed.Count -eq 0) { 0 } else { 1 })
