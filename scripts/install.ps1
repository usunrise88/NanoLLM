#Requires -Version 5.1
#
# Install NanoLLM on Windows as a service.
#
#   irm https://github.com/usunrise88/NanoLLM/releases/latest/download/install.ps1 | iex
#
# The one-liner is the whole interface: it works out the latest release,
# checks that this machine can run it, unpacks the archive into
# C:\Program Files\NanoLLM, writes a configuration with two API keys under
# C:\ProgramData\NanoLLM (SPEC section 10), downloads the default model and hands the
# registration to the binary itself: `nanollm.exe service install` creates
# the service, so the two can never disagree about how it is registered. Run
# it again to upgrade in place; the configuration, the keys and the weights
# are kept.
#
# A piped script cannot take parameters, so every option is also an
# environment variable:
#
#   $env:NANOLLM_ADDR = "0.0.0.0:8080"; $env:NANOLLM_CUDA = "1"
#   irm https://github.com/usunrise88/NanoLLM/releases/latest/download/install.ps1 | iex
#
# With the script on disk the parameters work as parameters:
#
#   .\install.ps1 -Addr 0.0.0.0:8080 -Cuda
#   .\install.ps1 -Archive .\nanollm-v1.0.0-windows-amd64.zip -NoDownload
#   .\install.ps1 -Uninstall -Purge
#
# It elevates itself: Program Files, the service control manager, the
# machine PATH and the firewall all need administrative rights.

[CmdletBinding()]
param(
    [string] $Version     = $env:NANOLLM_VERSION,
    [string] $Prefix      = $(if ($env:NANOLLM_PREFIX)   { $env:NANOLLM_PREFIX }   else { Join-Path $env:ProgramFiles 'NanoLLM' }),
    [string] $DataDir     = $(if ($env:NANOLLM_DATA_DIR) { $env:NANOLLM_DATA_DIR } else { Join-Path $env:ProgramData 'NanoLLM' }),
    [string] $Addr        = $(if ($env:NANOLLM_ADDR)     { $env:NANOLLM_ADDR }     else { '127.0.0.1:8080' }),
    [string] $ServiceName = $(if ($env:NANOLLM_SERVICE)  { $env:NANOLLM_SERVICE }  else { 'NanoLLM' }),
    [string] $Repo        = $(if ($env:NANOLLM_REPO)     { $env:NANOLLM_REPO }     else { 'usunrise88/NanoLLM' }),
    [string] $Archive     = $env:NANOLLM_ARCHIVE,
    [string] $CudaArchive = $env:NANOLLM_CUDA_ARCHIVE,
    [switch] $NoUI        = [bool]($env:NANOLLM_UI       -eq '0'),
    [switch] $NoDownload  = [bool]($env:NANOLLM_DOWNLOAD -eq '0'),
    [switch] $Cuda        = [bool]($env:NANOLLM_CUDA     -eq '1'),
    [switch] $NoCuda      = [bool]($env:NANOLLM_CUDA     -eq '0'),
    [switch] $NoStart     = [bool]($env:NANOLLM_START    -eq '0'),
    [switch] $NoPath      = [bool]($env:NANOLLM_PATH     -eq '0'),
    [switch] $Uninstall,
    [switch] $Purge,
    [switch] $Help
)

$ErrorActionPreference = 'Stop'
# Invoke-WebRequest spends most of a download drawing the progress bar.
$ProgressPreference = 'SilentlyContinue'
# Windows PowerShell 5.1 still defaults to TLS 1.0 on older builds, which
# GitHub closes the connection on.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$SelfUrl = "https://github.com/$Repo/releases/latest/download/install.ps1"
$Config  = Join-Path $DataDir 'nanollm.yaml'
$Exe     = Join-Path $Prefix 'nanollm.exe'
$LogFile = Join-Path $DataDir 'logs\nanollm.log'
# Whether an address was asked for at all: an upgrade keeps the configuration,
# so one that was asked for and not applied has to be reported.
$AddrGiven = $PSBoundParameters.ContainsKey('Addr') -or [bool]$env:NANOLLM_ADDR
if ($Archive)     { $Archive     = [IO.Path]::GetFullPath($Archive) }
if ($CudaArchive) { $CudaArchive = [IO.Path]::GetFullPath($CudaArchive); $Cuda = [switch]$true }

function Say  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "    warning: $m" -ForegroundColor Yellow }
function Note ($m) { Write-Host $m }

function Show-Usage {
    @"
Install NanoLLM as a Windows service.

  -Version TAG        release to install                     (default: the latest)
  -Prefix DIR         binary and libraries                   ($Prefix)
  -DataDir DIR        configuration, weights, job database   ($DataDir)
  -Addr HOST:PORT     listen address                         ($Addr)
  -ServiceName NAME   name in the service control manager    ($ServiceName)
  -NoUI               install the build without the web interface
  -NoDownload         write the configuration but fetch no model
  -Cuda               x64: add the CUDA libraries (needs an NVIDIA GPU and driver)
  -NoCuda             remove them again
  -NoStart            register the service but leave it stopped
  -NoPath             do not add the installation to the machine PATH
  -Archive PATH       install this archive instead of downloading one (tests, offline)
  -CudaArchive PATH   the CUDA add-on to go with -Archive
  -Uninstall          stop and remove the service and the installation
  -Purge              with -Uninstall: also delete the configuration, the keys and the weights

Each has an environment variable, for the piped form where parameters cannot
be passed: NANOLLM_VERSION, NANOLLM_PREFIX, NANOLLM_DATA_DIR, NANOLLM_ADDR,
NANOLLM_SERVICE, NANOLLM_UI=0, NANOLLM_DOWNLOAD=0, NANOLLM_CUDA=1 (or 0),
NANOLLM_START=0, NANOLLM_PATH=0, NANOLLM_ARCHIVE, NANOLLM_CUDA_ARCHIVE,
NANOLLM_UNINSTALL=1, NANOLLM_PURGE=1.
"@
}

function Test-Administrator {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Invoke-Elevated reopens this script in a window that has administrative
# rights. Every option is passed explicitly: an elevated process does not
# inherit the environment of the window that asked for it, so the NANOLLM_*
# variables of a piped run would be lost. A piped script has no file to
# reopen, so it is fetched first.
function Invoke-Elevated {
    $script = $PSCommandPath
    if (-not $script -or -not (Test-Path -LiteralPath $script)) {
        $script = Join-Path $env:TEMP 'nanollm-install.ps1'
        Say "downloading the installer so it can be reopened as administrator"
        Invoke-WebRequest -UseBasicParsing -Uri $SelfUrl -OutFile $script
    }
    $a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-NoExit', '-File', "`"$script`"",
           '-Prefix', "`"$Prefix`"", '-DataDir', "`"$DataDir`"",
           '-ServiceName', "`"$ServiceName`"", '-Repo', "`"$Repo`"")
    if ($AddrGiven)   { $a += @('-Addr', "`"$Addr`"") }
    if ($Version)     { $a += @('-Version', "`"$Version`"") }
    if ($Archive)     { $a += @('-Archive', "`"$Archive`"") }
    if ($CudaArchive) { $a += @('-CudaArchive', "`"$CudaArchive`"") }
    foreach ($s in 'NoUI', 'NoDownload', 'Cuda', 'NoCuda', 'NoStart', 'NoPath', 'Uninstall', 'Purge') {
        if ((Get-Variable -Name $s -ValueOnly)) { $a += "-$s" }
    }
    Say "asking for administrative rights"
    try {
        Start-Process -FilePath (Get-Process -Id $PID).Path -Verb RunAs -ArgumentList $a
    } catch {
        throw "administrative rights are required, and the request was refused. " +
              "Open PowerShell with `"Run as administrator`" and run this again."
    }
    # -NoExit keeps that window open: the API keys are printed there.
    Note "Continuing in the elevated window; the API keys are printed there."
}

# Invoke-Native runs the binary, shows what it printed and returns a copy.
# The error preference is relaxed for the call: "Stop" turns a native
# program's stderr into a terminating error, and the exit code is what says
# whether it worked.
function Invoke-Native {
    param([string] $File, [string[]] $Arguments, [switch] $Quiet)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $File @Arguments 2>&1 | ForEach-Object { $line = "$_"; if (-not $Quiet) { Write-Host $line }; $line }
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) {
        if ($Quiet) { $out | ForEach-Object { Write-Host $_ } }
        throw "$(Split-Path -Leaf $File) $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
    return $out
}

function Test-ConfigHasKeys {
    if (-not (Test-Path -LiteralPath $Config) -or -not (Test-Path -LiteralPath $Exe)) { return $false }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try   { $out = (& $Exe key list -config $Config 2>&1 | Out-String) }
    finally { $ErrorActionPreference = $previous }
    return ($LASTEXITCODE -eq 0 -and $out -notmatch 'no keys in')
}

# Get-LatestTag asks GitHub which release is current: the API first, and the
# redirect /releases/latest performs when the API has had enough of this
# address today.
function Get-LatestTag {
    try {
        return (Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$Repo/releases/latest").tag_name
    } catch {
        try {
            $req = [Net.WebRequest]::Create("https://github.com/$Repo/releases/latest")
            $req.AllowAutoRedirect = $false
            $req.Method = 'HEAD'
            $location = $req.GetResponse().Headers['Location']
            if ($location) { return $location.Split('/')[-1] }
        } catch { }
    }
    throw "cannot work out the latest release; pass -Version TAG"
}

# Test-Checksum checks File in Dir against the line for it in Sums.
function Test-Checksum {
    param([string] $Dir, [string] $File, [string] $Sums)
    $pattern = "\s\*?$([regex]::Escape($File))$"
    $line = Get-Content -LiteralPath $Sums | Where-Object { $_ -match $pattern } | Select-Object -First 1
    if (-not $line) { throw "$(Split-Path -Leaf $Sums) does not mention $File" }
    $want = ($line -split '\s+')[0]
    $got  = (Get-FileHash -LiteralPath (Join-Path $Dir $File) -Algorithm SHA256).Hash
    if ($want -ine $got) { throw "$File failed its checksum: expected $want, got $got" }
    Say "checksum ok: $File"
}

# Get-Archive returns the path of the verified archive Name: the local file
# when one was given (checked against a sha256sums.txt beside it, if any),
# or a download into Tmp checked against the release's sha256sums.txt.
function Get-Archive {
    param([string] $Name, [string] $Local, [string] $Tmp, [string] $Base)
    if ($Local) {
        if (-not (Test-Path -LiteralPath $Local)) { throw "no such archive: $Local" }
        $sums = Join-Path (Split-Path -Parent $Local) 'sha256sums.txt'
        if (Test-Path -LiteralPath $sums) {
            Test-Checksum -Dir (Split-Path -Parent $Local) -File (Split-Path -Leaf $Local) -Sums $sums
        } else {
            Say "using $Local (no sha256sums.txt beside it to check against)"
        }
        return $Local
    }
    Say "downloading $Name"
    $path = Join-Path $Tmp $Name
    try { Invoke-WebRequest -UseBasicParsing -Uri "$Base/$Name" -OutFile $path }
    catch { throw "no such release asset: $Base/$Name" }
    $sums = Join-Path $Tmp 'sha256sums.txt'
    if (-not (Test-Path -LiteralPath $sums)) {
        try { Invoke-WebRequest -UseBasicParsing -Uri "$Base/sha256sums.txt" -OutFile $sums }
        catch { throw "$Version publishes no sha256sums.txt, so $Name cannot be verified" }
    }
    Test-Checksum -Dir $Tmp -File $Name -Sums $sums
    return $path
}

function Get-ProbeUrl {
    $parts = $Addr -split ':'
    $port  = $parts[-1]
    $host_ = ($parts[0..($parts.Length - 2)] -join ':')
    if ($host_ -in @('', '0.0.0.0', '*', '::', '[::]')) { $host_ = '127.0.0.1' }
    return "http://${host_}:${port}"
}

function Test-Loopback {
    $host_ = ($Addr -split ':')[0]
    return ($host_ -in @('127.0.0.1', 'localhost', '[::1]', '::1'))
}

# Test-Dependencies names what the llama.cpp DLLs need from the system and
# the archive cannot carry: the Visual C++ runtime, which every ggml and
# llama DLL imports, and the Vulkan loader, which comes with a GPU driver.
# Nothing is installed behind the operator's back; it says what to run.
$script:DepsMissing = @()
function Test-Dependencies {
    param([string] $Arch)
    $sys = Join-Path $env:SystemRoot 'System32'
    $vc = @('VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'MSVCP140.dll') |
        Where-Object { -not (Test-Path -LiteralPath (Join-Path $sys $_)) }
    if ($vc) {
        $script:DepsMissing += $vc
        $redist = if ($Arch -eq 'arm64') { 'arm64' } else { 'x64' }
        Warn ("the Visual C++ runtime is not installed ($($vc -join ', ') missing from $sys).`n" +
              "    Every llama.cpp library imports it, so until it is there the server can load no model.`n" +
              "    Install it with:  winget install Microsoft.VCRedist.2015+.$redist`n" +
              "    or from https://aka.ms/vs/17/release/vc_redist.$redist.exe")
    }
    if (-not (Test-Path -LiteralPath (Join-Path $sys 'vulkan-1.dll'))) {
        $script:DepsMissing += 'vulkan-1.dll'
        Warn ("vulkan-1.dll (the Vulkan loader) is not installed, so the Vulkan backend is skipped`n" +
              "    and models run on the CPU. It comes with the GPU driver: install or update the`n" +
              "    driver from AMD, Intel or NVIDIA.")
    }
}

function Test-Nvidia {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return $true }
    try {
        return [bool](Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match 'NVIDIA' })
    } catch { return $false }
}

# 'cuda13' when every NVIDIA GPU is Turing or newer (compute capability 7.5+, which is what the CUDA 13.4 build
# targets) and its driver is 580 or newer (the CUDA 13 runtime's minimum); 'cuda' (CUDA 12.4: Maxwell and newer, driver
# 525+) otherwise, or when nvidia-smi cannot say.
function Get-CudaFlavour {
    try {
        $rows = @(nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2>$null)
        foreach ($r in $rows) {
            $p = "$r" -split ',\s*'
            if ($p.Count -lt 2) { return 'cuda' }
            if ([int](($p[0] -split '\.')[0]) -lt 580 -or [double]::Parse($p[1], [cultureinfo]::InvariantCulture) -lt 7.5) { return 'cuda' }
        }
        if ($rows.Count -gt 0) { return 'cuda13' }
    } catch { }
    return 'cuda'
}

# Whether the release has this asset (a HEAD request).
function Test-ReleaseAsset {
    param([string] $Name, [string] $Base)
    try { return (Invoke-WebRequest -UseBasicParsing -Method Head -Uri "$Base/$Name" -ErrorAction Stop).StatusCode -eq 200 }
    catch { return $false }
}

function Test-CudaLibs { return [bool](Get-ChildItem -Path (Join-Path $Prefix 'lib') -Filter '*ggml-cuda*' -ErrorAction SilentlyContinue) }

function Add-ToMachinePath {
    if ($NoPath) { return }
    $current = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    if (-not $current) { $current = '' }
    if (($current -split ';') -contains $Prefix) { return }
    Say "adding $Prefix to the machine PATH"
    [Environment]::SetEnvironmentVariable('Path', ($current.TrimEnd(';') + ";$Prefix"), 'Machine')
    $env:Path = "$env:Path;$Prefix"
}

function Remove-FromMachinePath {
    $current = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    if (-not $current) { return }
    $kept = ($current -split ';') | Where-Object { $_ -and $_ -ne $Prefix }
    [Environment]::SetEnvironmentVariable('Path', ($kept -join ';'), 'Machine')
}

# Open-Firewall is reached only for an address that is not the loopback,
# which is a deliberate choice by whoever passed it; it is announced.
function Open-Firewall {
    $port = ($Addr -split ':')[-1]
    $rule = "NanoLLM ($port/tcp)"
    try {
        if (Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue) { return }
        Say "opening TCP $port in the firewall, because $Addr is reachable from outside this machine"
        New-NetFirewallRule -DisplayName $rule -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $port -Profile Any | Out-Null
    } catch {
        Warn "the firewall rule was not created: $($_.Exception.Message)"
    }
}

function Close-Firewall {
    try {
        Get-NetFirewallRule -DisplayName 'NanoLLM (*' -ErrorAction SilentlyContinue |
            Remove-NetFirewallRule -ErrorAction SilentlyContinue
    } catch { }
}

function Wait-ForHealth {
    $url = Get-ProbeUrl
    for ($i = 0; $i -lt 60; $i++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "$url/healthz" -TimeoutSec 2 | Out-Null
            Say "$ServiceName answers on $url"
            return $true
        } catch { }
        $svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        if ($svc -and $svc.Status -eq 'Stopped') {
            Warn "the service stopped; the reason is in $LogFile"
            return $false
        }
        Start-Sleep -Seconds 1
    }
    Warn "no answer from $url/healthz after a minute; see $LogFile"
    return $false
}

function Install-NanoLLM {
    # PROCESSOR_ARCHITECTURE describes the PowerShell host, not the machine: a
    # 32-bit host on 64-bit Windows says x86 and puts the truth in the other.
    $machine = $env:PROCESSOR_ARCHITEW6432
    if (-not $machine) { $machine = $env:PROCESSOR_ARCHITECTURE }
    switch ($machine) {
        'AMD64' { $arch = 'amd64' }
        'ARM64' { $arch = 'arm64' }
        default { throw "there is no release for $machine" }
    }
    $target = "windows-$arch"

    if ($Archive) {
        # The tag is in the name: nanollm-<tag>-<target>[-noui].zip.
        $name = Split-Path -Leaf $Archive
        if (-not $Version) {
            $script:Version = ($name -replace '^nanollm-', '') -replace "-$target.*$", ''
        }
        $base = ''
    } else {
        if (-not $Version) {
            Say "asking GitHub for the latest release"
            $script:Version = Get-LatestTag
        }
        $suffix = if ($NoUI) { '-noui' } else { '' }
        $name   = "nanollm-$Version-$target$suffix.zip"
        $base   = "https://github.com/$Repo/releases/download/$Version"
    }

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("nanollm-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $zip = Get-Archive -Name $name -Local $Archive -Tmp $tmp -Base $base

        # Stopped before anything is overwritten: Windows refuses to replace
        # the exe and the DLLs of a running process, so an upgrade that
        # skipped this would fail halfway and leave a mixture behind.
        $existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        if ($existing -and $existing.Status -ne 'Stopped') {
            Say "stopping the running $ServiceName service"
            Stop-Service -Name $ServiceName -Force
            $existing.WaitForStatus('Stopped', [TimeSpan]::FromSeconds(90))
        }

        New-Item -ItemType Directory -Path $Prefix, $DataDir -Force | Out-Null
        # lib\ is replaced, not merged: the new binary checks the libraries'
        # commit, and a CUDA backend left from the last release would not match.
        $hadCuda = Test-CudaLibs
        $lib = Join-Path $Prefix 'lib'
        if (Test-Path -LiteralPath $lib) { Remove-Item -LiteralPath $lib -Recurse -Force }
        Say "unpacking $(Split-Path -Leaf $zip) into $Prefix"
        Expand-Archive -LiteralPath $zip -DestinationPath $Prefix -Force
        if (-not (Test-Path -LiteralPath $Exe)) { throw "$(Split-Path -Leaf $zip) did not contain nanollm.exe" }

        $wantCuda = $Cuda -or ($hadCuda -and -not $NoCuda)
        if ($wantCuda) {
            if ($arch -ne 'amd64') {
                Warn "the CUDA add-on is for x64 only; skipped"
            } elseif (-not $CudaArchive -and -not (Test-Nvidia)) {
                Warn "-Cuda: no NVIDIA GPU found; the add-on is not installed"
            } else {
                if ($hadCuda -and -not $Cuda) { Say "refreshing the CUDA add-on to match the new binary" }
                # CUDA 13.4 (Turing and newer, driver 580+) when the driver can run it and the release has the archive;
                # CUDA 12.4 otherwise (Maxwell and newer, driver 525+)
                $cudaName = 'cuda'
                if (-not $CudaArchive -and (Get-CudaFlavour) -eq 'cuda13' -and (Test-ReleaseAsset -Name "nanollm-$Version-windows-amd64-cuda13.zip" -Base $base)) { $cudaName = 'cuda13' }
                $addon = Get-Archive -Name "nanollm-$Version-windows-amd64-$cudaName.zip" -Local $CudaArchive -Tmp $tmp -Base $base
                Say "unpacking the CUDA add-on into $lib"
                Expand-Archive -LiteralPath $addon -DestinationPath $Prefix -Force
            }
        } elseif ($hadCuda) {
            Say "the CUDA add-on was removed (-NoCuda)"
        }
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }

    Test-Dependencies -Arch $arch
    Add-ToMachinePath

    $keys = @()
    if (Test-ConfigHasKeys) {
        Say "keeping the configuration in $Config"
        # The server listens where the kept file says, not where -Addr says,
        # and the health check and the summary have to ask the same place.
        $line = Get-Content -LiteralPath $Config | Where-Object { $_ -like '  addr:*' } | Select-Object -First 1
        if ($line -match '^  addr:\s*"?([^"#\s]+)' -and $Matches[1] -ne $Addr) {
            if ($AddrGiven) {
                Warn "-Addr was not applied: an upgrade keeps the configuration, and the server listens on $($Matches[1]). Set server.addr in $Config and run: nanollm service restart"
            }
            $script:Addr = $Matches[1]
        }
    } else {
        Say "writing the configuration with two fresh API keys"
        $out  = Invoke-Native $Exe @('init', '-config', $Config, '-data-dir', $DataDir, '-addr', $Addr, '-force', '-no-download') -Quiet
        $keys = @($out | Where-Object { $_ -match '^\s*(admin|user) key' })
    }
    if (-not $NoDownload) {
        Say "fetching the default model (gigabytes; -NoDownload skips it)"
        try { Invoke-Native $Exe @('models', 'pull', '-config', $Config) | Out-Null }
        catch { Warn "the model was not fetched; the server starts without it: nanollm models pull -config `"$Config`"" }
    }

    Say "registering the $ServiceName service"
    $svcArgs = @('service', 'install', '-name', $ServiceName, '-config', $Config, '-log-file', $LogFile)
    if ($NoStart) { $svcArgs += '-no-start' }
    Invoke-Native $Exe $svcArgs | Out-Null

    if (-not (Test-Loopback)) { Open-Firewall }
    $healthy = $true
    if (-not $NoStart) { $healthy = Wait-ForHealth }

    $url = Get-ProbeUrl
    Note ""
    Note "NanoLLM $Version is installed."
    Note ""
    Note "  service   $ServiceName          nanollm service status"
    Note "  binary    $Exe"
    Note "  config    $Config"
    Note "  data      $DataDir"
    Note "  log       $LogFile"
    Note "  cuda      $(if (Test-CudaLibs) { "installed in $Prefix\lib" } else { 'not installed (-Cuda adds it on an NVIDIA machine)' })"
    Note "  url       $url    ($url/ui/ in a browser)"
    Note ""
    if ($keys.Count) {
        $keys | ForEach-Object { Note "  $($_.Trim())" }
    } else {
        Note "  The API keys are in $Config (nanollm key list -config `"$Config`")."
    }
    if ($script:DepsMissing.Count) {
        Note ""
        Note "  Missing: $($script:DepsMissing -join ', '); see the warning above."
    }
    Note ""
    Note "  restart    nanollm --restart        (or Restart-Service $ServiceName)"
    Note "  status     nanollm --status"
    Note "  update     irm https://github.com/$Repo/releases/latest/download/install.ps1 | iex"
    Note "  uninstall  `$env:NANOLLM_UNINSTALL=1; irm https://github.com/$Repo/releases/latest/download/install.ps1 | iex"
    # /healthz does not wait for a model, so a service that does not answer
    # within a minute is broken, and a deployment script has to be told.
    if (-not $healthy) { throw "the service did not come up; see $LogFile" }
}

function Uninstall-NanoLLM {
    if (Test-Path -LiteralPath $Exe) {
        Say "removing the $ServiceName service"
        try   { Invoke-Native $Exe @('service', 'uninstall', '-name', $ServiceName) | Out-Null }
        catch { Warn $_.Exception.Message }
    }
    # Without the binary, or after it failed, ask the control manager directly.
    if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
        Say "removing the $ServiceName service with sc.exe"
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        & sc.exe delete $ServiceName | Out-Null
    }
    Close-Firewall
    Remove-FromMachinePath
    if (Test-Path -LiteralPath $Prefix) {
        Say "removing $Prefix"
        Remove-Item -LiteralPath $Prefix -Recurse -Force
    }
    if ($Purge) {
        Say "removing ${DataDir}: the configuration, the keys, the weights and the job database"
        Remove-Item -LiteralPath $DataDir -Recurse -Force -ErrorAction SilentlyContinue
    } else {
        Say "kept $DataDir (the configuration, the keys and the weights); -Purge deletes it"
    }
    Say "done"
}

if ($Help) { Show-Usage; return }

# The piped form cannot pass -Uninstall either.
if ($env:NANOLLM_UNINSTALL -eq '1') { $Uninstall = [switch]$true }
if ($env:NANOLLM_PURGE -eq '1')     { $Purge     = [switch]$true }

if (-not (Test-Administrator)) {
    Invoke-Elevated
    return
}

try {
    if ($Uninstall) { Uninstall-NanoLLM } else { Install-NanoLLM }
} catch {
    Write-Host ""
    Write-Host "install.ps1: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
