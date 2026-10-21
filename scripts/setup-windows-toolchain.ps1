#Requires -Version 5.1
#
# Installs, unattended, everything a local build of the patched llama.cpp
# libraries (CPU variants with clang-cl, upstream-style Vulkan backend and its
# shaders) needs on Windows. It follows the recipe of .github/workflows/
# libs-build.yml (Visual Studio C++ environment, LLVM clang, CMake, Ninja) and
# adds the Vulkan SDK (glslc) for the shaders of the iGPU work.
#
#   scripts\setup-windows-toolchain.cmd         (double click, or from any prompt)
#   powershell -ExecutionPolicy Bypass -File scripts\setup-windows-toolchain.ps1
#
# It asks for administrator rights itself (one UAC prompt) and then needs no
# answers: start it and leave. The machine is kept awake while it runs (about
# 30 to 60 minutes, the Visual Studio part is most of it); a package that fails
# does not stop the others. At the end it checks every tool, compiles and links
# a test program with cl and clang-cl, configures and builds one with CMake and
# Ninja, compiles one compute shader with glslc, and prints a summary.
#
# What it installs (winget, skipped when the tool is already there):
#   Kitware.CMake, Ninja-build.Ninja, Git.Git (only if git is missing),
#   LLVM.LLVM, KhronosGroup.VulkanSDK,
#   Microsoft.VisualStudio.BuildTools (2026, the generation CI uses; 2022 and a
#   plain C++ workload are the fallbacks) with the C++ workload and clang.
# It does not change PATH by hand, power plans, services or anything else;
# the installers add what they add. Nothing is removed.
#
# Output, in <repo>\.cache\toolchain-setup\ (git-ignored):
#   setup.log     everything printed here
#   summary.txt   the table at the end
#   toolchain.env NAME=full path of each tool, for the build scripts
#   winget-*.txt  what each winget call printed
#
# -DryRun    only look: no elevation, no installs, nothing changed
# -NoPause   do not wait for Enter at the end (for unattended runs)
# -SkipVisualStudio   leave out the Visual Studio Build Tools
[CmdletBinding()]
param(
    [switch] $DryRun,
    [switch] $NoPause,
    [switch] $SkipVisualStudio
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

function Test-Admin {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# --- elevation ------------------------------------------------------------
if (-not $DryRun -and -not (Test-Admin)) {
    Write-Host 'Not elevated: asking Windows for administrator rights (accept the UAC prompt)...'
    $relaunch = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $PSCommandPath))
    if ($NoPause) { $relaunch += '-NoPause' }
    if ($SkipVisualStudio) { $relaunch += '-SkipVisualStudio' }
    try {
        Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList $relaunch -Verb RunAs
    } catch {
        Write-Host "Elevation was refused or failed: $_"
        exit 1
    }
    exit 0
}

# --- log directory --------------------------------------------------------
$repoRoot = $null
if ($PSScriptRoot) { $repoRoot = Split-Path -Parent $PSScriptRoot }
if ($repoRoot -and (Test-Path -LiteralPath (Join-Path $repoRoot 'go.mod'))) {
    $LogDir = Join-Path $repoRoot '.cache\toolchain-setup'
} else {
    $LogDir = Join-Path $env:TEMP 'nanollm-toolchain-setup'
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$transcribing = $false
try { Start-Transcript -Path (Join-Path $LogDir 'setup.log') -Force | Out-Null; $transcribing = $true } catch {}

$script:Results = New-Object System.Collections.Generic.List[object]
$script:Paths = [ordered]@{}
$script:RebootNeeded = $false
$script:Winget = $null

function Write-Log([string]$Message) {
    Write-Host ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $Message)
}

function Write-Step([string]$Title) {
    Write-Host ''
    Write-Host ('=== {0} ===' -f $Title)
}

function Add-Result([string]$Name, [bool]$Ok, [string]$Detail, [bool]$Required = $true) {
    $script:Results.Add([pscustomobject]@{ Name = $Name; Ok = $Ok; Detail = $Detail; Required = $Required })
}

# PATH and VULKAN_SDK as a new process would see them: the installers write
# to the registry, this process still holds the old copy.
function Update-SessionEnv {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
    foreach ($n in 'VULKAN_SDK', 'VK_SDK_PATH') {
        $v = [Environment]::GetEnvironmentVariable($n, 'Machine')
        if ($v) { Set-Item -Path "Env:$n" -Value $v }
    }
}

# A tool by name on PATH, else by the first match of a list of wildcard paths
# (the newest directory first).
function Find-Exe([string]$Name, [string[]]$Fallback = @()) {
    $c = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return $c.Source }
    foreach ($p in $Fallback) {
        $hit = Get-ChildItem -Path $p -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

function Find-InPathString([string]$PathString, [string]$Exe) {
    foreach ($d in ($PathString -split ';')) {
        if (-not $d) { continue }
        $f = Join-Path $d $Exe
        if (Test-Path -LiteralPath $f) { return $f }
    }
    return $null
}

function Get-FirstLine([string]$Exe, [string]$Arg = '--version') {
    try { return ((& $Exe $Arg 2>&1 | Select-Object -First 1) -as [string]).Trim() } catch { return '' }
}

function Find-Winget {
    $c = Get-Command winget.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return $c.Source }
    $hit = Get-ChildItem "$env:ProgramFiles\WindowsApps\Microsoft.DesktopAppInstaller_*_x64__8wekyb3d8bbwe\winget.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
    if ($hit) { return $hit.FullName }
    return $null
}

# One `winget install`, with a timeout and a heartbeat; what it printed goes to
# the log. Returns the exit code (-1 when it had to be killed).
function Invoke-WingetInstall {
    param([string]$Id, [string]$Override = '', [int]$TimeoutMin = 40)
    $safe = $Id -replace '[^A-Za-z0-9._-]', '_'
    $out = Join-Path $LogDir "winget-$safe.out.txt"
    $err = Join-Path $LogDir "winget-$safe.err.txt"
    $line = "install --id $Id --exact --source winget --silent --disable-interactivity --accept-package-agreements --accept-source-agreements"
    if ($Override) { $line += " --override `"$Override`"" }
    Write-Log "winget $line"
    $p = Start-Process -FilePath $script:Winget -ArgumentList $line -PassThru -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle
    $t0 = Get-Date
    $killed = $false
    while (-not $p.WaitForExit(300000)) {
        $min = [int]((Get-Date) - $t0).TotalMinutes
        Write-Log "  ... still running after $min min"
        if ($min -ge $TimeoutMin) {
            Write-Log '  timeout: killing the installer'
            & taskkill.exe /PID $p.Id /T /F | Out-Null
            $killed = $true
            break
        }
    }
    $code = -1
    if (-not $killed) { $code = $p.ExitCode }
    foreach ($f in @($out, $err)) {
        if (Test-Path -LiteralPath $f) {
            Get-Content -LiteralPath $f -Encoding UTF8 -Tail 25 -ErrorAction SilentlyContinue |
                ForEach-Object { ($_ -replace '[^\x20-\x7E]', '').Trim() } |
                Where-Object { $_ -match '[A-Za-z0-9]{3}' } |
                ForEach-Object { Write-Log "  | $_" }
        }
    }
    Write-Log "  winget exit code: $code"
    if ($code -eq 3010 -or $code -eq -1978335223) { $script:RebootNeeded = $true }
    return $code
}

# --- Visual Studio --------------------------------------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'

function Get-VsInstall([string]$Requires = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64') {
    if (-not (Test-Path -LiteralPath $vswhere)) { return $null }
    $path = & $vswhere -products '*' -latest -requires $Requires -property installationPath 2>$null | Select-Object -First 1
    if (-not $path) { return $null }
    $vcvars = Join-Path $path 'VC\Auxiliary\Build\vcvarsall.bat'
    if (-not (Test-Path -LiteralPath $vcvars)) { return $null }
    return [pscustomobject]@{ Path = $path; Vcvars = $vcvars }
}

$vsComponents = @(
    'Microsoft.VisualStudio.Workload.VCTools',
    'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
    'Microsoft.VisualStudio.Component.VC.Llvm.Clang',
    'Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset'
)

function Install-VisualStudio {
    $existing = Get-VsInstall
    if ($existing) {
        Write-Log "C++ tools already installed in $($existing.Path)"
        return $existing
    }
    $add = ($vsComponents | ForEach-Object { "--add $_" }) -join ' '
    $full = "--quiet --wait --norestart --includeRecommended $add"
    $plain = '--quiet --wait --norestart --includeRecommended --add Microsoft.VisualStudio.Workload.VCTools'
    $attempts = @(
        @{ Id = 'Microsoft.VisualStudio.BuildTools'; Override = $full; Label = 'VS 2026 Build Tools, C++ and clang' },
        @{ Id = 'Microsoft.VisualStudio.2022.BuildTools'; Override = $full; Label = 'VS 2022 Build Tools, C++ and clang' },
        @{ Id = 'Microsoft.VisualStudio.BuildTools'; Override = $plain; Label = 'VS 2026 Build Tools, C++ workload only' }
    )
    foreach ($a in $attempts) {
        Write-Log "attempt: $($a.Label)"
        $code = Invoke-WingetInstall -Id $a.Id -Override $a.Override -TimeoutMin 150
        Update-SessionEnv
        $vs = Get-VsInstall
        if ($vs) { return $vs }
        # winget says "already installed" for a Build Tools without the C++
        # workload: add the workload through the Visual Studio Installer.
        $setup = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\setup.exe'
        $any = $null
        if (Test-Path -LiteralPath $vswhere) {
            $any = & $vswhere -products '*' -latest -property installationPath 2>$null | Select-Object -First 1
        }
        if ($any -and (Test-Path -LiteralPath $setup)) {
            Write-Log "Visual Studio is in $any without the C++ tools; modifying it"
            $line = "modify --installPath `"$any`" --quiet --norestart --includeRecommended $add"
            $p = Start-Process -FilePath $setup -ArgumentList $line -PassThru -Wait
            Write-Log "  installer exit code: $($p.ExitCode)"
            if ($p.ExitCode -eq 3010) { $script:RebootNeeded = $true }
            $vs = Get-VsInstall
            if ($vs) { return $vs }
        }
        Write-Log "that attempt left no C++ tools (winget code $code); trying the next one"
    }
    return $null
}

# --- the tools winget installs --------------------------------------------
$tools = @(
    @{ Name = 'CMake'; Id = 'Kitware.CMake'; Key = 'CMAKE'; Required = $true
       Probe = { Find-Exe 'cmake' @('C:\Program Files\CMake\bin\cmake.exe') } },
    @{ Name = 'Ninja'; Id = 'Ninja-build.Ninja'; Key = 'NINJA'; Required = $true
       Probe = { Find-Exe 'ninja' @(
            "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ninja.exe",
            "$env:ProgramFiles\WinGet\Links\ninja.exe",
            "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ninja-build.Ninja_*\ninja.exe",
            "$env:ProgramFiles\WinGet\Packages\Ninja-build.Ninja_*\ninja.exe") } },
    @{ Name = 'Git'; Id = 'Git.Git'; Key = 'GIT'; Required = $true
       Probe = { Find-Exe 'git' @('C:\Program Files\Git\cmd\git.exe') } },
    @{ Name = 'LLVM (clang-cl)'; Id = 'LLVM.LLVM'; Key = 'CLANG_CL'; Required = $true
       Probe = { Find-Exe 'clang-cl' @('C:\Program Files\LLVM\bin\clang-cl.exe') } },
    @{ Name = 'Vulkan SDK (glslc)'; Id = 'KhronosGroup.VulkanSDK'; Key = 'GLSLC'; Required = $true
       Probe = { Find-Exe 'glslc' @('C:\VulkanSDK\*\Bin\glslc.exe') } }
)

function Install-Tool($t) {
    Write-Step $t.Name
    $found = & $t.Probe
    if ($found) {
        Write-Log "already present: $found"
        $script:Paths[$t.Key] = $found
        Add-Result $t.Name $true "present: $found" $t.Required
        return
    }
    if ($DryRun) {
        Write-Log "missing; would run: winget install $($t.Id)"
        Add-Result $t.Name $false 'missing (dry run, nothing installed)' $t.Required
        return
    }
    [void](Invoke-WingetInstall -Id $t.Id -TimeoutMin 40)
    Update-SessionEnv
    $found = & $t.Probe
    if (-not $found) {
        Write-Log 'not found after the install; one more try'
        [void](Invoke-WingetInstall -Id $t.Id -TimeoutMin 40)
        Update-SessionEnv
        $found = & $t.Probe
    }
    if ($found) {
        Write-Log "installed: $found"
        $script:Paths[$t.Key] = $found
        Add-Result $t.Name $true "installed: $found" $t.Required
    } else {
        Write-Log 'FAILED: the tool is still not there (see winget-*.txt in the log folder)'
        Add-Result $t.Name $false "not found after winget install $($t.Id)" $t.Required
    }
}

# --- the check: environment of vcvarsall and a few real compilations -------
function Invoke-Checks($vs) {
    Write-Step 'Checks'
    $cmdFile = Join-Path $env:TEMP 'nanollm-vcvars.cmd'
    Set-Content -LiteralPath $cmdFile -Encoding ASCII -Value @('@echo off', ('call "{0}" x64 >nul 2>&1' -f $vs.Vcvars), 'set')
    $dump = & cmd.exe /d /c $cmdFile
    Remove-Item -LiteralPath $cmdFile -Force -ErrorAction SilentlyContinue
    $vars = @{}
    foreach ($l in $dump) {
        if ($l -match '^([^=]+)=(.*)$') { $vars[$Matches[1]] = $Matches[2] }
    }
    if (-not $vars.ContainsKey('Path') -or -not (Find-InPathString $vars['Path'] 'cl.exe')) {
        Add-Result 'VS C++ environment (vcvarsall x64, cl.exe)' $false "cl.exe not on PATH after $($vs.Vcvars)"
        return
    }
    foreach ($k in $vars.Keys) { [Environment]::SetEnvironmentVariable($k, $vars[$k], 'Process') }
    $cl = Find-InPathString $vars['Path'] 'cl.exe'
    $script:Paths['VS_PATH'] = $vs.Path
    $script:Paths['VCVARSALL'] = $vs.Vcvars
    $script:Paths['CL'] = $cl
    Add-Result 'VS C++ tools (cl, link, Windows SDK)' $true "$($vs.Path)  cl: $cl"

    $bundled = Get-ChildItem -Path (Join-Path $vs.Path 'VC\Tools\Llvm\x64\bin\clang-cl.exe') -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
    if ($bundled) {
        $script:Paths['CLANG_CL_VS'] = $bundled
        Add-Result 'clang-cl bundled with Visual Studio' $true $bundled $false
    } else {
        Add-Result 'clang-cl bundled with Visual Studio' $false 'not installed; the standalone LLVM is used' $false
    }

    # tools missing from the PATH of a plain prompt may be on the one vcvarsall made (CMake and Ninja of VS)
    foreach ($pair in @(@('CMAKE', 'cmake.exe', 'CMake'), @('NINJA', 'ninja.exe', 'Ninja'))) {
        if (-not $script:Paths.Contains($pair[0])) {
            $hit = Find-InPathString $vars['Path'] $pair[1]
            if ($hit) {
                $script:Paths[$pair[0]] = $hit
                foreach ($r in $script:Results) { if ($r.Name -eq $pair[2]) { $r.Ok = $true; $r.Detail = "from the Visual Studio environment: $hit" } }
            }
        }
    }

    $work = Join-Path $env:TEMP ('nanollm-toolchain-test-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    Push-Location $work
    try {
        Set-Content -Encoding ASCII -Path 't.c' -Value @('#include <windows.h>', 'int main(void) { return (int)(GetTickCount() * 0u); }')
        Set-Content -Encoding ASCII -Path 'CMakeLists.txt' -Value @('cmake_minimum_required(VERSION 3.20)', 'project(t C)', 'add_executable(t t.c)')
        Set-Content -Encoding ASCII -Path 't.comp' -Value @('#version 450', 'layout(local_size_x = 1) in;', 'void main() {}')

        $o = & cl.exe /nologo t.c /Fet_cl.exe 2>&1 | Out-String
        $ok = ($LASTEXITCODE -eq 0) -and (Test-Path 't_cl.exe')
        Add-Result 'test: cl compiles and links a Windows program' $ok $(if ($ok) { 'ok' } else { $o.Trim() })

        $clangcl = $script:Paths['CLANG_CL']
        if (-not $clangcl) { $clangcl = $script:Paths['CLANG_CL_VS'] }
        if ($clangcl) {
            $o = & $clangcl /nologo t.c /Fet_clang.exe 2>&1 | Out-String
            $ok = ($LASTEXITCODE -eq 0) -and (Test-Path 't_clang.exe')
            Add-Result 'test: clang-cl compiles and links a Windows program' $ok $(if ($ok) { "ok ($(Get-FirstLine $clangcl))" } else { $o.Trim() })
        } else {
            Add-Result 'test: clang-cl compiles and links a Windows program' $false 'no clang-cl found'
        }

        $cmake = $script:Paths['CMAKE']; $ninja = $script:Paths['NINJA']
        if ($cmake -and $ninja -and $clangcl) {
            $cc = $clangcl -replace '\\', '/'
            $nj = $ninja -replace '\\', '/'
            $o = & $cmake -S . -B build -G Ninja "-DCMAKE_MAKE_PROGRAM=$nj" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_COMPILER=$cc" 2>&1 | Out-String
            $ok = ($LASTEXITCODE -eq 0)
            if ($ok) {
                $o += & $cmake --build build 2>&1 | Out-String
                $ok = ($LASTEXITCODE -eq 0) -and (Test-Path 'build\t.exe')
            }
            Add-Result 'test: CMake + Ninja + clang-cl configure and build' $ok $(if ($ok) { 'ok' } else { ($o.Trim() -split "`n" | Select-Object -Last 12) -join ' | ' })
        } else {
            Add-Result 'test: CMake + Ninja + clang-cl configure and build' $false 'cmake, ninja or clang-cl is missing'
        }

        $glslc = $script:Paths['GLSLC']
        if ($glslc) {
            $o = & $glslc -fshader-stage=compute t.comp -o t.spv 2>&1 | Out-String
            $ok = ($LASTEXITCODE -eq 0) -and (Test-Path 't.spv')
            Add-Result 'test: glslc compiles a compute shader' $ok $(if ($ok) { "ok ($(Get-FirstLine $glslc))" } else { $o.Trim() })
        } else {
            Add-Result 'test: glslc compiles a compute shader' $false 'no glslc found'
        }
    } finally {
        Pop-Location
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# --- main -----------------------------------------------------------------
$ready = $false
try {
    Write-Step 'NanoLLM Windows toolchain setup'
    Write-Log ('user: {0}, admin: {1}, dry run: {2}' -f $env:USERNAME, (Test-Admin), [bool]$DryRun)
    Write-Log ('OS: {0}' -f (Get-CimInstance Win32_OperatingSystem).Caption)
    Write-Log ('logs: {0}' -f $LogDir)
    $freeGb = [math]::Round((Get-PSDrive -Name ($env:SystemDrive.TrimEnd(':'))).Free / 1GB)
    Write-Log "free space on $env:SystemDrive : $freeGb GB (needs about 15)"
    if ($freeGb -lt 20) { Write-Log 'WARNING: little free space; the Visual Studio part may fail' }

    if (-not $DryRun) {
        try {
            Add-Type -Namespace Nano -Name Power -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
            [void][Nano.Power]::SetThreadExecutionState([uint32]2147483649)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            Write-Log 'the machine is kept awake while this script runs'
        } catch { Write-Log "could not set the keep-awake state: $_" }
    }

    Update-SessionEnv
    $script:Winget = Find-Winget
    if (-not $script:Winget) {
        Add-Result 'winget' $false 'not found: install "App Installer" from the Microsoft Store and run this again'
    } else {
        Write-Log ('winget: {0} ({1})' -f $script:Winget, (Get-FirstLine $script:Winget))
        # small tools first, so that a failed Visual Studio does not hide them
        foreach ($t in $tools) { Install-Tool $t }

        Write-Step 'Visual Studio Build Tools (C++ workload, clang)'
        $vs = $null
        if ($SkipVisualStudio) {
            Write-Log 'skipped (-SkipVisualStudio)'
            $vs = Get-VsInstall
        } elseif ($DryRun) {
            $vs = Get-VsInstall
            if (-not $vs) { Write-Log 'missing; would run: winget install Microsoft.VisualStudio.BuildTools (with the C++ workload and clang)' }
        } else {
            $vs = Install-VisualStudio
        }
        if ($vs) {
            Invoke-Checks $vs
        } else {
            Add-Result 'VS C++ tools (cl, link, Windows SDK)' $false $(if ($DryRun) { 'missing (dry run, nothing installed)' } else { 'not installed (see winget-Microsoft.VisualStudio*.txt and %TEMP%\dd_*.log)' })
        }
    }

    # Perl for the OpenMP fetch of the Windows build (CMake finds none on PATH): the one of Git for Windows
    $perl = Find-Exe 'perl' @('C:\Program Files\Git\usr\bin\perl.exe')
    if ($perl) { $script:Paths['PERL'] = $perl }
    Add-Result 'Perl (for the build; Git for Windows has one)' ([bool]$perl) $(if ($perl) { $perl } else { 'none found' }) $false
    $py = Find-Exe 'python' @()
    Add-Result 'Python 3' ([bool]$py) $(if ($py) { "$py ($(Get-FirstLine $py))" } else { 'none found' }) $false
} catch {
    Write-Log "UNEXPECTED ERROR: $_"
    Add-Result 'setup script' $false "$_"
}

# --- summary --------------------------------------------------------------
Write-Step 'SUMMARY'
$lines = New-Object System.Collections.Generic.List[string]
foreach ($r in $script:Results) {
    $tag = 'OK  '
    if (-not $r.Ok) { if ($r.Required) { $tag = 'FAIL' } else { $tag = 'note' } }
    $lines.Add(('{0}  {1}  -  {2}' -f $tag, $r.Name, $r.Detail))
}
$bad = @($script:Results | Where-Object { $_.Required -and -not $_.Ok })
$ready = ($bad.Count -eq 0) -and ($script:Results.Count -gt 0)
$verdict = 'READY: every required tool is installed and the test builds passed.'
if (-not $ready) { $verdict = 'NOT READY: {0} required item(s) failed (marked FAIL above).' -f $bad.Count }
if ($DryRun) { $verdict = 'DRY RUN (nothing was installed). ' + $verdict }
$lines.Add('')
$lines.Add($verdict)
if ($script:RebootNeeded) { $lines.Add('A reboot is recommended before the first build (an installer asked for it).') }
$lines.Add("Logs: $LogDir")
$lines | ForEach-Object { Write-Host $_ }
Set-Content -LiteralPath (Join-Path $LogDir 'summary.txt') -Encoding ASCII -Value $lines
Set-Content -LiteralPath (Join-Path $LogDir 'toolchain.env') -Encoding ASCII -Value @($script:Paths.Keys | ForEach-Object { '{0}={1}' -f $_, $script:Paths[$_] })

if ($transcribing) { try { Stop-Transcript | Out-Null } catch {} }
if (-not $NoPause -and -not $DryRun) {
    try { [void](Read-Host 'Done. Press Enter to close this window') } catch {}
}
if ($ready) { exit 0 } else { exit 1 }
