#Requires -Version 5.1
#
# Installs, unattended, the CUDA Toolkit for local work on the CUDA backend:
# nvcc and the libraries a ggml-cuda build links (cudart, cuBLAS), and the
# profilers (Nsight Systems, Nsight Compute, CUPTI, NVTX). It is the CUDA
# companion of setup-windows-toolchain.ps1, whose Visual Studio C++ tools,
# CMake and Ninja it expects to find.
#
#   scripts\setup-windows-cuda.cmd         (double click, or from any prompt)
#   powershell -ExecutionPolicy Bypass -File scripts\setup-windows-cuda.ps1
#
# It asks for administrator rights itself (one UAC prompt) and then needs no
# answers. About 10 to 30 minutes: winget downloads the full installer (about
# 3 GB) and checks its hash.
#
# CUDA 13.4 by default: the first line of releases that supports Visual Studio
# 2026 (MSVC 195x) as the host compiler, which is what the toolchain script
# installs; CUDA 12.x refuses it. Since CUDA 13.1 the Windows installer carries
# no display driver, and the subpackages below are named one by one anyway, so
# the installed driver is not touched. CUDA 13.x programs run on drivers 580
# and later (minor version compatibility). The driver version is recorded
# before and after and compared.
#
# Subpackages (names as in the CUDA 13.4 Installation Guide, Table 2):
#   compiler     nvcc crt nvvm nvptxcompiler nvfatbin nvjitlink
#   runtime      cudart cublas cublas_dev nvrtc nvrtc_dev thrust
#   profiling    nsight_systems nsight_compute cupti nvtx cuda_profiler_api
#   inspection   cuobjdump nvdisasm nvprune cuxxfilt sanitizer occupancy_calculator nvml_dev
#   integration  visual_studio_integration (MSBuild files; CMake + Ninja do not need them)
# Left out: the display driver, documentation, samples, cuFFT, cuRAND,
# cuSOLVER, cuSPARSE, NPP, nvJPEG, OpenCL, the Nsight VS plugin.
#
# Checks at the end: nvcc, nsys and ncu run; nvcc with the Visual Studio
# environment compiles a kernel for this GPU and the program runs on it;
# CMake + Ninja configure and build a CUDA project; Nsight Systems records a
# CUDA trace of that program. It also reports (without changing it) whether
# GPU performance counters are open to non-administrators, which Nsight Compute
# needs when it is not run elevated.
#
# Output, in <repo>\.cache\cuda-setup\ (git-ignored):
#   setup.log     everything printed here
#   summary.txt   the table at the end
#   cuda.env      NAME=full path of each tool, for the build scripts
#   winget-*.txt  what the winget call printed
#
# -DryRun    only look: no elevation, no installs, nothing changed
# -NoPause   do not wait for Enter at the end (for unattended runs)
# -Version   the CUDA Toolkit version winget installs (default 13.4)
[CmdletBinding()]
param(
    [switch] $DryRun,
    [switch] $NoPause,
    [string] $Version = '13.4'
)

$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

function Test-Admin {
    ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# --- elevation ------------------------------------------------------------
if (-not $DryRun -and -not (Test-Admin)) {
    Write-Host 'Not elevated: asking Windows for administrator rights (accept the UAC prompt)...'
    $relaunch = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $PSCommandPath), '-Version', $Version)
    if ($NoPause) { $relaunch += '-NoPause' }
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
    $LogDir = Join-Path $repoRoot '.cache\cuda-setup'
} else {
    $LogDir = Join-Path $env:TEMP 'nanollm-cuda-setup'
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

# PATH and CUDA_PATH as a new process would see them: the installer writes to
# the registry, this process still holds the old copy.
function Update-SessionEnv {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($machine, $user) | Where-Object { $_ }) -join ';'
    foreach ($n in @('CUDA_PATH', ('CUDA_PATH_V' + ($Version -replace '\.', '_')))) {
        $v = [Environment]::GetEnvironmentVariable($n, 'Machine')
        if ($v) { Set-Item -Path "Env:$n" -Value $v }
    }
}

# The newest match of a list of wildcard paths.
function Find-File([string[]]$Patterns) {
    foreach ($p in $Patterns) {
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

function Get-Lines([string]$Exe, [string[]]$Arguments = @('--version')) {
    try { return (& $Exe @Arguments 2>&1 | ForEach-Object { "$_" }) } catch { return @() }
}

function Find-Winget {
    $c = Get-Command winget.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return $c.Source }
    $hit = Get-ChildItem "$env:ProgramFiles\WindowsApps\Microsoft.DesktopAppInstaller_*_x64__8wekyb3d8bbwe\winget.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
    if ($hit) { return $hit.FullName }
    return $null
}

function Get-DriverVersion {
    $smi = Find-File @("$env:SystemRoot\System32\nvidia-smi.exe", 'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    if (-not $smi) { return '' }
    $l = Get-Lines $smi @('--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader')
    return (($l | Select-Object -First 1) -as [string]).Trim()
}

# One `winget install`, with a timeout and a heartbeat; what it printed goes to
# the log. Returns the exit code (-1 when it had to be killed).
function Invoke-WingetInstall {
    param([string]$Id, [string]$PackageVersion, [string]$Override, [int]$TimeoutMin = 60)
    $safe = $Id -replace '[^A-Za-z0-9._-]', '_'
    $out = Join-Path $LogDir "winget-$safe.out.txt"
    $err = Join-Path $LogDir "winget-$safe.err.txt"
    $line = "install --id $Id --version $PackageVersion --exact --source winget --silent --disable-interactivity --accept-package-agreements --accept-source-agreements --override `"$Override`""
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

# --- what is installed ----------------------------------------------------
$cudaDir = Join-Path $env:ProgramFiles ("NVIDIA GPU Computing Toolkit\CUDA\v$Version")
$subpackages = @(
    'nvcc', 'crt', 'nvvm', 'nvptxcompiler', 'nvfatbin', 'nvjitlink',
    'cudart', 'cublas', 'cublas_dev', 'nvrtc', 'nvrtc_dev', 'thrust',
    'nsight_systems', 'nsight_compute', 'cupti', 'nvtx', 'cuda_profiler_api',
    'cuobjdump', 'nvdisasm', 'nvprune', 'cuxxfilt', 'sanitizer', 'occupancy_calculator', 'nvml_dev',
    'visual_studio_integration'
) | ForEach-Object { "{0}_{1}" -f $_, $Version }

function Find-Nvcc { Find-File @((Join-Path $cudaDir 'bin\nvcc.exe')) }
function Find-Nsys {
    Find-File @(
        "$env:ProgramFiles\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe",
        (Join-Path $cudaDir 'nsight-systems*\target-windows-x64\nsys.exe'))
}
function Find-Ncu {
    Find-File @(
        "$env:ProgramFiles\NVIDIA Corporation\Nsight Compute *\ncu.exe",
        "$env:ProgramFiles\NVIDIA Corporation\Nsight Compute *\ncu.bat",
        (Join-Path $cudaDir 'nsight-compute*\ncu.exe'))
}

# --- Visual Studio environment (from the toolchain script) ----------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'

function Get-VsInstall {
    if (-not (Test-Path -LiteralPath $vswhere)) { return $null }
    $path = & $vswhere -products '*' -latest -requires 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' -property installationPath 2>$null | Select-Object -First 1
    if (-not $path) { return $null }
    $vcvars = Join-Path $path 'VC\Auxiliary\Build\vcvarsall.bat'
    if (-not (Test-Path -LiteralPath $vcvars)) { return $null }
    $ver = & $vswhere -products '*' -latest -requires 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' -property installationVersion 2>$null | Select-Object -First 1
    return [pscustomobject]@{ Path = $path; Vcvars = $vcvars; Version = $ver }
}

# vcvarsall x64 applied to this process (cl.exe on PATH, INCLUDE, LIB).
function Enter-VsEnv($vs) {
    $cmdFile = Join-Path $env:TEMP 'nanollm-cuda-vcvars.cmd'
    Set-Content -LiteralPath $cmdFile -Encoding ASCII -Value @('@echo off', ('call "{0}" x64 >nul 2>&1' -f $vs.Vcvars), 'set')
    $dump = & cmd.exe /d /c $cmdFile
    Remove-Item -LiteralPath $cmdFile -Force -ErrorAction SilentlyContinue
    $vars = @{}
    foreach ($l in $dump) {
        if ($l -match '^([^=]+)=(.*)$') { $vars[$Matches[1]] = $Matches[2] }
    }
    if (-not $vars.ContainsKey('Path') -or -not (Find-InPathString $vars['Path'] 'cl.exe')) { return $false }
    foreach ($k in $vars.Keys) { [Environment]::SetEnvironmentVariable($k, $vars[$k], 'Process') }
    return $true
}

function Find-Tool([string]$Name, [string[]]$Fallback) {
    $c = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return $c.Source }
    return (Find-File $Fallback)
}

# --- the checks -----------------------------------------------------------
function Invoke-Checks($vs, [string]$nvcc, [string]$nsys) {
    Write-Step 'Checks'
    if (-not (Enter-VsEnv $vs)) {
        Add-Result 'VS C++ environment (vcvarsall x64, cl.exe)' $false "cl.exe not on PATH after $($vs.Vcvars)"
        return
    }
    $cl = Find-InPathString $env:Path 'cl.exe'
    Add-Result 'VS C++ environment (vcvarsall x64)' $true "$($vs.Path) $($vs.Version), cl: $cl"

    $work = Join-Path $env:TEMP ('nanollm-cuda-test-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    Push-Location $work
    try {
        Set-Content -Encoding ASCII -Path 't.cu' -Value @(
            '#include <cstdio>',
            '#include <cuda_runtime.h>',
            '__global__ void add(const float * a, const float * b, float * c, int n) {',
            '    int i = blockIdx.x * blockDim.x + threadIdx.x;',
            '    if (i < n) c[i] = a[i] + b[i];',
            '}',
            'int main() {',
            '    const int n = 1 << 20;',
            '    static float ha[1 << 20], hb[1 << 20], hc[1 << 20];',
            '    for (int i = 0; i < n; i++) { ha[i] = 1.0f; hb[i] = 2.0f; }',
            '    float *a, *b, *c;',
            '    if (cudaMalloc(&a, n * sizeof(float)) != cudaSuccess) { printf("cudaMalloc failed\n"); return 2; }',
            '    cudaMalloc(&b, n * sizeof(float));',
            '    cudaMalloc(&c, n * sizeof(float));',
            '    cudaMemcpy(a, ha, n * sizeof(float), cudaMemcpyHostToDevice);',
            '    cudaMemcpy(b, hb, n * sizeof(float), cudaMemcpyHostToDevice);',
            '    add<<<(n + 255) / 256, 256>>>(a, b, c, n);',
            '    cudaError_t e = cudaMemcpy(hc, c, n * sizeof(float), cudaMemcpyDeviceToHost);',
            '    if (e != cudaSuccess) { printf("kernel failed: %s\n", cudaGetErrorString(e)); return 3; }',
            '    for (int i = 0; i < n; i++) if (hc[i] != 3.0f) { printf("wrong result at %d\n", i); return 4; }',
            '    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);',
            '    int rt = 0, drv = 0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);',
            '    printf("ok: %s sm_%d%d, runtime %d, driver API %d\n", p.name, p.major, p.minor, rt, drv);',
            '    return 0;',
            '}')

        $o = & $nvcc -O2 -arch=native t.cu -o t_cuda.exe 2>&1 | Out-String
        $ok = ($LASTEXITCODE -eq 0) -and (Test-Path 't_cuda.exe')
        Add-Result 'test: nvcc + MSVC compile a kernel for this GPU (-arch=native)' $ok $(if ($ok) { 'ok' } else { ($o.Trim() -split "`n" | Select-Object -Last 8) -join ' | ' })
        if ($ok) {
            $o = & .\t_cuda.exe 2>&1 | Out-String
            $ok = ($LASTEXITCODE -eq 0) -and ($o -match '^ok:')
            Add-Result 'test: the kernel runs on the GPU' $ok $o.Trim()
        }

        $cmake = Find-Tool 'cmake' @('C:\Program Files\CMake\bin\cmake.exe')
        $ninja = Find-Tool 'ninja' @(
            "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ninja.exe",
            "$env:ProgramFiles\WinGet\Links\ninja.exe",
            "$env:USERPROFILE\AppData\Local\Microsoft\WinGet\Links\ninja.exe")
        if (-not $ninja) {
            # the per-user winget link of the user who ran the toolchain script
            $ninja = Find-File @('C:\Users\*\AppData\Local\Microsoft\WinGet\Links\ninja.exe')
        }
        if ($cmake -and $ninja) {
            Set-Content -Encoding ASCII -Path 'CMakeLists.txt' -Value @(
                'cmake_minimum_required(VERSION 3.24)',
                'project(t CUDA)',
                'set(CMAKE_CUDA_ARCHITECTURES native)',
                'add_executable(t t.cu)')
            $nj = $ninja -replace '\\', '/'
            $nv = $nvcc -replace '\\', '/'
            $o = & $cmake -S . -B build -G Ninja "-DCMAKE_MAKE_PROGRAM=$nj" "-DCMAKE_CUDA_COMPILER=$nv" -DCMAKE_BUILD_TYPE=Release 2>&1 | Out-String
            $ok = ($LASTEXITCODE -eq 0)
            if ($ok) {
                $o += & $cmake --build build 2>&1 | Out-String
                $ok = ($LASTEXITCODE -eq 0) -and (Test-Path 'build\t.exe')
            }
            Add-Result 'test: CMake + Ninja configure and build a CUDA project' $ok $(if ($ok) { "ok ($cmake)" } else { ($o.Trim() -split "`n" | Select-Object -Last 12) -join ' | ' })
        } else {
            Add-Result 'test: CMake + Ninja configure and build a CUDA project' $false 'cmake or ninja not found (run setup-windows-toolchain first)'
        }

        if ($nsys -and (Test-Path 't_cuda.exe')) {
            $o = & $nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o (Join-Path $work 'trace') .\t_cuda.exe 2>&1 | Out-String
            $rep = Get-ChildItem -Path $work -Filter 'trace.*' -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in '.nsys-rep', '.qdrep' } | Select-Object -First 1
            $ok = [bool]$rep
            Add-Result 'test: Nsight Systems records a CUDA trace' $ok $(if ($ok) { "ok ($($rep.Name), $([math]::Round($rep.Length / 1KB)) KB)" } else { ($o.Trim() -split "`n" | Select-Object -Last 8) -join ' | ' })
        }
    } finally {
        Pop-Location
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Whether GPU performance counters are restricted to administrators (NVIDIA
# Control Panel > Developer > Manage GPU Performance Counters). Only read.
function Get-CounterAccess {
    $keys = @(
        'HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak',
        'HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Parameters\Global\NVTweak')
    foreach ($k in $keys) {
        $v = (Get-ItemProperty -Path $k -Name 'RmProfilingAdminOnly' -ErrorAction SilentlyContinue).RmProfilingAdminOnly
        if ($null -ne $v) { return [int]$v }
    }
    return $null
}

# --- main -----------------------------------------------------------------
$ready = $false
try {
    Write-Step "NanoLLM CUDA Toolkit $Version setup"
    Write-Log ('user: {0}, admin: {1}, dry run: {2}' -f $env:USERNAME, (Test-Admin), [bool]$DryRun)
    Write-Log ('OS: {0}' -f (Get-CimInstance Win32_OperatingSystem).Caption)
    Write-Log ('logs: {0}' -f $LogDir)
    $freeGb = [math]::Round((Get-PSDrive -Name ($env:SystemDrive.TrimEnd(':'))).Free / 1GB)
    Write-Log "free space on $env:SystemDrive : $freeGb GB (needs about 10)"
    if ($freeGb -lt 12) { Write-Log 'WARNING: little free space; the install may fail' }

    if (-not $DryRun) {
        try {
            Add-Type -Namespace Nano -Name Power -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
            [void][Nano.Power]::SetThreadExecutionState([uint32]2147483649)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            Write-Log 'the machine is kept awake while this script runs'
        } catch { Write-Log "could not set the keep-awake state: $_" }
    }

    $driverBefore = Get-DriverVersion
    if ($driverBefore) {
        Write-Log "GPU and driver: $driverBefore"
        Add-Result 'NVIDIA GPU and driver (nvidia-smi)' $true $driverBefore
    } else {
        Add-Result 'NVIDIA GPU and driver (nvidia-smi)' $false 'nvidia-smi not found: install the NVIDIA driver (580 or later) first'
    }

    $vs = Get-VsInstall
    if ($vs) {
        Write-Log "Visual Studio C++ tools: $($vs.Path) $($vs.Version)"
    } else {
        Add-Result 'Visual Studio C++ tools' $false 'not found: run scripts\setup-windows-toolchain.cmd first'
    }

    Update-SessionEnv
    Write-Step "CUDA Toolkit $Version"
    $nvcc = Find-Nvcc
    $nsys = Find-Nsys
    if ($nvcc -and $nsys) {
        Write-Log "already installed: $nvcc"
    } elseif ($DryRun) {
        Write-Log "missing; would run: winget install Nvidia.CUDA --version $Version --override `"-s $($subpackages -join ' ')`""
        Add-Result "CUDA Toolkit $Version (nvcc)" $false 'missing (dry run, nothing installed)'
    } else {
        $script:Winget = Find-Winget
        if (-not $script:Winget) {
            Add-Result 'winget' $false 'not found: install "App Installer" from the Microsoft Store and run this again'
        } else {
            Write-Log ('winget: {0}' -f $script:Winget)
            $override = '-s ' + ($subpackages -join ' ')
            $code = Invoke-WingetInstall -Id 'Nvidia.CUDA' -PackageVersion $Version -Override $override -TimeoutMin 90
            Update-SessionEnv
            $nvcc = Find-Nvcc
            $nsys = Find-Nsys
            if (-not $nvcc) { Write-Log "nvcc not found after the install (winget code $code)" }
        }
    }

    if ($nvcc) {
        $script:Paths['CUDA_PATH'] = $cudaDir
        $script:Paths['NVCC'] = $nvcc
        $v = (Get-Lines $nvcc) | Where-Object { $_ -match 'release' } | Select-Object -First 1
        Add-Result "CUDA Toolkit $Version (nvcc)" $true "$nvcc ($v)"
    } elseif (-not $DryRun) {
        Add-Result "CUDA Toolkit $Version (nvcc)" $false "nvcc not in $cudaDir\bin (see winget-Nvidia.CUDA.*.txt and %TEMP%\NVIDIA*.log)"
    }
    # the runtime DLLs a ggml-cuda build ships next to itself
    foreach ($lib in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
        if ($nvcc) {
            $f = Find-File @((Join-Path $cudaDir "bin\x64\$lib"), (Join-Path $cudaDir "bin\$lib"))
            Add-Result "  $lib" ([bool]$f) $(if ($f) { $f } else { "not in $cudaDir\bin" })
        }
    }
    if ($nsys) {
        $script:Paths['NSYS'] = $nsys
        Add-Result 'Nsight Systems (nsys)' $true "$nsys ($((Get-Lines $nsys) | Select-Object -First 1))"
    } elseif (-not $DryRun) {
        Add-Result 'Nsight Systems (nsys)' $false 'not found after the install'
    }
    $ncu = Find-Ncu
    if ($ncu) {
        $script:Paths['NCU'] = $ncu
        Add-Result 'Nsight Compute (ncu)' $true $ncu
    } elseif (-not $DryRun) {
        Add-Result 'Nsight Compute (ncu)' $false 'not found after the install'
    }

    $driverAfter = Get-DriverVersion
    if ($driverBefore -and $driverAfter) {
        Add-Result 'display driver unchanged' ($driverBefore -eq $driverAfter) $(if ($driverBefore -eq $driverAfter) { $driverAfter } else { "was $driverBefore, now $driverAfter" })
    }

    if ($vs -and $nvcc) { Invoke-Checks $vs $nvcc $nsys }

    $acc = Get-CounterAccess
    if ($null -eq $acc) {
        Add-Result 'GPU performance counters for non-administrators' $false 'setting not found: Nsight Compute may need to run elevated' $false
    } elseif ($acc -eq 0) {
        Add-Result 'GPU performance counters for non-administrators' $true 'open (RmProfilingAdminOnly = 0)' $false
    } else {
        Add-Result 'GPU performance counters for non-administrators' $false 'administrators only: run ncu elevated, or allow all users in NVIDIA Control Panel > Developer > Manage GPU Performance Counters (Nsight Systems kernel traces do not need it)' $false
    }
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
$verdict = "READY: CUDA Toolkit $Version, nvcc with Visual Studio, CMake + Ninja and Nsight Systems work."
if (-not $ready) { $verdict = 'NOT READY: {0} required item(s) failed (marked FAIL above).' -f $bad.Count }
if ($DryRun) { $verdict = 'DRY RUN (nothing was installed). ' + $verdict }
$lines.Add('')
$lines.Add($verdict)
if ($script:RebootNeeded) { $lines.Add('A reboot is recommended before the first build (the installer asked for it).') }
$lines.Add("Logs: $LogDir")
$lines | ForEach-Object { Write-Host $_ }
Set-Content -LiteralPath (Join-Path $LogDir 'summary.txt') -Encoding ASCII -Value $lines
Set-Content -LiteralPath (Join-Path $LogDir 'cuda.env') -Encoding ASCII -Value @($script:Paths.Keys | ForEach-Object { '{0}={1}' -f $_, $script:Paths[$_] })

if ($transcribing) { try { Stop-Transcript | Out-Null } catch {} }
if (-not $NoPause -and -not $DryRun) {
    try { [void](Read-Host 'Done. Press Enter to close this window') } catch {}
}
if ($ready) { exit 0 } else { exit 1 }
