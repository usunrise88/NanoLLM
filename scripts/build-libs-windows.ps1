#Requires -Version 5.1
<#
.SYNOPSIS
  Build NanoLLM's llama.cpp libraries on Windows: the Windows twin of
  scripts/build-libs.sh plus the "Build (Windows)" step of
  .github/workflows/libs-build.yml (VS 18 environment, clang-cl through
  cmake/x64-windows-llvm.cmake, Ninja Multi-Config, LLVM OpenMP fetched).

.DESCRIPTION
  The source is the upstream tag in scripts/llama.cpp.base with every
  patches/llama.cpp/*.patch applied (and patches/next/*.patch with -Next), in
  .cache/llama.cpp/src, by the same script build-libs.sh uses
  (--prepare-only, through Git's bash), so the commit hash is the same on every
  machine. The tools it needs are those scripts\setup-windows-toolchain.ps1
  installs (it reads .cache\toolchain-setup\toolchain.env).

  .\scripts\build-libs-windows.ps1                       # alderlake only, patched series
  .\scripts\build-libs-windows.ps1 -Next                 # plus patches/next
  .\scripts\build-libs-windows.ps1 -Variants all -Vulkan # what CI builds, with ggml-vulkan.dll
  .\scripts\build-libs-windows.ps1 -Variants alderlake -Out C:\x\lib -Reuse
  .\scripts\build-libs-windows.ps1 -Variants none -Core -Reuse   # llama/ggml/ggml-base only (a libllama change)
  .\scripts\build-libs-windows.ps1 -Cuda -Tests -Tools -Next -Reuse   # ggml-cuda.dll for sm_86 (+ CUDA runtime) in out-windows_amd64-next-cuda\lib,
                                                                      # test-backend-ops, llama-bench, llama-perplexity in ..\tools

  The result is a lib directory: llama.dll, ggml-base.dll, ggml.dll, the CPU
  variants asked for, libomp.dll. Put the ggml-cpu-*.dll files of a build with
  a changed kernel into a copy of dist\windows_amd64\lib (the ggml-base.dll
  must be the one of the same commit; set NANOLLM_DEV_SERIES to the printed
  commit, patches/llama.cpp/README.md).
#>
param(
    # CPU variants to build: names without the ggml-cpu- prefix, 'all', or 'none' (GGML_CPU_ALL_VARIANTS off, in a
    # build directory of its own: the ggml target depends on every variant, and sapphirerapids' amx/mmq.cpp takes
    # clang 22 most of an hour).
    [string[]]$Variants = @('alderlake'),
    [switch]$Next,
    [switch]$Vulkan,
    # ggml-cuda.dll with nvcc (CUDA_PATH of .cache\cuda-setup\cuda.env, scripts\setup-windows-cuda.ps1) and MSVC as the
    # host compiler: nvcc does not take clang-cl, so this is a build directory of its own, without the LLVM toolchain
    # file and with the plain ggml-cpu.dll (MSVC; no variants) that test-backend-ops compares against.
    [switch]$Cuda,
    # -Cuda: the architectures, CMAKE_CUDA_ARCHITECTURES. 86-real is the dev card only (a build of minutes); the
    # release list is the one of ggml/src/ggml-cuda/CMakeLists.txt, which -CudaArch default leaves to ggml.
    [string]$CudaArch = '86-real',
    # -Cuda: also build test-backend-ops (the op tests and perf runs of the CUDA kernels against the CPU backend).
    [switch]$Tests,
    # The ggml-base/llama/ggml targets. Off: only the CPU variants (a kernel change).
    [switch]$Core,
    # Also build llama-server, llama-bench and llama-perplexity (the golden recorder, the A/B bench and the KLD
    # check) into <Out>\..\tools.
    [switch]$Tools,
    [string]$Out,
    [string]$Build,
    [string]$Src,
    # Skip the source preparation (the tree is already patched; keep local edits).
    [switch]$Reuse,
    [switch]$PrepareOnly,
    # OpenMP runtime: 'vs' uses the libomp that ships with the VS clang (dev builds, no download);
    # 'fetch' is what CI does (GGML_OPENMP_FETCH: LLVM 20.1.8, needs clang 20.x and 7-Zip).
    [ValidateSet('vs', 'fetch')][string]$OpenMP = 'vs'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Read-Env($path) {
    $h = @{}
    if (Test-Path $path) {
        foreach ($l in Get-Content $path) { if ($l -match '^([A-Z_0-9]+)=(.*)$') { $h[$Matches[1]] = $Matches[2] } }
    }
    $h
}
$tc = Read-Env (Join-Path $root '.cache\toolchain-setup\toolchain.env')
if (-not $tc.VCVARSALL) { throw 'run scripts\setup-windows-toolchain.cmd first (no .cache\toolchain-setup\toolchain.env)' }
$perl = if ($tc.PERL) { $tc.PERL } else { 'C:\Program Files\Git\usr\bin\perl.exe' }
$bash = 'C:\Program Files\Git\bin\bash.exe'

if (-not $Src) { $Src = Join-Path $root '.cache\llama.cpp\src' }
$noVariants = $Variants -contains 'none'
$cu = @{}
if ($Cuda) {
    $cu = Read-Env (Join-Path $root '.cache\cuda-setup\cuda.env')
    if (-not $cu.CUDA_PATH) { throw 'run scripts\setup-windows-cuda.cmd first (no .cache\cuda-setup\cuda.env)' }
    if ($Vulkan) { throw '-Cuda and -Vulkan are separate builds' }
}
$suffix = $(if ($Next) { '-next' } else { '' }) + $(if ($Cuda) { '-cuda' } else { '' })
if (-not $Out) { $Out = Join-Path $root ('.cache\llama.cpp\out-windows_amd64' + $suffix + '\lib') }
if (-not $Build) { $Build = Join-Path $root ('.cache\llama.cpp\build-windows_amd64' + $suffix + $(if ($noVariants) { '-novariants' } else { '' })) }

# --- the source -------------------------------------------------------------
if (-not $Reuse) {
    $a = @('scripts/build-libs.sh', '--prepare-only', '--src', ($Src -replace '\\', '/'))
    if ($Next) { $a += '--next' }
    & $bash @a
    if ($LASTEXITCODE -ne 0) { throw "source preparation failed ($LASTEXITCODE)" }
}
$commit = (git -C $Src rev-parse HEAD).Trim()
Write-Host "source: $Src at $commit"
if ($PrepareOnly) { return }

# --- the build --------------------------------------------------------------
$targets = @()
if ($Core) { $targets += @('llama', 'ggml-base', 'ggml') }
if ($Variants -contains 'all') {
    $targets += @('ggml')
} elseif (-not $noVariants) {
    foreach ($v in $Variants) { $targets += "ggml-cpu-$v" }
}
if ($Vulkan) { $targets += 'ggml-vulkan' }
if ($Cuda) {
    # the CUDA build has no CPU variants of its own: one ggml-cpu.dll, the reference of test-backend-ops and the
    # CPU backend of the llama-bench beside it
    $targets = @('ggml-cuda')
    if ($Tools -or $Tests) { $targets += 'ggml-cpu' }
}
if ($Tools) { $targets += @('llama-server', 'llama-bench', 'llama-perplexity') }
if ($Tests) { $targets += 'test-backend-ops'; if ($Cuda) { $targets += 'test-nano-batch-invariance' } }
$targets = $targets | Select-Object -Unique

$flags = @(
    '-DGGML_BACKEND_DL=ON', '-DGGML_NATIVE=OFF',
    ('-DLLAMA_BUILD_COMMON=' + $(if ($Tools) { 'ON' } else { 'OFF' })), ('-DLLAMA_BUILD_TOOLS=' + $(if ($Tools) { 'ON' } else { 'OFF' })), '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=OFF', ('-DLLAMA_BUILD_SERVER=' + $(if ($Tools) { 'ON' } else { 'OFF' })), '-DLLAMA_OPENSSL=OFF',
    '-DGGML_OPENMP=ON', '-DGGML_CPU=ON', ('-DGGML_CPU_ALL_VARIANTS=' + $(if ($noVariants) { 'OFF' } else { 'ON' })),
    ('-DPERL_EXECUTABLE="{0}"' -f ($perl -replace '\\', '/')),
    # clang 21+ makes this an error; upstream's quants.c passes a block pointer to _mm_prefetch
    '-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types'
)
if ($Vulkan) { $flags += '-DGGML_VULKAN=ON' }
$vsLlvm = $null
if ($OpenMP -eq 'fetch') {
    $flags += '-DGGML_OPENMP_FETCH=ON'
} else {
    $vsLlvm = Split-Path -Parent (Split-Path -Parent $tc.CLANG_CL_VS)   # the x64 directory of VS's Llvm
    $lib = ((Join-Path $vsLlvm 'lib\libomp.lib') -replace '\\', '/')
    $flags += @('-DOpenMP_C_FLAGS=-fopenmp=libomp', '-DOpenMP_CXX_FLAGS=-fopenmp=libomp',
        '-DOpenMP_C_LIB_NAMES=libomp', '-DOpenMP_CXX_LIB_NAMES=libomp', ('-DOpenMP_libomp_LIBRARY="{0}"' -f $lib))
}

if ($Cuda) {
    # nvcc takes MSVC as its host compiler (cl.exe from vcvarsall), not clang-cl; MSVC's own OpenMP serves ggml-cpu.
    $vsLlvm = $null
    $flags = $flags | Where-Object { $_ -notmatch '^-D(OpenMP_|CMAKE_C_FLAGS=)' -and $_ -notmatch 'GGML_OPENMP_FETCH' }
    $flags = $flags | ForEach-Object {
        $_ -replace '^-DLLAMA_BUILD_TESTS=OFF$', ('-DLLAMA_BUILD_TESTS=' + $(if ($Tests) { 'ON' } else { 'OFF' })) `
           -replace '^-DLLAMA_BUILD_COMMON=OFF$', ('-DLLAMA_BUILD_COMMON=' + $(if ($Tools -or $Tests) { 'ON' } else { 'OFF' })) `
           -replace '^-DGGML_CPU_ALL_VARIANTS=ON$', '-DGGML_CPU_ALL_VARIANTS=OFF'
    }
    $cudaHome = $cu.CUDA_PATH -replace '\\', '/'
    $flags += @('-DGGML_CUDA=ON', '-DGGML_CUDA_GRAPHS=ON', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl',
        ('-DCMAKE_CUDA_COMPILER="{0}/bin/nvcc.exe"' -f $cudaHome), ('-DCUDAToolkit_ROOT="{0}"' -f $cudaHome))
    if ($CudaArch -and $CudaArch -ne 'default') { $flags += ('-DCMAKE_CUDA_ARCHITECTURES={0}' -f $CudaArch) }
    $configure = 'cmake -S "{0}" -B "{1}" -G "Ninja Multi-Config" {2} || exit /b 2' -f $Src, $Build, ($flags -join ' ')
    $cudaEnv = "set CUDA_PATH=$($cu.CUDA_PATH)`r`nset PATH=$($cu.CUDA_PATH)\bin;%PATH%"
} else {
    $configure = 'cmake -S "{0}" -B "{1}" -G "Ninja Multi-Config" -D CMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake {2} || exit /b 2' -f $Src, $Build, ($flags -join ' ')
    $cudaEnv = ''
}
$cmd = Join-Path $env:TEMP 'nanollm-build-libs.cmd'
@"
@echo off
call "$($tc.VCVARSALL)" x64 || exit /b 1
$(if ($Vulkan -and $tc.GLSLC) { 'set VULKAN_SDK=' + (Split-Path -Parent (Split-Path -Parent $tc.GLSLC)) + "`r`nset PATH=" + (Split-Path -Parent $tc.GLSLC) + ';%PATH%' })
$cudaEnv
set PATH=$([IO.Path]::GetDirectoryName($tc.NINJA));$([IO.Path]::GetDirectoryName($tc.CMAKE));%PATH%
$configure
cmake --build "$Build" --config Release --target $($targets -join ' ') || exit /b 3
"@ | Set-Content -Encoding ascii $cmd

Write-Host "building: $($targets -join ' ')"
$t0 = Get-Date
$ErrorActionPreference = 'Continue'   # cmd/cmake write warnings to stderr (vcvarsall's vswhere); not errors
& cmd.exe /c $cmd
$ErrorActionPreference = 'Stop'
if ($LASTEXITCODE -ne 0) { throw "build failed (step $LASTEXITCODE; see above)" }
Write-Host ("built in {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds)

# --- collect ----------------------------------------------------------------
New-Item -ItemType Directory -Force $Out | Out-Null
$bin = Get-ChildItem -Recurse -File $Build -Include '*.dll' |
    Where-Object { $_.Name -match '^(llama|ggml-base|ggml|ggml-cpu-[0-9a-z_]+|ggml-vulkan|ggml-cuda|libomp|omp)\.dll$' -and $_.FullName -match 'Release' }
if ($Cuda) {
    # a lib directory of the CUDA build is the add-on: ggml-cuda.dll and the CUDA runtime beside it (the CPU backend
    # and the core stay the ones of the n7 libs)
    $bin = $bin | Where-Object { $_.Name -eq 'ggml-cuda.dll' }
    $rt = Join-Path $cu.CUDA_PATH 'bin\x64'
    if (-not (Test-Path $rt)) { $rt = Join-Path $cu.CUDA_PATH 'bin' }
    foreach ($n in 'cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll') {
        Get-ChildItem -File (Join-Path $rt $n) | ForEach-Object { Copy-Item -Force $_.FullName $Out }
    }
}
foreach ($f in $bin) { Copy-Item -Force $f.FullName $Out }
if ($vsLlvm) { Copy-Item -Force (Join-Path $vsLlvm 'bin\libomp.dll') $Out }
if ($Tools -or $Tests) {
    # the tools with every DLL of the build beside them (llama-common among them)
    $td = Join-Path (Split-Path -Parent $Out) 'tools'
    New-Item -ItemType Directory -Force $td | Out-Null
    Get-ChildItem -Recurse -File $Build -Include '*.dll', 'llama-server.exe', 'llama-bench.exe', 'llama-perplexity.exe', 'test-backend-ops.exe', 'test-nano-batch-invariance.exe' |
        Where-Object { $_.FullName -match 'Release' -and $_.FullName -match '\\bin\\' } | ForEach-Object { Copy-Item -Force $_.FullName $td }
    if ($vsLlvm) { Copy-Item -Force (Join-Path $vsLlvm 'bin\libomp.dll') $td }
    if ($Cuda) { Get-ChildItem -File $Out -Filter '*64_*.dll' | ForEach-Object { Copy-Item -Force $_.FullName $td } }
    Write-Host "tools dir: $td"
}
Write-Host "lib dir: $Out"
Get-ChildItem $Out | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
Write-Host "commit: $commit"
