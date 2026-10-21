#Requires -Version 5.1
<#
.SYNOPSIS
  The kernel harnesses of run.sh on Windows: Q4_0 8x8 GEMV/GEMM, old against new, bit-exactness and timing.

.DESCRIPTION
  .\scripts\kernel-bench\run-windows.ps1 check            # bit-exactness only (random and extreme data)
  .\scripts\kernel-bench\run-windows.ps1 gemv [-Cpu 4]    # decode kernels, one thread pinned to a logical CPU
  .\scripts\kernel-bench\run-windows.ps1 gemm [-Cpu 4]    # prefill kernels
  .\scripts\kernel-bench\run-windows.ps1 multi [-Cpu 4]   # GEMV with 1 to 7 rows of the activation in one call

  The generated includes (old kernels from the pinned upstream file, "real" from the patched tree in
  .cache\llama.cpp\src) come from gen.sh, run through Git's bash. The harness is compiled with VS's clang++ in the
  environment of vcvarsall, with the flags of ggml's alderlake variant (-Vnni:$false gives the haswell flags).
  CPU 0..11 are the threads of the six P-cores of an i7-12700H; even numbers are the first thread of a core.
#>
param(
    [Parameter(Position = 0)][ValidateSet('check', 'gemv', 'gemm', 'multi')][string]$Mode = 'check',
    [int]$Cpu = 4,
    [bool]$Vnni = $true,
    [string]$Src
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root
$tc = @{}
foreach ($l in Get-Content (Join-Path $root '.cache\toolchain-setup\toolchain.env')) { if ($l -match '^([A-Z_0-9]+)=(.*)$') { $tc[$Matches[1]] = $Matches[2] } }
if (-not $Src) { $Src = Join-Path $root '.cache\llama.cpp\src' }
$gen = Join-Path $root '.cache\kernel-bench'

$bash = 'C:\Program Files\Git\bin\bash.exe'
$env:GEN = ($gen -replace '\\', '/')
$env:LLAMA_SRC = ($Src -replace '\\', '/')
& $bash scripts/kernel-bench/gen.sh
if ($LASTEXITCODE -ne 0) { throw 'gen.sh failed' }

$clangxx = Join-Path (Split-Path -Parent $tc.CLANG_CL_VS) 'clang++.exe'
$flags = '-O3 -DNDEBUG -std=c++17 -mavx2 -mfma -mf16c -mbmi2' + $(if ($Vnni) { ' -mavxvnni' } else { '' })
$here = Join-Path $root 'scripts\kernel-bench'
$cmd = Join-Path $env:TEMP 'nanollm-kb-build.cmd'
@"
@echo off
call "$($tc.VCVARSALL)" x64 || exit /b 1
"$clangxx" $flags -Wno-everything -I"$gen" -I"$here" -o "$gen\kb.exe" "$here\kb.cpp" || exit /b 2
"$clangxx" $flags -Wno-everything -I"$gen" -I"$here" -o "$gen\kg.exe" "$here\kg.cpp" || exit /b 3
"@ | Set-Content -Encoding ascii $cmd
& cmd.exe /c $cmd
if ($LASTEXITCODE -ne 0) { throw "compile failed (step $LASTEXITCODE)" }

function Run($exe, $argv) {
    $sp = @{ FilePath = $exe; NoNewWindow = $true; PassThru = $true }
    if ($argv) { $sp.ArgumentList = $argv }
    $p = Start-Process @sp
    try { $p.ProcessorAffinity = [IntPtr]([int64]1 -shl $Cpu) } catch {}
    $p.WaitForExit()
    $p.ExitCode
}
switch ($Mode) {
    'check' { $a = Run "$gen\kb.exe" 'check'; $b = Run "$gen\kg.exe" 'check'; if ($a -ne 0 -or $b -ne 0) { exit 1 } }
    'gemv'  { [void](Run "$gen\kb.exe" '') }
    'multi' { [void](Run "$gen\kb.exe" 'multi') }
    'gemm'  { [void](Run "$gen\kg.exe" '') }
}
