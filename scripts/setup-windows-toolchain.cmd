@echo off
rem Double-click or run from any prompt: starts the toolchain installer with
rem the execution policy out of the way; the script asks for administrator
rem rights itself. Options are passed on (-DryRun, -NoPause, -SkipVisualStudio).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-windows-toolchain.ps1" %*
