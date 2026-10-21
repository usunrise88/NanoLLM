@echo off
rem Double-click or run from any prompt: starts the CUDA Toolkit installer with
rem the execution policy out of the way; the script asks for administrator
rem rights itself. Options are passed on (-DryRun, -NoPause, -Version 13.4).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-windows-cuda.ps1" %*
