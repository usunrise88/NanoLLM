# Flash attention on or off on the CPU (n6 libraries, alderlake variant): llama-bench prefill with 20 threads and
# decode with 6 threads on the P-cores (mask 0x555), at several KV depths. The Vulkan DLL is not in the directory.
param([string]$Depths = '0,4096,8192', [int]$Reps = 1, [string]$Out = 'cpu-fa.txt')
$R = 'C:\Projects\runtimes'
$dir = "$R\bench-cpu-n6"
if (-not (Test-Path $dir)) {
    Copy-Item -Recurse "$R\bench-patched" $dir
    Get-ChildItem 'C:\Projects\nanollm\dist\windows_amd64\lib\*.dll' | Where-Object { $_.Name -notmatch 'vulkan|rpc' } |
        ForEach-Object { Copy-Item -Force $_.FullName $dir }
}
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$outf = Join-Path $PSScriptRoot "raw\$Out"
foreach ($case in @(@('pp-t20', @('-p', '512', '-n', '0', '-t', '20'), 0), @('tg-t6', @('-p', '0', '-n', '32', '-t', '6'), 0x555))) {
    $name, $extra, $mask = $case
    $tmp = Join-Path $PSScriptRoot "raw\cpu-fa-$name.md"
    $a = @('-m', $m, '-ngl', '0', '-fa', '0,1', '-d', $Depths, '-r', "$Reps", '-o', 'md') + $extra
    $p = Start-Process -FilePath "$dir\llama-bench.exe" -ArgumentList $a -WorkingDirectory $dir -PassThru -NoNewWindow `
        -RedirectStandardOutput $tmp -RedirectStandardError "$tmp.err"
    if ($mask) { try { $p.ProcessorAffinity = [IntPtr]$mask } catch {} }
    if (-not $p.WaitForExit(3600000)) { $p.Kill(); Add-Content $outf "$name : TIMEOUT"; continue }
    Get-Content $tmp | Where-Object { $_ -match '\|\s*(pp|tg)\d+' } | ForEach-Object {
        $f = ($_ -replace '[^\x20-\x7E]', '') -split '\|'
        '{0,-7} fa={1} {2,-14} {3}' -f $name, $f[-4].Trim(), $f[-3].Trim(), $f[-2].Trim()
    } | Add-Content $outf
    Start-Sleep -Seconds 10
}
'DONE' | Add-Content $outf
