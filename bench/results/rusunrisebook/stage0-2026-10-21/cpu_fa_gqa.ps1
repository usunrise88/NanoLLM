# The grouped-query flash attention of the CPU (patch 0008 under test) against the n6 kernel: llama-bench with -fa 1,
# 6 threads on the P-cores, decode (tg32) and the MTP verification shape (pp3) at several depths, the two DLLs in
# alternating invocations. One line per run in raw\<Out>.
param([string]$New = 'C:\Projects\nanollm\.cache\llama.cpp\out-fagqa\lib\ggml-cpu-alderlake.dll',
      [string]$Old = 'C:\Projects\nanollm\dist\windows_amd64\lib\ggml-cpu-alderlake.dll',
      [string]$Depths = '0,4096,8192', [int]$Rounds = 2, [string]$Tests = 'tg', [string]$Fa = '1', [string]$Out = 'cpu-fa-gqa.txt')
$R = 'C:\Projects\runtimes'
$dir = "$R\bench-cpu-gqa"
if (-not (Test-Path $dir)) { Copy-Item -Recurse "$R\bench-cpu-n6" $dir }
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$outf = Join-Path $PSScriptRoot "raw\$Out"
$shape = @{ 'tg' = @('-p', '0', '-n', '32'); 'pp3' = @('-p', '3', '-n', '0') }
for ($i = 1; $i -le $Rounds; $i++) {
    foreach ($c in @(@('old', $Old), @('new', $New))) {
        Copy-Item -Force $c[1] "$dir\ggml-cpu-alderlake.dll"
        foreach ($t in ($Tests -split ',')) {
            $tmp = Join-Path $env:TEMP 'cpu-fa-gqa.md'
            $a = @('-m', $m, '-ngl', '0', '-t', '6', '-fa', $Fa, '-d', $Depths, '-r', '2', '-o', 'md') + $shape[$t]
            $p = Start-Process -FilePath "$dir\llama-bench.exe" -ArgumentList $a -WorkingDirectory $dir -PassThru -NoNewWindow `
                -RedirectStandardOutput $tmp -RedirectStandardError "$tmp.err"
            try { $p.ProcessorAffinity = [IntPtr]0x555 } catch {}
            $p.WaitForExit()
            $rows = Get-Content $tmp | Where-Object { $_ -match '\|\s*(pp|tg)\d+' } | ForEach-Object { $f = ($_ -split '\|'); '{0}={1}' -f ($f[-3].Trim() -replace ' ', ''), ($f[-2].Trim() -split ' ')[0] }
            $err = Get-Content "$tmp.err" | Where-Object { $_ -match 'error|abort|assert' } | Select-Object -First 1
            $line = 'round {0} {1} fa={2} {3} {4}' -f $i, $c[0], $Fa, ($rows -join '  '), $err
            $line; Add-Content $outf $line
            Start-Sleep -Seconds 5
        }
    }
}
